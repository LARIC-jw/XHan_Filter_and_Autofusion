"""
AutoFusion v1 - input/output module.

Input
- HALF and XHanFilter results: a combined CSV or a folder of per-lake CSVs.
  Required columns: lake_id, time_str, wse, filter_flag (1 = kept, 0 = removed);
  optional: wse_u, lake_name.

Output
- Per-lake fused WSE table (keep flags + compact diagnostics) and per-lake
  figure showing which segment comes from HALF and which from XHan.
"""
from __future__ import annotations

import glob
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["lake_id", "time_str", "wse", "filter_flag"]
POINT_COLUMNS = [
    "lake_id", "time_str", "Time", "wse", "wse_u", "half_flag", "xhan_flag", "fused_flag",
    "selected_method", "segment_id", "half_score", "xhan_score", "judge_zscore", "judge_confidence",
]

COLOR_HALF = (0.85, 0.33, 0.10)
COLOR_XHAN = (0.49, 0.18, 0.56)
COLOR_REMOVED = (0.45, 0.45, 0.45)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_naive(dates):
    """Timezone-naive UTC datetime Series."""
    t = pd.to_datetime(dates, errors="coerce")
    if not isinstance(t, pd.Series):
        t = pd.Series(t)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t


def sanitize_filename(name):
    out = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in str(name))
    return out.strip(" ._") or "Unknown"


def first_lake_name(series):
    if series is None or len(series) == 0:
        return "Unknown Lake"
    raw = series.dropna()
    if raw.empty:
        return "Unknown Lake"
    name = str(raw.iloc[0]).strip()
    if not name or name.lower() in ("nan", "none", "no_data"):
        return "Unknown Lake"
    return name.split(";")[0].strip()


# -----------------------------------------------------------------------------
# Filter results (HALF / XHanFilter)
# -----------------------------------------------------------------------------
def load_filter_results(path, label):
    """Load filter outputs from a combined CSV or a folder -> {lake_id: DataFrame}.

    In folder mode every CSV holding the required columns is read and exact
    duplicate rows are dropped (a folder may contain both per-lake and combined
    tables). Each lake table has Time, wse, wse_u, keep (bool), lake_name.
    """
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    parts = []
    for file in files:
        df = pd.read_csv(file, low_memory=False, dtype={"lake_id": str, "time_str": str})
        if any(c not in df.columns for c in REQUIRED_COLUMNS):
            continue
        cols = REQUIRED_COLUMNS + [c for c in ("wse_u", "lake_name") if c in df.columns]
        parts.append(df[cols])
    if not parts:
        raise FileNotFoundError(f"No {label} CSV with columns {REQUIRED_COLUMNS} found at {path}")
    df = pd.concat(parts, ignore_index=True).drop_duplicates()

    df["lake_id"] = df["lake_id"].astype(str).str.strip()
    df["time_str"] = df["time_str"].astype(str).str.strip()
    df["Time"] = to_naive(df["time_str"])
    df["wse"] = pd.to_numeric(df["wse"], errors="coerce")
    df["wse_u"] = pd.to_numeric(df["wse_u"], errors="coerce") if "wse_u" in df.columns else np.nan
    if "lake_name" not in df.columns:
        df["lake_name"] = ""
    df["keep"] = pd.to_numeric(df["filter_flag"], errors="coerce") == 1
    df = df.dropna(subset=["Time", "wse"]).sort_values("Time", kind="mergesort")
    print(f"[INFO] {label}: {df['lake_id'].nunique()} lakes, {len(df)} observations from {path}")
    return {lid: g.reset_index(drop=True) for lid, g in df.groupby("lake_id")}


def align_filter_results(half_df, xhan_df):
    """Align the two filter results on time_str (same SWOT observations).

    Rows with identical time_str are paired by occurrence order. Returns a table
    with Time, wse, wse_u, lake_name, half/xhan availability, keep flags.
    """
    def prepare(df, prefix):
        out = df[["time_str", "Time", "wse", "wse_u", "keep", "lake_name"]].copy()
        out = out.rename(columns={c: f"{prefix}_{c}" for c in ("Time", "wse", "wse_u", "keep", "lake_name")})
        out[f"{prefix}_available"] = True
        out["_occurrence"] = out.groupby("time_str", sort=False).cumcount()
        return out

    merged = prepare(half_df, "half").merge(prepare(xhan_df, "xhan"),
                                            on=["time_str", "_occurrence"], how="outer", validate="one_to_one")
    out = pd.DataFrame({
        "time_str": merged["time_str"],
        "Time": merged["half_Time"].combine_first(merged["xhan_Time"]),
        "wse": merged["half_wse"].combine_first(merged["xhan_wse"]),
        "wse_u": merged["half_wse_u"].combine_first(merged["xhan_wse_u"]),
        "lake_name": merged["half_lake_name"].combine_first(merged["xhan_lake_name"]),
        "half_available": merged["half_available"].fillna(False).astype(bool),
        "xhan_available": merged["xhan_available"].fillna(False).astype(bool),
    })
    out["half_keep"] = out["half_available"] & merged["half_keep"].fillna(False).astype(bool)
    out["xhan_keep"] = out["xhan_available"] & merged["xhan_keep"].fillna(False).astype(bool)
    return out.dropna(subset=["Time", "wse"]).sort_values("Time", kind="mergesort").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
def build_points_table(df, lake_id):
    """Compact per-observation diagnostics: one 1/0 flag per method (blank when the method has no row)."""
    out = df[["time_str", "Time", "wse", "wse_u", "selected_method", "segment_id"]].copy()
    out.insert(0, "lake_id", lake_id)
    for prefix in ("half", "xhan"):
        flag = pd.array(df[f"{prefix}_keep"].astype(int), dtype="Int64")
        flag[~df[f"{prefix}_available"].to_numpy()] = pd.NA
        out[f"{prefix}_flag"] = flag
    out["fused_flag"] = df["fused_keep"].astype(int)
    for col in ("half_score", "xhan_score", "judge_zscore", "judge_confidence"):
        out[col] = df[col].round(4)
    return out[POINT_COLUMNS]


def summary_row(lake_id, lake_name, df, segments):
    counts = df["selected_method"].value_counts()
    return {
        "lake_id": lake_id,
        "lake_name": lake_name,
        "n_observations": int(len(df)),
        "half_kept": int(df["half_keep"].sum()),
        "xhan_kept": int(df["xhan_keep"].sum()),
        "fused_kept": int(df["fused_keep"].sum()),
        "n_segments": int(len(segments)),
        "selected_from_half": int(counts.get("HALF", 0)),
        "selected_from_xhan": int(counts.get("XHan", 0)),
    }


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def _kept_ylim(df, keep_column):
    kept = df.loc[df[keep_column].to_numpy(dtype=bool), "wse"].to_numpy(dtype=float)
    kept = kept[np.isfinite(kept)]
    if kept.size == 0:
        return None
    span = float(kept.max() - kept.min())
    pad = max(0.05 * span, 0.10) if span > 0 else 0.50
    return float(kept.min()) - pad, float(kept.max()) + pad


def _draw_removed(ax, df, keep, ylim, color):
    removed = ~keep
    if not removed.any():
        return
    label = f"removed (n={int(removed.sum())})"
    if ylim is not None:
        wse = df.loc[removed, "wse"].to_numpy(dtype=float)
        n_clipped = int(np.sum(np.isfinite(wse) & ((wse < ylim[0]) | (wse > ylim[1]))))
        if n_clipped:
            label += f", {n_clipped} outside y-range"
    ax.plot(df.loc[removed, "Time"], df.loc[removed, "wse"], linestyle="none", marker="o",
            markerfacecolor="none", markeredgecolor=color, markersize=4.5, markeredgewidth=0.9,
            label=label, zorder=3)


def _shade_segments(ax, segments, pad_days, with_labels):
    pad = pd.Timedelta(days=pad_days)
    seen = set()
    for row in segments.itertuples(index=False):
        color = COLOR_HALF if row.selected_method == "HALF" else COLOR_XHAN
        label = f"segment from {row.selected_method}" if with_labels and row.selected_method not in seen else None
        seen.add(row.selected_method)
        ax.axvspan(pd.Timestamp(row.start_time) - pad, pd.Timestamp(row.end_time) + pad,
                   color=color, alpha=0.10, linewidth=0, label=label)


def _finish_panel(ax, title, ylim=None, ylabel="WSE (m)"):
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.45)
    # Legend in one row directly above the axes, title above the legend: neither covers data.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7.5, ncol=6, frameon=False,
              borderaxespad=0.2)
    ax.set_title(title, loc="left", fontsize=9, pad=17)


def plot_fusion(lake_id, lake_name, df, segments, path, show_scores=True, dpi=220):
    """HALF panel, XHan panel, fused panel coloured by source method, optional score panel."""
    n_panels = 4 if show_scores else 3
    ratios = [1.0, 1.0, 1.1, 0.6][:n_panels]
    fig, axes = plt.subplots(n_panels, 1, figsize=(13.5, 2.9 * n_panels), sharex=True,
                             gridspec_kw={"height_ratios": ratios})

    # Single-method panels.
    for ax, keep_column, name, color in ((axes[0], "half_keep", "HALF", COLOR_HALF),
                                         (axes[1], "xhan_keep", "XHan", COLOR_XHAN)):
        keep = df[keep_column].to_numpy(dtype=bool)
        ylim = _kept_ylim(df, keep_column)
        ax.plot(df.loc[keep, "Time"], df.loc[keep, "wse"], "-s", color=color, markersize=4.0, linewidth=1.2,
                label=f"kept (n={int(keep.sum())})", zorder=4)
        _draw_removed(ax, df, keep, ylim, color)
        _finish_panel(ax, f"{name} filter", ylim)

    # Fused panel: line through all kept points, markers coloured by the source method.
    ax = axes[2]
    gaps = df["Time"].diff().dt.total_seconds().div(86400.0)
    typical_days = float(gaps[gaps > 0].median()) if (gaps > 0).any() else 21.0
    _shade_segments(ax, segments, max(typical_days * 0.5, 1.0), with_labels=not show_scores)
    keep = df["fused_keep"].to_numpy(dtype=bool)
    ylim = _kept_ylim(df, "fused_keep")
    ax.plot(df.loc[keep, "Time"], df.loc[keep, "wse"], "-", color="0.25", linewidth=1.0, zorder=4)
    for method, color in (("HALF", COLOR_HALF), ("XHan", COLOR_XHAN)):
        sel = keep & (df["selected_method"].to_numpy() == method)
        if sel.any():
            ax.plot(df.loc[sel, "Time"], df.loc[sel, "wse"], linestyle="none", marker="s", color=color,
                    markersize=4.5, label=f"kept from {method} (n={int(sel.sum())})", zorder=5)
    _draw_removed(ax, df, keep, ylim, COLOR_REMOVED)
    _finish_panel(ax, "Fused (segment colour = source method)", ylim)

    if show_scores:
        ax = axes[3]
        _shade_segments(ax, segments, max(typical_days * 0.5, 1.0), with_labels=True)
        ax.plot(df["Time"], df["half_score"], color=COLOR_HALF, linewidth=1.2, label="HALF local cost")
        ax.plot(df["Time"], df["xhan_score"], color=COLOR_XHAN, linewidth=1.2, label="XHan local cost")
        _finish_panel(ax, "Smoothed local cost of each method (the lower one is selected)",
                      ylabel="Local cost")

    pad = pd.Timedelta(days=30)
    axes[-1].set_xlim(df["Time"].min() - pad, df["Time"].max() + pad)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    counts = df["selected_method"].value_counts()
    fig.suptitle(f"{lake_id}  {lake_name}\nobservations assigned: "
                 f"HALF={int(counts.get('HALF', 0))}, XHan={int(counts.get('XHan', 0))}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
