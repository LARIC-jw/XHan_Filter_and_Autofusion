"""
Statistical post-filter and output module.

3. Local robust ensemble post-filter: repeated-subsampling robust local linear
   (LOESS-type, tricube x bisquare) fits of WSE versus time. Attribute anchors
   are re-checked leave-one-out; attribute-rejected candidates are then tested
   against the cleaned anchors and restored when the consensus is high enough.
4. Start-of-series backtrack protection: before the first finally retained
   observation, rows that passed the attribute filters are restored.

Output helpers write the per-lake CSV (original columns + filter_flag), the
per-lake text report, the combined CSV, and the batch summary.

"""
from __future__ import annotations

import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

from XHanFilter_v2_attribute_filters import EPOCH_2000

EPS = np.finfo(float).eps


# -----------------------------------------------------------------------------
# Robust statistics
# -----------------------------------------------------------------------------
def robust_mad_scale(values):
    """MAD scaled to be consistent with a Gaussian standard deviation."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def rcond_2x2(a):
    """Reciprocal 1-norm condition number of a 2x2 matrix (0 when singular)."""
    a00, a01, a10, a11 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
    det = a00 * a11 - a01 * a10
    if not np.isfinite(det) or det == 0:
        return 0.0
    norm_a = max(abs(a00) + abs(a10), abs(a01) + abs(a11))
    norm_inv = max(abs(a11) + abs(a10), abs(a01) + abs(a00)) / abs(det)
    if norm_a == 0 or not np.isfinite(norm_inv):
        return 0.0
    return 1.0 / (norm_a * norm_inv)


def robust_local_linear(x, y):
    """Robust local linear LOESS fit (tricube x bisquare) predicting y at x = 0.

    Returns (prediction, beta, x_scale, ok). beta applies to x / x_scale.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    prediction, beta, x_scale, ok = np.nan, np.array([np.nan, np.nan]), np.nan, False

    if y.size < 3:
        return prediction, beta, x_scale, ok
    x_scale = np.max(np.abs(x))
    if not np.isfinite(x_scale) or x_scale <= EPS:
        return prediction, beta, x_scale, ok
    xn = x / x_scale

    # Tricube distance weights; the bandwidth is slightly enlarged so the farthest point is not exactly 0.
    bandwidth = np.max(np.abs(xn)) * 1.001
    base_weight = (1 - np.minimum(np.abs(xn) / bandwidth, 1) ** 3) ** 3
    robust_weight = np.ones_like(y)
    design = np.column_stack([np.ones_like(xn), xn])

    for _ in range(5):
        weight = base_weight * robust_weight
        if np.sum(weight > EPS) < 2:
            return prediction, beta, x_scale, ok
        normal_matrix = design.T @ (design * weight[:, None])
        if rcond_2x2(normal_matrix) < 1e-12:
            return prediction, beta, x_scale, ok
        beta = np.linalg.solve(normal_matrix, design.T @ (weight * y))
        residual = y - design @ beta
        residual_scale = robust_mad_scale(residual)
        if not np.isfinite(residual_scale) or residual_scale <= EPS:
            break
        u = residual / (6 * residual_scale)
        robust_weight = np.zeros_like(u)
        inside = np.abs(u) < 1
        robust_weight[inside] = (1 - u[inside] ** 2) ** 2

    prediction = beta[0]
    ok = bool(np.isfinite(prediction) and np.all(np.isfinite(beta)))
    return prediction, beta, x_scale, ok


# -----------------------------------------------------------------------------
# Step 3: ensemble consensus evaluation
# -----------------------------------------------------------------------------
class EnsembleEvaluation:
    """Per-observation outputs of evaluate_by_ensemble()."""

    def __init__(self, n):
        self.can_judge = np.zeros(n, dtype=bool)
        self.support_rate = np.full(n, np.nan)
        self.prediction = np.full(n, np.nan)
        self.prediction_scale = np.full(n, np.nan)
        self.local_scale = np.full(n, np.nan)
        self.standardized_residual = np.full(n, np.nan)
        self.valid_iterations = np.zeros(n, dtype=int)
        self.anchor_count = np.zeros(n, dtype=int)
        self.window_expanded = np.zeros(n, dtype=bool)


def evaluate_by_ensemble(obs_days, wse, reference_mask, target_idx, cfg, rng):
    """Ensemble LOESS consensus test of each target observation against reference anchors."""
    half_window = cfg["local_half_window_days"]
    min_anchor = cfg["local_min_anchor_count"]
    n_iterations = cfg["ensemble_iterations"]
    sample_ratio = cfg["ensemble_sample_ratio"]
    scale_coeff = cfg["local_scale_coeff"]
    scale_floor = cfg["local_scale_floor"]

    n = len(wse)
    result = EnsembleEvaluation(n)
    valid_reference = reference_mask & np.isfinite(obs_days) & np.isfinite(wse)
    reference_idx = np.flatnonzero(valid_reference)
    min_valid_iterations = max(1, math.ceil(n_iterations * cfg["min_valid_fit_ratio"]))

    for idx in target_idx:
        if not np.isfinite(obs_days[idx]) or not np.isfinite(wse[idx]):
            continue

        anchor_idx = reference_idx[reference_idx != idx]
        if len(anchor_idx) < min_anchor:
            result.anchor_count[idx] = len(anchor_idx)
            continue

        distance_days = np.abs(obs_days[anchor_idx] - obs_days[idx])
        local_idx = anchor_idx[distance_days <= half_window]
        if len(local_idx) < min_anchor:
            order = np.argsort(distance_days, kind="stable")
            local_idx = anchor_idx[order[:min_anchor]]
            result.window_expanded[idx] = True

        n_local = len(local_idx)
        result.anchor_count[idx] = n_local
        n_sample = min(n_local, max(3, math.ceil(sample_ratio * n_local)))
        local_x = obs_days[local_idx] - obs_days[idx]
        local_y = wse[local_idx]

        predictions = np.full(n_iterations, np.nan)
        scales = np.full(n_iterations, np.nan)
        supported = np.zeros(n_iterations, dtype=bool)
        valid_fit = np.zeros(n_iterations, dtype=bool)

        for b in range(n_iterations):
            sampled_pos = rng.permutation(n_local)[:n_sample]
            prediction, beta, x_scale, ok = robust_local_linear(local_x[sampled_pos], local_y[sampled_pos])
            if not ok:
                continue
            fitted_local = beta[0] + beta[1] * (local_x / x_scale)
            mad = robust_mad_scale(local_y - fitted_local)
            scale_b = max(mad, scale_floor) if np.isfinite(mad) else scale_floor
            predictions[b] = prediction
            scales[b] = scale_b
            supported[b] = abs(wse[idx] - prediction) <= scale_coeff * scale_b
            valid_fit[b] = True

        n_valid = int(valid_fit.sum())
        result.valid_iterations[idx] = n_valid
        if n_valid < min_valid_iterations:
            continue

        result.can_judge[idx] = True
        result.support_rate[idx] = supported[valid_fit].mean()
        result.prediction[idx] = np.median(predictions[valid_fit])
        result.prediction_scale[idx] = robust_mad_scale(predictions[valid_fit])
        result.local_scale[idx] = np.median(scales[valid_fit])
        combined_scale = math.sqrt(result.local_scale[idx] ** 2 + result.prediction_scale[idx] ** 2)
        result.standardized_residual[idx] = (
            abs(wse[idx] - result.prediction[idx]) / max(combined_scale, scale_floor)
        )
    return result


# -----------------------------------------------------------------------------
# Entry point: steps 3-4, final reasons, mapping back to the raw table
# -----------------------------------------------------------------------------
def apply_statistical_postfilter(attr, cfg, rng):
    """Run steps 3-4 on the output of apply_attribute_filters() and finalize the lake result.

    Returns a dict containing final_keep_raw (bool per raw row), final_reason_raw,
    the full statistics, and the arrays needed by write_report().
    """
    obs_days = attr["obs_days_pre"]
    wse = attr["wse_pre"]
    combined_mask = attr["combined_mask"]
    n_total = len(obs_days)
    stats = dict(attr["stats"])

    # --- Step 3: local robust ensemble post-filter ---
    initial_anchor_mask = ~combined_mask
    candidate_mask = combined_mask.copy()
    anchor_eval = EnsembleEvaluation(n_total)
    candidate_eval = EnsembleEvaluation(n_total)
    anchor_post_removed_mask = np.zeros(n_total, dtype=bool)
    candidate_restored_mask = np.zeros(n_total, dtype=bool)

    if cfg["postfilter_enable"]:
        # Leave-one-out re-check of every attribute anchor against all anchors.
        anchor_eval = evaluate_by_ensemble(obs_days, wse, initial_anchor_mask,
                                           np.flatnonzero(initial_anchor_mask), cfg, rng)
        anchor_post_removed_mask = (initial_anchor_mask & anchor_eval.can_judge
                                    & (anchor_eval.support_rate < cfg["consensus_threshold"]))
        # Anchors that cannot be judged are kept. Candidates are then evaluated
        # against the cleaned anchor set.
        clean_anchor_mask = initial_anchor_mask & ~anchor_post_removed_mask
        candidate_eval = evaluate_by_ensemble(obs_days, wse, clean_anchor_mask,
                                              np.flatnonzero(candidate_mask), cfg, rng)
        candidate_restored_mask = (candidate_mask & candidate_eval.can_judge
                                   & (candidate_eval.support_rate >= cfg["consensus_threshold"]))
    else:
        clean_anchor_mask = initial_anchor_mask

    final_keep_pre = clean_anchor_mask | candidate_restored_mask

    # --- Step 4: start-of-series backtrack protection ---
    start_backtrack_restored_mask = np.zeros(n_total, dtype=bool)
    if cfg["start_backtrack_enable"]:
        valid_final = final_keep_pre & np.isfinite(obs_days)
        if valid_final.any():
            first_kept_time = np.min(obs_days[valid_final])
            with np.errstate(invalid="ignore"):
                before_first = np.isfinite(obs_days) & (obs_days < first_kept_time)
            start_backtrack_restored_mask = before_first & initial_anchor_mask & ~final_keep_pre
            final_keep_pre |= start_backtrack_restored_mask

    stats.update({
        "n_initial_anchor": int(initial_anchor_mask.sum()),
        "n_candidate": int(candidate_mask.sum()),
        "n_anchor_post_removed": int(anchor_post_removed_mask.sum()),
        "n_candidate_restored": int(candidate_restored_mask.sum()),
        "n_start_backtrack_restored": int(start_backtrack_restored_mask.sum()),
        "n_anchor_unjudged": int(np.sum(initial_anchor_mask & ~anchor_eval.can_judge)) if cfg["postfilter_enable"] else 0,
        "n_candidate_unjudged": int(np.sum(candidate_mask & ~candidate_eval.can_judge)) if cfg["postfilter_enable"] else 0,
    })
    print(f"Post-filter: anchors {stats['n_initial_anchor']}, anchors removed {stats['n_anchor_post_removed']}, "
          f"candidates restored {stats['n_candidate_restored']}, backtrack restored {stats['n_start_backtrack_restored']}.")

    # --- Final row-level reasons ---
    final_reason_pre = list(attr["reason_pre"])
    for k in range(n_total):
        if anchor_post_removed_mask[k]:
            final_reason_pre[k] = (f"post-filter: local robust fit not supported "
                                   f"(anchor re-check, consensus = {anchor_eval.support_rate[k]:.3f})")
        elif candidate_mask[k] and not candidate_restored_mask[k]:
            if candidate_eval.can_judge[k]:
                post_reason = (f"post-filter: local robust fit not supported "
                               f"(consensus = {candidate_eval.support_rate[k]:.3f})")
            else:
                post_reason = (f"post-filter: undecidable (available anchors = "
                               f"{candidate_eval.anchor_count[k]}, required = {cfg['local_min_anchor_count']})")
            final_reason_pre[k] = post_reason if not final_reason_pre[k] else final_reason_pre[k] + " + " + post_reason

    # --- Map back to the raw table ---
    pre_kept_raw_idx = attr["pre_kept_raw_idx"]
    final_keep_raw = np.zeros(attr["n_raw"], dtype=bool)
    final_keep_raw[pre_kept_raw_idx[final_keep_pre]] = True
    final_reason_raw = list(attr["pre_reason_raw"])
    for k in range(n_total):
        final_reason_raw[pre_kept_raw_idx[k]] = "" if final_keep_pre[k] else final_reason_pre[k]

    stats["n_final"] = int(final_keep_raw.sum())
    stats["n_final_removed"] = attr["n_raw"] - stats["n_final"]
    print(f"Done: retained {stats['n_final']} rows, removed {stats['n_final_removed']} rows.")

    return {
        "stats": stats,
        "final_keep_raw": final_keep_raw,
        "final_reason_raw": final_reason_raw,
        "pre_kept_raw_idx": pre_kept_raw_idx,
        "obs_days_pre": obs_days,
        "wse_raw": attr["wse_raw"],
        "area_raw": attr["area_raw"],
        "wse_std_raw": attr["wse_std_raw"],
        "reason_pre": attr["reason_pre"],
        "candidate_restored_mask": candidate_restored_mask,
        "start_backtrack_restored_mask": start_backtrack_restored_mask,
        "candidate_eval": candidate_eval,
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "lake", "raw", "invalid", "h72_removed", "anchors", "anchors_removed",
    "candidates_restored", "backtrack_restored", "final_removed", "final_retained",
]
SUMMARY_HEADERS = ["raw", "invalid", "72h_rm", "anchors", "anc_rm", "cand_rst", "bt_rst", "removed", "retained"]


def format_days(day_value, fmt="%Y/%m/%d %H:%M:%S"):
    """Format float days since 2000-01-01 UTC as text ("NaT" when invalid)."""
    if not np.isfinite(day_value):
        return "NaT"
    return (EPOCH_2000 + pd.Timedelta(days=float(day_value))).strftime(fmt)


def save_lake_csv(df_lake, res, path):
    """Write the per-lake CSV: original columns + filter_flag (1 = retained, 0 = removed) as the last column."""
    df_out = df_lake.copy()
    df_out["filter_flag"] = res["final_keep_raw"].astype(int)
    df_out.to_csv(path, index=False)
    return df_out


def summary_row(out_stem, stats):
    """One batch-summary row in SUMMARY_COLUMNS order."""
    s = stats
    return [
        out_stem, s["n_raw"], s["n_invalid"], s["n_pre_removed"], s["n_initial_anchor"],
        s["n_anchor_post_removed"], s["n_candidate_restored"], s["n_start_backtrack_restored"],
        s["n_final_removed"], s["n_final"],
    ]


def write_batch_outputs(output_parts, summary_rows, work_dir, script_version, label):
    """Write the combined observation table and the batch summary; print the summary table."""
    combined_csv = os.path.join(work_dir, f"df_LakeSP_XHanFilter_{script_version}_LakeSP_{label}.csv")
    combined = pd.concat(output_parts, ignore_index=True) if output_parts else pd.DataFrame()
    combined.to_csv(combined_csv, index=False)

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_csv = os.path.join(work_dir, f"batch_summary_XHanFilter_{script_version}_LakeSP_{label}.csv")
    summary.to_csv(summary_csv, index=False)

    if len(summary) > 1:
        print("\n" + "=" * 64)
        print(f"  Batch summary ({len(summary)} lakes)")
        print("=" * 64)
        print(f"{'lake':<32} " + " ".join(f"{h:>8}" for h in SUMMARY_HEADERS))
        for row in summary_rows:
            print(f"{row[0][:32]:<32} " + " ".join(f"{v:>8d}" for v in row[1:]))
        totals = summary.iloc[:, 1:].sum()
        print(f"{'Total':<32} " + " ".join(f"{int(v):>8d}" for v in totals))

    print(f"\nSaved combined table: {combined_csv}")
    print(f"Saved batch summary : {summary_csv}")
    print(f"Processed lakes: {len(summary)}")
    return combined, summary


def write_report(path, lake_id, source_label, df_raw, res, cfg):
    """Write the per-lake text report (parameters, statistics, restored and removed rows)."""
    s = res["stats"]
    sep = "=" * 64 + "\n"
    thin = "-" * 64 + "\n"
    with open(path, "w", encoding="utf-8") as fid:
        fid.write(sep + "  SWOT LakeSP outlier filter report (XHanFilter v2)\n" + sep)
        fid.write(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fid.write(f"Lake ID   : {lake_id}\n")
        fid.write(f"Source    : {source_label}\n")
        fid.write(thin + "Workflow and parameters:\n")
        fid.write("  [Step 0] Invalid-record screening\n")
        fid.write("    Rule                    = rows without a valid time or WSE are removed\n")
        fid.write("  [Step 1] 72 h nearby-observation selection\n")
        fid.write(f"    Maximum window          = {cfg['nearby_hours']:.1f} h\n")
        fid.write(f"    Quality field order     = {', '.join(cfg['quality_fields'])}\n")
        fid.write("    Rule                    = minimum rank sum within group; ties broken in the order above\n")
        fid.write("  [Step 2] Flag (attribute) rules (rows outside range removed; missing values kept)\n")
        for j, info in enumerate(s["rules"], start=1):
            fid.write(f"    Rule {j:<19}= {info['text']}\n")
            if info["kind"] == "area_total":
                fid.write(f"      area_total median     = {info['median_area']:.4f}\n")
                fid.write(f"      p_ref_area reference  = {info['p_ref_area_val']:.4f}\n")
                fid.write(f"      Max_A                 = {info['max_a']:.4f}\n")
                fid.write(f"      Retained range        = {info['range']}  (bounds x Max_A)\n")
            elif info["kind"] == "overlap":
                fid.write(f"      Retained when         = exactly one overlap value satisfies {info['range']}\n")
            else:
                fid.write(f"      Retained range        = {info['range']}\n")
        fid.write("  [Step 3] Local robust ensemble post-filter\n")
        fid.write(f"    Enabled                 = {cfg['postfilter_enable']}\n")
        fid.write(f"    Local window            = +/- {cfg['local_half_window_days']:.1f} days\n")
        fid.write(f"    Minimum anchors         = {cfg['local_min_anchor_count']} (window expanded to nearest anchors if fewer)\n")
        fid.write(f"    Sample ratio per round  = {cfg['ensemble_sample_ratio']:.2f}\n")
        fid.write(f"    Iterations              = {cfg['ensemble_iterations']}\n")
        fid.write(f"    Support criterion       = |residual| <= {cfg['local_scale_coeff']:.2f} x local robust scale\n")
        fid.write(f"    Consensus threshold     = {cfg['consensus_threshold']:.2f}\n")
        fid.write(f"    Local scale floor       = {cfg['local_scale_floor']:.3f} m\n")
        fid.write("  [Step 4] Start-of-series backtrack protection\n")
        fid.write(f"    Enabled                 = {cfg['start_backtrack_enable']}\n")
        fid.write("    Rule                    = before the first final retained point, restore rows passing steps 1-2\n")
        fid.write(thin + "Statistics:\n")
        fid.write(f"  Raw rows                        = {s['n_raw']}\n")
        fid.write(f"  Invalid time/WSE rows           = {s['n_invalid']}\n")
        fid.write(f"  72 h selection removed          = {s['n_pre_removed']}\n")
        fid.write(f"  Rows after preprocessing        = {s['n_after_preprocess']}\n")
        for info in s["rules"]:
            fid.write(f"  Rule hits {info['text']:<22}= {info['n_hit']} (only reason: {info['n_only']})\n")
        fid.write(f"  Initial attribute anchors       = {s['n_initial_anchor']}\n")
        fid.write(f"  Attribute-rejected candidates   = {s['n_candidate']}\n")
        fid.write(f"  Anchors removed by post-filter  = {s['n_anchor_post_removed']}\n")
        fid.write(f"  Anchors kept (undecidable)      = {s['n_anchor_unjudged']}\n")
        fid.write(f"  Candidates restored             = {s['n_candidate_restored']}\n")
        fid.write(f"  Candidates undecidable          = {s['n_candidate_unjudged']}\n")
        fid.write(f"  Backtrack restored anchors      = {s['n_start_backtrack_restored']}\n")
        fid.write(f"  Final removed rows              = {s['n_final_removed']}\n")
        fid.write(f"  Final retained rows             = {s['n_final']}\n")
        fid.write(sep)

        pre_idx_all = res["pre_kept_raw_idx"]
        obs_days = res["obs_days_pre"]
        wse_raw = res["wse_raw"]
        cand_eval = res["candidate_eval"]

        # Restored rows use slash dates so plotting scripts do not mistake them for removals.
        restored = np.flatnonzero(res["candidate_restored_mask"])
        if restored.size:
            fid.write("\nCandidates restored by the post-filter (dates with slashes):\n")
            fid.write(f"{'No.':<6}  {'Datetime':<19}  {'wse':>12}  {'consensus':>10}  "
                      f"{'predicted':>12}  {'local scale':>12}  attribute reason\n")
            fid.write("------  -------------------  ------------  ----------  ------------  ------------  "
                      "----------------------------------------\n")
            for k, pre_idx in enumerate(restored, start=1):
                raw_idx = pre_idx_all[pre_idx]
                fid.write(f"{k:<6d}  {format_days(obs_days[pre_idx]):<19}  {wse_raw[raw_idx]:12.4f}  "
                          f"{cand_eval.support_rate[pre_idx]:10.3f}  {cand_eval.prediction[pre_idx]:12.4f}  "
                          f"{cand_eval.local_scale[pre_idx]:12.4f}  {res['reason_pre'][pre_idx]}\n")

        backtracked = np.flatnonzero(res["start_backtrack_restored_mask"])
        if backtracked.size:
            fid.write("\nAnchors restored by start-of-series backtrack (dates with slashes):\n")
            fid.write(f"{'No.':<6}  {'Datetime':<19}  {'wse':>12}  reason\n")
            fid.write("------  -------------------  ------------  ----------------------------------------\n")
            for k, pre_idx in enumerate(backtracked, start=1):
                raw_idx = pre_idx_all[pre_idx]
                fid.write(f"{k:<6d}  {format_days(obs_days[pre_idx]):<19}  {wse_raw[raw_idx]:12.4f}  "
                          f"passed 72 h selection and attribute filters; start-of-series protection\n")

        removed = np.flatnonzero(~res["final_keep_raw"])
        if removed.size:
            time_str = df_raw["time_str"].to_numpy() if "time_str" in df_raw.columns else [""] * len(df_raw)
            overlap = df_raw["overlap"].to_numpy() if "overlap" in df_raw.columns else [""] * len(df_raw)
            fid.write("\nFinally removed observations:\n")
            fid.write(f"{'No.':<6}  {'Datetime (time_str)':<24}  {'wse':>12}  {'area_total':>12}  "
                      f"{'wse_std':>12}  {'overlap':<20}  final reason\n")
            fid.write("------  ------------------------  ------------  ------------  ------------  "
                      "--------------------  ----------------------------------------\n")
            for k, idx in enumerate(removed, start=1):
                ov = str(overlap[idx])
                ov_disp = ov[:17] + "..." if len(ov) > 20 else ov
                fid.write(f"{k:<6d}  {str(time_str[idx]):<24}  {wse_raw[idx]:12.4f}  {res['area_raw'][idx]:12.4f}  "
                          f"{res['wse_std_raw'][idx]:12.4f}  {ov_disp:<20}  {res['final_reason_raw'][idx]}\n")
        else:
            fid.write("\nNo observation was removed; all rows are retained.\n")

        fid.write("\n" + sep + "  End of report\n" + sep)
