"""
AutoFusion v1 - segment-wise fusion of HALF and XHanFilter results for SWOT
LakeSP water surface elevation (WSE) time series.

Both filters label the same SWOT observations as kept or removed. AutoFusion
decides, segment by segment, which filter's labels to use. WSE values are
never changed; only the keep/remove mask is fused.

Workflow (per lake)
-------------------
1. Align the HALF and XHanFilter tables on time_str.
2. Robust local judge: leave-neighbourhood-out robust polynomial fit of WSE
   versus time gives each observation an outlier z-score and a confidence.
3. Local cost per method: keeping likely outliers, removing likely valid
   points, roughness of the retained trajectory, and coverage loss.
4. Viterbi segment selection with a switch penalty; restarts at long gaps.

Modules (keep them in the same directory as this script)
--------------------------------------------------------
AutoFusion_v1_io.py     : loading filter results, alignment, tables, figure.
AutoFusion_v1_fusion.py : robust judge, method costs, Viterbi selection, segments.

Input
-----
RUN_CONFIG["half_results"] and RUN_CONFIG["xhan_results"]: a combined CSV or a
folder of per-lake CSVs with lake_id, time_str, wse, filter_flag (1 = kept,
0 = removed), optionally wse_u and lake_name. These are the standard outputs of
half_v1_run.py and XHanFilterLakeSP_v2_run.py.

Output (written to RUN_CONFIG["work_dir"])
------
- FusedWSE_ts_dignoise/<lake>_AutoFusion_<version>.csv : one row per observation
  with half_flag, xhan_flag, fused_flag (1/0), selected_method, segment_id, scores.
- figures/<lake>_AutoFusion_<version>.png : HALF, XHan and fused panels; the fused
  panel is coloured by the source method of each segment.
"""
# %%
from __future__ import annotations

import os

from AutoFusion_v1_io import (
    align_filter_results,
    build_points_table,
    first_lake_name,
    load_filter_results,
    plot_fusion,
    sanitize_filename,
    summary_row,
)
from AutoFusion_v1_fusion import fuse_lake


# =============================================================================
# User configuration
# =============================================================================
RUN_CONFIG = {
    # HALF output: combined CSV (df_Hydrocron_HALF_*.csv) or a folder of per-lake CSVs.
    "half_results": r"",
    # XHanFilter output: combined CSV (df_LakeSP_XHanFilter_*.csv) or the XHanFilter work_dir.
    "xhan_results": r"",
    # Lakes to process; empty list = all lakes present in both inputs.
    "lake_ids": [],
    # Output directory (created if missing).
    "work_dir": r"",
    "script_version": "v1",
    # Add the local-cost panel to the figure ("yes"/"no").
    "plot_score_panel": "yes",
    "figure_dpi": 220,
}

FUSION_CONFIG = {
    # Robust local judge.
    "judge_window_days": 240.0,         # base +/- window around each observation
    "judge_max_window_days": 540.0,     # never look further than this
    "judge_min_neighbors": 8,           # expand the window to reach this many neighbours
    "judge_max_neighbors": 30,          # cap on neighbours used in one fit
    "same_visit_exclude_days": 2.5,     # ignore observations from the same visit
    "absolute_scale_floor_m": 0.08,     # minimum robust scale (m)

    # Local method cost.
    "score_window_points": 17,          # smoothing window (observations) for costs and retention
    "min_local_retention": 0.35,        # retention below this is penalised
    "false_remove_weight": 0.30,        # weight of removing a likely valid point
    "trajectory_weight": 0.90,          # weight of the retained-trajectory roughness term
    "short_roughness_weight": 1.20,     # weight of the saw-tooth term
    "coverage_weight": 0.80,            # weight of the coverage term
    "max_keep_gap_days": 120.0,         # kept-point gaps longer than this are penalised

    # Segment selection.
    "switch_penalty": 2.0,              # cost of switching between HALF and XHan
    "natural_break_days": 180.0,        # data gaps longer than this restart the selection
}

# %%
# =============================================================================
# Main workflow
# =============================================================================
def main(run=RUN_CONFIG, cfg=FUSION_CONFIG):
    work_dir = os.path.abspath(run["work_dir"])
    table_dir = os.path.join(work_dir, "FusedWSE_ts_dignoise")
    figure_dir = os.path.join(work_dir, "figures")
    for path in (table_dir, figure_dir):
        os.makedirs(path, exist_ok=True)
    version = run["script_version"]

    half_tables = load_filter_results(run["half_results"], "HALF")
    xhan_tables = load_filter_results(run["xhan_results"], "XHan")
    common_ids = sorted(set(half_tables) & set(xhan_tables))
    lake_ids = [str(i) for i in run["lake_ids"]] or common_ids

    n_done = 0
    for lake_id in lake_ids:
        if lake_id not in half_tables or lake_id not in xhan_tables:
            print(f"**Warning** {lake_id}: missing HALF or XHan result, skipped")
            continue

        # Fusion.
        aligned = align_filter_results(half_tables[lake_id], xhan_tables[lake_id])
        points, segments = fuse_lake(aligned, cfg)
        lake_name = first_lake_name(points["lake_name"])

        # Outputs.
        stem = f"{lake_id}_{sanitize_filename(lake_name)}_AutoFusion_{version}"
        build_points_table(points, lake_id).to_csv(
            os.path.join(table_dir, f"{stem}.csv"), index=False, encoding="utf-8-sig")
        plot_fusion(lake_id, lake_name, points, segments, os.path.join(figure_dir, f"{stem}.png"),
                    show_scores=run["plot_score_panel"] == "yes", dpi=run["figure_dpi"])

        row = summary_row(lake_id, lake_name, points, segments)
        n_done += 1
        print(f"**Processing in progress** {lake_id} {lake_name}: HALF kept={row['half_kept']}, XHan kept={row['xhan_kept']}, "
              f"fused kept={row['fused_kept']}, segments={row['n_segments']} "
              f"(HALF {row['selected_from_half']} / XHan {row['selected_from_xhan']} obs)")

    print(f"**Finished** {n_done} lakes, outputs: {work_dir}")


if __name__ == "__main__":
    main()
# %%
