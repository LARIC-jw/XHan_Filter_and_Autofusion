"""
At the beginning, the parallel filtering algorithm HALF can be found below: https://doi.org/10.5281/zenodo.20782248
-------------------------------------------------------------------------------------------------------------------

This script is XHan filter, below is the description of the filter:

Overall workflow:
-------------------------------------------------------------------------------------------------------------------
0. Invalid-record screening: rows without a valid time or WSE (missing or LakeSP fill value) are flagged as removed
   and excluded from later steps. In this script, the invalid-record screening is not implemented.
1. 72 h nearby-observation selection: observations falling in the same fixed 72 h window are reduced to the one
   with the best combined quality rank
   N.B. (wse_u -> layovr_val -> wse_std -> wse_r_u; smaller is better).
2. Attribute filters: you CAN use "area_total", "wse_std", or "overlap" or any flags else that fit for your case
3. Local robust ensemble post-filter: repeated-subsampling robust local linear (LOESS) fits of WSE versus time
   re-check the intial anchor points.
4. Start-of-series backtrack protection: before the first finally retained observation, rows that passed steps
   1-2 are restored.

Modules and functions dependency (Please KEEP them in the same directory as this script):
-------------------------------------------------------------------------------------------------------------------
XHanFilter_v2_hydrocron.py           : Hydrocron download, local CSV reading, split by lake_id, same with HALF
XHanFilter_v2_attribute_filters.py   : flag based filters to identify the intial anchor points
XHanFilter_v2_statistical_filters.py : robust trend fitting, ensemble decision and outputs

Input
-------------------------------------------------------------------------------------------------------------------
CHOOSE your input mode first: hydrocron OR folder you have OR single file
RUN_CONFIG["input_mode"]:
    "hydrocron" : query the NASA Hydrocron PriorLake time-series endpoint for
                  each PLD lake ID (request logic follows half_v1_run.py).
    "folder"    : all *.csv files in RUN_CONFIG["csv_folder"].
    "file"      : the single CSV given by RUN_CONFIG["csv_filepath"].
Every input table is split by its lake_id column (first Hydrocron column) and each lake is filtered independently.
**!! Please make sure the first column is ID !!**

Output (written to RUN_CONFIG["work_dir"])
-------------------------------------------------------------------------------------------------------------------
- One CSV per lake: all original columns plus a last column "filter_flag"
  (1 = retained, 0 = removed).
- One text report per lake with parameters, statistics, and row-level reasons.
- A combined CSV of all processed lakes and a batch summary CSV.

"""
# %%
from __future__ import annotations

import os
import numpy as np
from XHanFilter_v2_hydrocron import iter_lake_tables, output_label
from XHanFilter_v2_attribute_filters import apply_attribute_filters, required_columns
from XHanFilter_v2_statistical_filters import (
    apply_statistical_postfilter,
    save_lake_csv,
    write_report,
    summary_row,
    write_batch_outputs,
)


# =============================================================================
# User configuration
# =============================================================================
RUN_CONFIG = {
    # Input source: "hydrocron" | "folder" | "file".
    "input_mode": "hydrocron",

    # --- "hydrocron" mode ---
    # One or more PLD_IDs
    "lake_ids": [4620019453, 4620033793],
    # LakeSP collection short name:
    #   "SWOT_L2_HR_LakeSP_2.0" (Version C) OR "SWOT_L2_HR_LakeSP_D" (Version D).
    "collection_shortname": "SWOT_L2_HR_LakeSP_D",
    # UTC query window passed directly to Hydrocron.
    "start_time": "2023-07-21T05:33:45Z",
    "end_time": "2026-02-10T21:36:19Z",

    # --- "folder" / "file" mode (local CSVs with Hydrocron LakeSP columns) ---
    "csv_folder": r"",
    "csv_filepath": r"",

    # Output directory (created if missing).
    "work_dir": r"",
    # Label used in output file names.
    "script_version": "v2",
    # Write a per-lake text report: "yes" | "no".
    "write_lake_report": "yes",
}

FILTER_CONFIG = {
    # [Step 1] 72 h nearby-observation selection.
    "nearby_hours": 72,
    # Quality fields used for the combined rank and for tie-breaking (in order).
    "quality_fields": ["wse_u", "layovr_val", "wse_std", "wse_r_u"],

    # [Step 2] Flag (attribute) rules. Use one or more rules; rows whose value
    # falls outside the range are removed (missing values are never removed).
    # Any numeric LakeSP column can be used. Accepted forms:
    #   "field<b", "field<=b", "field>a", "field>=a", "a<field<b", "a<=field<=b"
    # Two fields have special semantics:
    #   area_total : bounds are multiples of Max_A = max(median(area_total), p_ref_area).
    #   overlap    : the semicolon-separated list must contain exactly one value in range.
    "attribute_rules": [
        #"0<=area_total<=2",
        "wse_std<1.5",
        "overlap>=10",
    ],

    # [Step 3] local robust ensemble post-filter.
    "postfilter_enable": True,
    "local_half_window_days": 45,     # +/- window around the target observation
    "local_min_anchor_count": 7,      # expand to the nearest N anchors when the window is too sparse
    "ensemble_iterations": 500,       # number of random sub-sampling rounds
    "ensemble_sample_ratio": 0.65,    # fraction of local anchors drawn per round
    "local_scale_coeff": 2.5,         # |residual| <= coeff * local robust scale counts as support
    "consensus_threshold": 0.80,      # minimum fraction of supporting rounds
    "local_scale_floor": 0.10,        # lower bound of the local scale (m)
    "min_valid_fit_ratio": 0.50,      # minimum fraction of successful fits to make a decision
    "random_seed": 20260824,

    # [Step 4] start-of-series backtrack protection.
    "start_backtrack_enable": True,
}

# %%
# =============================================================================
# Main function of the script:
# 1. Download data from Hydrocron OR read local CSV files (if you already have), same with HALF
# 2. XHan filter first step: using flags to filter & find the intial anchor points
# 3. XHan filter second step: using statistical filters to filter the data
# =============================================================================
def main(run=RUN_CONFIG, cfg=FILTER_CONFIG):
    work_dir = os.path.abspath(run["work_dir"])
    os.makedirs(work_dir, exist_ok=True)
    script_version = run["script_version"]
    label = output_label(run)

    # A single generator is seeded once, as in the MATLAB script.
    rng = np.random.default_rng(cfg["random_seed"])

    output_parts = []
    summary_rows = []

    # Columns needed by the configured rules (also validates the rule syntax up front).
    required = required_columns(cfg)

    # Download / read input tables and process each lake independently.
    for source_label, lake_id, out_stem, df_lake in iter_lake_tables(run, required):
        print(f"\n##########  Lake {lake_id}  ({source_label})  ##########")

        # Steps 0-2: attribute (flag) filters.
        attr = apply_attribute_filters(df_lake, cfg)

        # Steps 3-4: statistical post-filter and final decision per raw row.
        res = apply_statistical_postfilter(attr, cfg, rng)

        # Per-lake outputs.
        csv_path = os.path.join(work_dir, f"{out_stem}_XHanFilter_{script_version}.csv")
        df_out = save_lake_csv(df_lake, res, csv_path)
        print(f"Saved: {csv_path}")

        if run["write_lake_report"] == "yes":
            report_path = os.path.join(work_dir, f"{out_stem}_XHanFilter_{script_version}_Report.txt")
            write_report(report_path, lake_id, source_label, df_lake, res, cfg)
            print(f"Saved: {report_path}")

        output_parts.append(df_out)
        summary_rows.append(summary_row(out_stem, res["stats"]))

    # Combined observation-level table and batch summary.
    combined, summary = write_batch_outputs(output_parts, summary_rows, work_dir, script_version, label)
    return combined, summary


if __name__ == "__main__":
    main()
# %%
