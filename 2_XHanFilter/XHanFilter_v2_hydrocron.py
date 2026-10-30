"""
This modude is using Hydrocron to download the SWOT LakeSP, same logic as HALF.
Please add reference to HALF also: https://doi.org/10.5281/zenodo.20782248

Hydrocron download of SWOT LakeSP time series (request logic follows
half_v1_run.py), local CSV reading, and splitting of tables into single-lake
tables by the lake_id column.
"""
from __future__ import annotations

import glob
import os
from io import StringIO

import numpy as np
import pandas as pd
import requests

# Hydrocron request settings (identical to half_v1_run.py).
HYDROCRON_URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
HYDROCRON_FIELDS = (
    "lake_id,reach_id,obs_id,overlap,n_overlap,time,time_tai,time_str,wse,"
    "wse_u,wse_r_u,wse_std,area_total,area_tot_u,area_detct,area_det_u,"
    "layovr_val,xtrk_dist,ds1_l,ds1_l_u,ds1_q,ds1_q_u,ds2_l,ds2_l_u,"
    "ds2_q,ds2_q_u,quality_f,dark_frac,ice_clim_f,ice_dyn_f,partial_f,"
    "xovr_cal_q,geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,"
    "dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,lake_name,p_res_id,p_lon,"
    "p_lat,p_ref_wse,p_ref_area,p_date_t0,p_ds_t0,p_storage,cycle_id,"
    "pass_id,continent_id,range_start_time,range_end_time,crid,geometry,"
    "PLD_version,collection_shortname"
)
VERSION_FILENAME_BY_COLLECTION = {
    "SWOT_L2_HR_LakeSP_2.0": "vC",
    "SWOT_L2_HR_LakeSP_D": "vD",
}



# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------
def read_lakesp_csv(source):
    """Read a LakeSP CSV (path or buffer) with every column kept as text."""
    return pd.read_csv(source, dtype=str, keep_default_na=False)


def query_hydrocron(feature_id, start_time, end_time, collection_shortname):
    """Download one PriorLake time series from Hydrocron.

    Returns a string-typed DataFrame, or None when the request fails or no
    usable table is returned.
    """
    output = "csv"
    enquiry_input = (
        HYDROCRON_URL
        + "feature=PriorLake"
        + "&feature_id=" + str(feature_id)
        + "&start_time=" + start_time
        + "&end_time=" + end_time
        + "&output=" + output
        + "&fields=" + HYDROCRON_FIELDS
        + "&collection_name=" + collection_shortname
    )

    try:
        response = requests.get(enquiry_input, timeout=120)
        response.raise_for_status()
        hydrocron_response = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Hydrocron request failed for lake {feature_id}: {exc}. Skipping.")
        return None

    if not isinstance(hydrocron_response, dict):
        print(f"Hydrocron returned an unexpected response for lake {feature_id}; skipping.")
        return None

    results = hydrocron_response.get("results")
    if not isinstance(results, dict):
        print(f"Hydrocron returned no result table for lake {feature_id}; skipping.")
        return None

    extracted_data = results.get(output, "")
    if not isinstance(extracted_data, str) or not extracted_data.strip():
        print(f"Hydrocron returned no LakeSP records for lake {feature_id}; skipping.")
        return None

    try:
        df = read_lakesp_csv(StringIO(extracted_data))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Hydrocron CSV could not be read for lake {feature_id}: {exc}. Skipping.")
        return None

    if df.empty:
        print(f"Hydrocron returned an empty LakeSP table for lake {feature_id}; skipping.")
        return None
    return df


def check_required_columns(df, label, required):
    """True when all required columns are present; otherwise print and return False."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"{label} is missing required columns {missing}; skipping.")
        return False
    return True


# -----------------------------------------------------------------------------
# Input iteration and per-lake splitting
# -----------------------------------------------------------------------------
def iter_input_tables(run):
    """Yield (source_label, table) for RUN_CONFIG["input_mode"].

    source_label is "hydrocron:<feature_id>" for downloads and the file path
    for local CSVs.
    """
    mode = run["input_mode"]
    if mode == "hydrocron":
        lake_ids = run["lake_ids"]
        if isinstance(lake_ids, (str, int, np.integer)):
            lake_ids = [lake_ids]
        lake_ids = list(dict.fromkeys(lake_ids))
        if not lake_ids:
            raise ValueError("RUN_CONFIG['lake_ids'] must contain at least one PLD lake ID.")
        for feature_id in lake_ids:
            print(f"\nQuerying Hydrocron for lake ID: {feature_id}")
            df = query_hydrocron(feature_id, run["start_time"], run["end_time"],
                                 run["collection_shortname"])
            if df is not None:
                yield f"hydrocron:{feature_id}", df
    elif mode == "folder":
        files = sorted(glob.glob(os.path.join(run["csv_folder"], "*.csv")))
        if not files:
            raise FileNotFoundError(f"No .csv file found in folder '{run['csv_folder']}'.")
        print(f"Folder mode: {len(files)} CSV files found.")
        for path in files:
            yield path, read_lakesp_csv(path)
    elif mode == "file":
        path = run["csv_filepath"]
        print(f"Single-file mode: {path}")
        yield path, read_lakesp_csv(path)
    else:
        raise ValueError("RUN_CONFIG['input_mode'] must be 'hydrocron', 'folder', or 'file'.")


def split_by_lake_id(df, fallback_id):
    """Split a table by lake_id (order of first appearance). Yields (lake_id, table)."""
    if "lake_id" not in df.columns:
        yield str(fallback_id), df.reset_index(drop=True)
        return
    ids = df["lake_id"].astype(str).str.strip()
    for lake_id in ids.unique():
        yield lake_id, df.loc[ids == lake_id].reset_index(drop=True)


def iter_lake_tables(run, required):
    """Yield (source_label, lake_id, out_stem, table) for every single-lake table.

    required is the list of columns the filter needs ("time" is optional, time_str
    is the fallback). out_stem is the base name used for the per-lake output files.
    """
    label = output_label(run)
    for source_label, table in iter_input_tables(run):
        if not check_required_columns(table, source_label, required):
            continue
        is_hydrocron = source_label.startswith("hydrocron:")
        source_stem = os.path.splitext(os.path.basename(source_label))[0]
        fallback_id = source_label.split(":", 1)[1] if is_hydrocron else source_stem
        lake_tables = list(split_by_lake_id(table, fallback_id))
        for lake_id, df_lake in lake_tables:
            if is_hydrocron:
                out_stem = f"lakeID_{lake_id}_LakeSP_{label}"
            elif len(lake_tables) == 1:
                out_stem = source_stem
            else:
                out_stem = f"{source_stem}_lakeID_{lake_id}"
            yield source_label, lake_id, out_stem, df_lake


def output_label(run):
    """LakeSP version label used in output file names ("vC"/"vD", or "local")."""
    if run["input_mode"] != "hydrocron":
        return "local"
    collection = run["collection_shortname"]
    if collection not in VERSION_FILENAME_BY_COLLECTION:
        raise ValueError(f"Unsupported collection_shortname={collection!r}. "
                         f"Expected one of: {', '.join(VERSION_FILENAME_BY_COLLECTION)}.")
    return VERSION_FILENAME_BY_COLLECTION[collection]
