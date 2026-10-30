"""
Flag filter module.

Filters that use only the LakeSP attributes stored in the CSV:

0. Invalid-record screening: rows without a valid time or WSE (missing or
   LakeSP fill value) are removed and excluded from all later steps.
1. 72 h nearby-observation selection: observations within a fixed 72 h window
   are reduced to the one with the best combined quality rank
   (wse_u -> layovr_val -> wse_std -> wse_r_u; smaller is better).
2. Flag (attribute) rules on the remaining rows, given by the user as range
   strings such as "wse_u<0.1" or "0.01<wse_u<0.1". Rows outside the range are
   removed; missing values are never removed. 

   !!**N.B.: Two fields are special:**!!
   - area_total : bounds are multiples of Max_A = max(median(area_total), p_ref_area).
   - overlap    : the semicolon-separated list must contain exactly one value in range.

"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FILL_FLOAT = -999999999999
EPOCH_2000 = pd.Timestamp("2000-01-01 00:00:00", tz="UTC")
REASON_INVALID = "invalid time or WSE (missing or fill value)"
BASE_REQUIRED_COLUMNS = ["lake_id", "time_str", "wse"]


# -----------------------------------------------------------------------------
# Column helpers
# -----------------------------------------------------------------------------
def numeric_column(df, name):
    """Float array of a column; non-numeric entries and LakeSP fill values become NaN."""
    if name not in df.columns:
        return np.full(len(df), np.nan)
    values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    values[values == FILL_FLOAT] = np.nan
    return values


def parse_observation_time_days(df):
    """Observation time as float days since 2000-01-01 UTC (NaN when unavailable).

    The high-precision 'time' column (seconds since 2000-01-01 UTC) is preferred;
    'time_str' (ISO 8601) is used for rows without a valid 'time'.
    """
    days = np.full(len(df), np.nan)
    if "time" in df.columns:
        days = numeric_column(df, "time") / 86400.0
    if "time_str" in df.columns:
        pending = ~np.isfinite(days)
        if pending.any():
            raw = df["time_str"].to_numpy(dtype=str)[pending]
            parsed = pd.to_datetime(np.char.strip(raw), utc=True, errors="coerce", format="ISO8601")
            days[pending] = ((parsed - EPOCH_2000) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    return days


# -----------------------------------------------------------------------------
# Step 1: 72 h nearby-observation selection
# -----------------------------------------------------------------------------
def choose_quality_winner(quality):
    """Index (within the group) of the best observation.

    Each field is ranked ascending (NaN/Inf treated as worst); the smallest rank
    sum wins. Ties are broken by the field values in the given order and finally
    by group position, so the result is reproducible.
    """
    q = np.array(quality, dtype=float)
    q[~np.isfinite(q)] = np.inf
    m, p = q.shape
    ranks = np.ones((m, p))
    for j in range(p):
        for i in range(m):
            ranks[i, j] = 1 + np.sum(q[:, j] < q[i, j])
    rank_sum = ranks.sum(axis=1)
    keys = [rank_sum] + [q[:, j] for j in range(p)] + [np.arange(m)]
    order = np.lexsort(tuple(reversed(keys)))  # lexsort uses the last key as primary
    return int(order[0])


def select_best_within_hours(obs_days, quality, max_hours):
    """Fixed-window grouping: a group starts at its earliest observation and spans at most max_hours.

    Returns (keep_mask, reasons); rows with NaN time are never grouped.
    """
    n = len(obs_days)
    keep_mask = np.ones(n, dtype=bool)
    reasons = [""] * n
    window_days = max_hours / 24.0

    valid_idx = np.flatnonzero(np.isfinite(obs_days))
    sorted_idx = valid_idx[np.argsort(obs_days[valid_idx], kind="stable")]
    m = len(sorted_idx)

    pos = 0
    while pos < m:
        group_start = obs_days[sorted_idx[pos]]
        last = pos
        while last + 1 < m and obs_days[sorted_idx[last + 1]] <= group_start + window_days:
            last += 1
        group_idx = sorted_idx[pos:last + 1]
        if len(group_idx) > 1:
            winner = group_idx[choose_quality_winner(quality[group_idx])]
            for loser in group_idx[group_idx != winner]:
                keep_mask[loser] = False
                reasons[loser] = (
                    f"72 h nearby duplicate: not the best combined quality rank "
                    f"(group of {len(group_idx)} rows)"
                )
        pos = last + 1
    return keep_mask, reasons


# -----------------------------------------------------------------------------
# Step 2: flag (attribute) range rules
# -----------------------------------------------------------------------------
@dataclass
class AttributeRule:
    """One parsed range rule. A missing bound is None; *_inclusive selects <= versus <."""
    text: str
    field: str
    lower: float | None
    lower_inclusive: bool
    upper: float | None
    upper_inclusive: bool

    def describe(self):
        """Human-readable range, e.g. "0.01 < wse_u <= 0.1"."""
        parts = []
        if self.lower is not None:
            parts.append(f"{self.lower:g} {'<=' if self.lower_inclusive else '<'} ")
        parts.append(self.field)
        if self.upper is not None:
            parts.append(f" {'<=' if self.upper_inclusive else '<'} {self.upper:g}")
        return "".join(parts)

    def violated(self, values):
        """Boolean array: True where a finite value lies outside the range (NaN never violates)."""
        v = np.asarray(values, dtype=float)
        bad = np.zeros(v.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            if self.lower is not None:
                bad |= (v < self.lower) if self.lower_inclusive else (v <= self.lower)
            if self.upper is not None:
                bad |= (v > self.upper) if self.upper_inclusive else (v >= self.upper)
        return bad

    def scaled(self, factor):
        """Copy of the rule with both bounds multiplied by factor (used for area_total x Max_A)."""
        return AttributeRule(
            self.text, self.field,
            None if self.lower is None else self.lower * factor, self.lower_inclusive,
            None if self.upper is None else self.upper * factor, self.upper_inclusive,
        )


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RULE_RE = re.compile(
    rf"^\s*(?:({_NUMBER})\s*(<=|<)\s*)?"          # optional "a <" / "a <="
    rf"([A-Za-z_]\w*)"                             # field name
    rf"(?:\s*(<=|<|>=|>)\s*({_NUMBER}))?\s*$"      # optional "< b" / ">= a" ...
)


def parse_attribute_rule(text):
    """Parse "field<b", "field<=b", "field>a", "field>=a", "a<field<b", or "a<=field<=b"."""
    m = _RULE_RE.match(str(text))
    if not m:
        raise ValueError(f"Cannot parse attribute rule {text!r}.")
    left_val, left_op, field, right_op, right_val = m.groups()
    lower = upper = None
    lower_inc = upper_inc = False
    if left_val is not None:
        lower, lower_inc = float(left_val), left_op == "<="
    if right_op is not None:
        if right_op in ("<", "<="):
            upper, upper_inc = float(right_val), right_op == "<="
        else:
            if left_val is not None:
                raise ValueError(f"Invalid attribute rule {text!r}: use 'a<field<b' for two-sided ranges.")
            lower, lower_inc = float(right_val), right_op == ">="
    if lower is None and upper is None:
        raise ValueError(f"Attribute rule {text!r} has no bound.")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"Attribute rule {text!r} has lower bound greater than upper bound.")
    return AttributeRule(str(text).strip(), field, lower, lower_inc, upper, upper_inc)


def parse_attribute_rules(rules):
    """Parse a list of rule strings into AttributeRule objects (validates syntax early)."""
    return [parse_attribute_rule(r) for r in rules]


def required_columns(cfg):
    """Columns the configured filter needs: base columns, quality fields, and rule fields."""
    cols = list(BASE_REQUIRED_COLUMNS) + list(cfg["quality_fields"])
    for rule in parse_attribute_rules(cfg["attribute_rules"]):
        cols.append(rule.field)
        if rule.field == "area_total":
            cols.append("p_ref_area")
    return list(dict.fromkeys(cols))


def evaluate_overlap(ov_value, rule):
    """Overlap rule: keep only when exactly one semicolon-separated value satisfies the range.

    Returns (remove, detail).
    """
    s = "" if ov_value is None else str(ov_value).strip()
    if s == "" or s.lower() == "nan":
        return True, "overlap is empty"
    nums = []
    for part in s.split(";"):
        try:
            v = float(part)
        except ValueError:
            continue
        if np.isfinite(v):
            nums.append(v)
    if not nums:
        return True, "overlap has no numeric value"
    n_in = int(np.sum(~rule.violated(nums)))
    if n_in == 0:
        return True, f"no value satisfying {rule.describe()} ({s})"
    if n_in >= 2:
        return True, f"{n_in} values satisfying {rule.describe()} ({s})"
    return False, ""


def compute_max_a(area_vals, p_ref_vals):
    """Max_A = max(median(area_total), first valid p_ref_area); NaN inputs are ignored."""
    median_area = np.nanmedian(area_vals) if np.isfinite(area_vals).any() else np.nan
    p_ref_finite = p_ref_vals[np.isfinite(p_ref_vals)]
    p_ref_area_val = p_ref_finite[0] if p_ref_finite.size else np.nan
    if not p_ref_finite.size:
        print("Warning: p_ref_area is entirely missing; Max_A uses the area_total median only.")
    finite = [v for v in (median_area, p_ref_area_val) if np.isfinite(v)]
    max_a = max(finite) if finite else np.nan
    return median_area, p_ref_area_val, max_a


def apply_attribute_rule(df_raw, row_idx, rule):
    """Evaluate one rule on the rows row_idx of df_raw.

    Returns (violation_mask, detail_list, info) where info holds reporting values
    (Max_A details for area_total; the effective range for all rules).
    """
    n = len(row_idx)
    info = {"text": rule.text, "field": rule.field, "kind": "range", "range": rule.describe()}

    if rule.field == "overlap":
        info["kind"] = "overlap"
        values = df_raw["overlap"].to_numpy()[row_idx] if "overlap" in df_raw.columns else [""] * n
        mask = np.zeros(n, dtype=bool)
        detail = [""] * n
        for k in range(n):
            mask[k], detail[k] = evaluate_overlap(values[k], rule)
        return mask, ["overlap: " + d if d else "" for d in detail], info

    values = numeric_column(df_raw, rule.field)[row_idx]
    effective = rule
    if rule.field == "area_total":
        info["kind"] = "area_total"
        median_area, p_ref_area_val, max_a = compute_max_a(values, numeric_column(df_raw, "p_ref_area")[row_idx])
        effective = rule.scaled(max_a)
        info.update({"median_area": median_area, "p_ref_area_val": p_ref_area_val, "max_a": max_a,
                     "range": effective.describe()})
        print(f"area_total median = {median_area:.4f}, p_ref_area = {p_ref_area_val:.4f}, Max_A = {max_a:.4f}")

    mask = effective.violated(values)
    reason = f"{rule.field} outside {effective.describe()}"
    return mask, [reason if m else "" for m in mask], info


# -----------------------------------------------------------------------------
# Entry point: steps 0-2 on one lake table
# -----------------------------------------------------------------------------
def apply_attribute_filters(df_raw, cfg):
    """Run steps 0-2 on one single-lake table.

    Returns a dict with raw-level arrays (suffix _raw), the index map
    pre_kept_raw_idx from the preprocessed ("pre") table to the raw table,
    pre-level masks/reasons from the attribute filters, and statistics.
    """
    n_raw = len(df_raw)
    obs_days_raw = parse_observation_time_days(df_raw)
    wse_raw = numeric_column(df_raw, "wse")

    # --- Step 0: invalid-record screening ---
    invalid_mask = ~np.isfinite(obs_days_raw) | ~np.isfinite(wse_raw)
    n_invalid = int(invalid_mask.sum())

    # --- Step 1: 72 h nearby-observation selection (invalid rows excluded) ---
    fields = cfg["quality_fields"]
    quality_raw = (np.column_stack([numeric_column(df_raw, f) for f in fields])
                   if fields else np.zeros((n_raw, 0)))
    grouping_days = obs_days_raw.copy()
    grouping_days[invalid_mask] = np.nan
    pre_keep_mask, pre_reason = select_best_within_hours(grouping_days, quality_raw, cfg["nearby_hours"])
    pre_keep_mask &= ~invalid_mask
    for i in np.flatnonzero(invalid_mask):
        pre_reason[i] = REASON_INVALID
    n_pre_removed = int(np.sum(~pre_keep_mask & ~invalid_mask))

    pre_kept_raw_idx = np.flatnonzero(pre_keep_mask)
    n_total = len(pre_kept_raw_idx)
    print(f"72 h selection: raw {n_raw} rows, invalid {n_invalid}, removed {n_pre_removed}, kept {n_total}.")

    # --- Step 2: user-defined flag (attribute) rules, evaluated on the pre table ---
    rules = parse_attribute_rules(cfg["attribute_rules"])
    rule_masks = []
    rule_details = []
    rule_infos = []
    for rule in rules:
        mask, detail, info = apply_attribute_rule(df_raw, pre_kept_raw_idx, rule)
        rule_masks.append(mask)
        rule_details.append(detail)
        rule_infos.append(info)

    hit_matrix = (np.column_stack(rule_masks) if rule_masks else np.zeros((n_total, 0), dtype=bool))
    combined_mask = hit_matrix.any(axis=1)
    n_hits_per_row = hit_matrix.sum(axis=1)
    for j, info in enumerate(rule_infos):
        info["n_hit"] = int(hit_matrix[:, j].sum())
        info["n_only"] = int(np.sum(hit_matrix[:, j] & (n_hits_per_row == 1)))
    print("Attribute rules: "
          + ", ".join(f"{info['text']} {info['n_hit']}" for info in rule_infos)
          + f", removed (union) {int(combined_mask.sum())}.")

    # Row-level reasons from the attribute rules.
    reason_pre = [""] * n_total
    for k in range(n_total):
        reason_pre[k] = " + ".join(d[k] for d in rule_details if d[k])

    stats = {
        "n_raw": n_raw,
        "n_invalid": n_invalid,
        "n_pre_removed": n_pre_removed,
        "n_after_preprocess": n_total,
        "rules": rule_infos,
    }

    return {
        "n_raw": n_raw,
        "wse_raw": wse_raw,
        "area_raw": numeric_column(df_raw, "area_total"),
        "wse_std_raw": numeric_column(df_raw, "wse_std"),
        "pre_reason_raw": pre_reason,
        "pre_kept_raw_idx": pre_kept_raw_idx,
        "obs_days_pre": obs_days_raw[pre_kept_raw_idx],
        "wse_pre": wse_raw[pre_kept_raw_idx],
        "combined_mask": combined_mask,
        "reason_pre": reason_pre,
        "stats": stats,
    }
