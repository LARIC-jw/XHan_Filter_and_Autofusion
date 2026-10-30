"""
AutoFusion v1 - fusion module.

Selects, segment by segment, whether the HALF or the XHanFilter keep/remove
decision is used for a SWOT LakeSP WSE series. Only SWOT time and WSE are
used; WSE values are never modified, only the keep/remove mask is fused.

1. Robust local judge: a leave-neighbourhood-out robust local polynomial
   (tricube time weights x Tukey bisquare) predicts every observation from its
   surroundings and yields an outlier z-score and a confidence.
2. Method cost: for each method, a smoothed local cost combines
   - keeping a likely outlier / removing a likely valid point,
   - roughness of the method's own retained trajectory,
   - coverage loss (low local retention, long artificial gaps).
3. Segment selection: a Viterbi path with a switch penalty picks the method
   per observation and restarts at long natural gaps. Consecutive picks of the
   same method form segments.

Entry point: fuse_lake(df, cfg) where df comes from
AutoFusion_v1_io.align_filter_results().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NS_PER_DAY = 86_400_000_000_000.0


@dataclass
class LocalPrediction:
    """Per-observation outputs of the robust local judge."""
    prediction: np.ndarray
    scale: np.ndarray
    zscore: np.ndarray
    confidence: np.ndarray
    neighbor_count: np.ndarray
    short_roughness_zscore: np.ndarray | None = None


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def time_to_days(time):
    """Datetime series -> float days (NaN for NaT)."""
    t = pd.to_datetime(time, errors="coerce")
    days = t.astype("int64").to_numpy(dtype=float) / NS_PER_DAY
    days[t.isna().to_numpy()] = np.nan
    return days


def robust_mad(values):
    """MAD scaled to a Gaussian standard deviation; NaN when empty."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=float), -40.0, 40.0)))


def lake_scale_floor(wse, absolute_floor):
    """Scale floor adapted to the lake's WSE range (1% of the 5-95% span, capped at 0.25 m)."""
    v = np.asarray(wse, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return absolute_floor
    span = float(np.nanpercentile(v, 95) - np.nanpercentile(v, 5))
    return max(absolute_floor, min(0.25, 0.01 * max(span, 0.0)))


def robust_polynomial_fit(x, y, base_weight, degree, scale_floor):
    """Weighted robust polynomial fit (base weights x Tukey bisquare, up to 8 iterations).

    Returns (beta, robust_weight, scale) or (None, None, nan) when the fit fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base_weight = np.asarray(base_weight, dtype=float)
    design = np.column_stack([x ** power for power in range(degree + 1)])
    robust_weight = np.ones(y.size, dtype=float)
    beta = None

    for _ in range(8):
        weight = base_weight * robust_weight
        if np.sum(weight > 1e-8) < degree + 2:
            return None, None, np.nan
        sqrt_w = np.sqrt(weight)
        try:
            beta, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
        except np.linalg.LinAlgError:
            return None, None, np.nan
        residual = y - design @ beta
        scale = max(robust_mad(residual), scale_floor)
        u = residual / (4.685 * scale)
        new_weight = np.zeros_like(u)
        inside = np.abs(u) < 1.0
        new_weight[inside] = (1.0 - u[inside] ** 2) ** 2
        converged = np.max(np.abs(new_weight - robust_weight)) < 1e-3
        robust_weight = new_weight
        if converged:
            break

    if beta is None:
        return None, None, np.nan
    residual = y - design @ beta
    final_scale = max(robust_mad(residual[robust_weight > 0.05]), scale_floor)
    return beta, robust_weight, final_scale


# -----------------------------------------------------------------------------
# Step 1: robust local judge
# -----------------------------------------------------------------------------
def local_leave_neighborhood_out(time, wse, cfg, reference_mask=None):
    """Predict each observation from reference observations outside its own visit.

    Observations closer than cfg["same_visit_exclude_days"] are excluded to
    avoid leakage from near-duplicate passes. The window grows to the nearest
    cfg["judge_min_neighbors"] points when sparse and is capped at
    cfg["judge_max_neighbors"].
    """
    t_days = time_to_days(time)
    y = np.asarray(wse, dtype=float)
    n = y.size
    reference_mask = np.ones(n, dtype=bool) if reference_mask is None else np.asarray(reference_mask, dtype=bool)

    out = LocalPrediction(
        prediction=np.full(n, np.nan), scale=np.full(n, np.nan), zscore=np.full(n, np.nan),
        confidence=np.zeros(n), neighbor_count=np.zeros(n, dtype=int),
    )
    scale_floor = lake_scale_floor(y, cfg["absolute_scale_floor_m"])
    min_nb, max_nb = cfg["judge_min_neighbors"], cfg["judge_max_neighbors"]

    for i in range(n):
        if not np.isfinite(t_days[i]) or not np.isfinite(y[i]):
            continue
        distance = np.abs(t_days - t_days[i])
        eligible = (reference_mask & np.isfinite(t_days) & np.isfinite(y)
                    & (distance > cfg["same_visit_exclude_days"])
                    & (distance <= cfg["judge_max_window_days"]))
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size < min_nb:
            continue

        inside = eligible_idx[distance[eligible_idx] <= cfg["judge_window_days"]]
        if inside.size >= min_nb:
            local_idx = inside
        else:
            local_idx = eligible_idx[np.argsort(distance[eligible_idx])[:min_nb]]
        if local_idx.size > max_nb:
            local_idx = local_idx[np.argsort(distance[local_idx])[:max_nb]]

        dt = t_days[local_idx] - t_days[i]
        x = dt / max(float(np.max(np.abs(dt))), 1.0)
        bandwidth = max(float(np.max(np.abs(x))) * 1.001, 1e-6)
        time_weight = (1.0 - np.minimum(np.abs(x) / bandwidth, 1.0) ** 3) ** 3

        left_count, right_count = int(np.sum(dt < 0)), int(np.sum(dt > 0))
        two_sided = min(left_count, right_count) >= 2
        degree = 2 if two_sided and local_idx.size >= 9 else 1
        beta, robust_weight, local_scale = robust_polynomial_fit(x, y[local_idx], time_weight, degree, scale_floor)
        if beta is None:
            continue

        # Inflate the scale for one-sided support and for distant neighbours.
        nearest_days = float(np.min(np.abs(dt)))
        balanced_factor = 1.0 if two_sided else 1.45
        gap_factor = 1.0 + min(nearest_days / 120.0, 1.0)
        effective_scale = max(local_scale * balanced_factor * gap_factor, scale_floor)

        confidence = min(float(np.sum(robust_weight > 0.05)) / 12.0, 1.0)
        confidence *= 1.0 if two_sided else 0.70
        confidence *= float(np.exp(-nearest_days / 180.0))

        out.prediction[i] = beta[0]
        out.scale[i] = effective_scale
        out.zscore[i] = abs(y[i] - beta[0]) / effective_scale
        out.confidence[i] = float(np.clip(confidence, 0.0, 1.0))
        out.neighbor_count[i] = local_idx.size
    return out


# -----------------------------------------------------------------------------
# Step 2: per-method local cost
# -----------------------------------------------------------------------------
def local_coverage_penalty(time, keep, cfg):
    """Penalise low local retention and long gaps created where observations exist."""
    keep = np.asarray(keep, dtype=bool)
    n = keep.size
    window = cfg["score_window_points"]
    min_ret = cfg["min_local_retention"]
    retention = (pd.Series(keep.astype(float))
                 .rolling(window, center=True, min_periods=min(5, max(1, n))).mean().to_numpy())
    low_retention = np.maximum(0.0, (min_ret - retention) / min_ret) ** 2

    gap_penalty = np.zeros(n, dtype=float)
    t = time_to_days(time)
    kept_idx = np.flatnonzero(keep)
    max_gap = cfg["max_keep_gap_days"]
    for left, right in zip(kept_idx[:-1], kept_idx[1:]):
        gap_days = t[right] - t[left]
        if gap_days > max_gap and right > left + 1:
            excess = min((gap_days - max_gap) / max_gap, 1.0)
            gap_penalty[left + 1:right] = np.maximum(gap_penalty[left + 1:right], excess)
    return 0.70 * low_retention + 0.30 * gap_penalty


def short_scale_roughness(time, wse, keep, judge_scale, cfg):
    """Saw-tooth detection: residual of each kept point from the line between its kept neighbours.

    Scored only when both neighbours are within 90 days, and normalised by the
    judge scale (not the method's own scale). Returns (cost, zscore).
    """
    keep = np.asarray(keep, dtype=bool)
    y = np.asarray(wse, dtype=float)
    scale = np.asarray(judge_scale, dtype=float)
    n = keep.size
    zscore = np.full(n, np.nan)
    cost = np.zeros(n, dtype=float)
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size < 3:
        return cost, zscore

    t = time_to_days(time)
    fallback_scale = lake_scale_floor(y, cfg["absolute_scale_floor_m"])
    for left, center, right in zip(kept_idx[:-2], kept_idx[1:-1], kept_idx[2:]):
        dt_left, dt_right = t[center] - t[left], t[right] - t[center]
        if dt_left <= 0 or dt_right <= 0 or dt_left > 90.0 or dt_right > 90.0:
            continue
        interpolated = y[left] + dt_left / (dt_left + dt_right) * (y[right] - y[left])
        local_scale = scale[center] if np.isfinite(scale[center]) else fallback_scale
        z = abs(y[center] - interpolated) / max(local_scale, fallback_scale)
        zscore[center] = z
        cost[center] = float(sigmoid((z - 1.20) * 1.8))  # ~0 below 0.5 scales, ~1 above 2 scales
    return cost, zscore


def candidate_point_cost(time, wse, keep, available, judge, cfg):
    """Smoothed local cost of one method (lower is better).

    Returns (smoothed_cost, trajectory_prediction, coverage_cost).
    """
    keep = np.asarray(keep, dtype=bool)
    available = np.asarray(available, dtype=bool)

    # Outlier probability from the judge; z ~ 3 is the transition.
    outlier_probability = sigmoid((judge.zscore - 3.0) * 1.35)
    outlier_probability = np.where(np.isfinite(outlier_probability), outlier_probability, 0.5)
    confidence = np.asarray(judge.confidence, dtype=float)
    keep_cost = confidence * outlier_probability + (1.0 - confidence) * 0.10
    remove_cost = cfg["false_remove_weight"] * (confidence * (1.0 - outlier_probability) + (1.0 - confidence) * 0.75)
    label_cost = np.where(keep, keep_cost, remove_cost)

    # Roughness of the method's own retained trajectory.
    trajectory = local_leave_neighborhood_out(time, wse, cfg, reference_mask=keep)
    trajectory_probability = sigmoid((trajectory.zscore - 2.6) * 1.25)
    trajectory_probability = np.where(np.isfinite(trajectory_probability), trajectory_probability, 0.30)
    trajectory_cost = np.where(keep, trajectory_probability, 0.0)

    roughness_cost, roughness_zscore = short_scale_roughness(time, wse, keep, judge.scale, cfg)
    coverage_cost = local_coverage_penalty(time, keep, cfg)

    raw_cost = (label_cost
                + cfg["trajectory_weight"] * trajectory_cost
                + cfg["short_roughness_weight"] * roughness_cost
                + cfg["coverage_weight"] * coverage_cost)
    raw_cost = np.where(available, raw_cost, raw_cost + 1.25)  # method has no decision here

    smooth_cost = (pd.Series(raw_cost)
                   .rolling(cfg["score_window_points"], center=True, min_periods=min(5, max(1, raw_cost.size)))
                   .mean().to_numpy())
    trajectory.short_roughness_zscore = roughness_zscore
    return smooth_cost, trajectory, coverage_cost


# -----------------------------------------------------------------------------
# Step 3: Viterbi segment selection
# -----------------------------------------------------------------------------
def viterbi_select(time, half_cost, xhan_cost, cfg):
    """Choose state 0 (HALF) or 1 (XHan) per observation with a switch penalty.

    The path restarts after gaps longer than cfg["natural_break_days"].
    """
    costs = np.column_stack([half_cost, xhan_cost])
    n = costs.shape[0]
    selected = np.zeros(n, dtype=int)
    if n == 0:
        return selected

    t = time_to_days(time)
    break_before = np.zeros(n, dtype=bool)
    break_before[0] = True
    if n > 1:
        break_before[1:] = np.diff(t) > cfg["natural_break_days"]
    starts = np.flatnonzero(break_before)
    ends = np.r_[starts[1:], n]
    penalty = cfg["switch_penalty"]

    for start, end in zip(starts, ends):
        length = end - start
        dp = np.full((length, 2), np.inf)
        back = np.zeros((length, 2), dtype=int)
        dp[0] = costs[start]
        for pos in range(1, length):
            for state in (0, 1):
                stay = dp[pos - 1, state]
                switch = dp[pos - 1, 1 - state] + penalty
                if stay <= switch:
                    dp[pos, state] = costs[start + pos, state] + stay
                    back[pos, state] = state
                else:
                    dp[pos, state] = costs[start + pos, state] + switch
                    back[pos, state] = 1 - state
        state = int(np.argmin(dp[-1]))
        selected[end - 1] = state
        for pos in range(length - 1, 0, -1):
            state = back[pos, state]
            selected[start + pos - 1] = state
    return selected


def assign_segments(df, cfg):
    """Add segment_id (new segment at a method switch or a long gap) and build the segment table."""
    out = df.copy()
    gap_days = out["Time"].diff().dt.total_seconds().div(86400.0)
    changed = out["selected_method"].ne(out["selected_method"].shift()) | gap_days.gt(cfg["natural_break_days"]).fillna(False)
    out["segment_id"] = changed.cumsum().astype(int)

    rows = []
    for segment_id, group in out.groupby("segment_id", sort=True):
        rows.append({
            "segment_id": int(segment_id),
            "start_time": group["Time"].min(),
            "end_time": group["Time"].max(),
            "selected_method": str(group["selected_method"].iloc[0]),
            "n_observations": int(len(group)),
            "half_kept": int(group["half_keep"].sum()),
            "xhan_kept": int(group["xhan_keep"].sum()),
            "fused_kept": int(group["fused_keep"].sum()),
            "half_mean_score": round(float(group["half_score"].mean()), 4),
            "xhan_mean_score": round(float(group["xhan_score"].mean()), 4),
        })
    return out, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def fuse_lake(df, cfg):
    """Fuse HALF and XHan masks for one lake.

    df needs: Time, wse, half_keep, xhan_keep, half_available, xhan_available.
    Returns (points_df with judge/score/selection columns, segments_df).
    """
    df = df.copy()
    wse = df["wse"].to_numpy(dtype=float)

    judge = local_leave_neighborhood_out(df["Time"], wse, cfg)
    df["judge_zscore"] = judge.zscore
    df["judge_confidence"] = judge.confidence

    half_score, _, _ = candidate_point_cost(
        df["Time"], wse, df["half_keep"].to_numpy(), df["half_available"].to_numpy(), judge, cfg)
    xhan_score, _, _ = candidate_point_cost(
        df["Time"], wse, df["xhan_keep"].to_numpy(), df["xhan_available"].to_numpy(), judge, cfg)
    df["half_score"] = half_score
    df["xhan_score"] = xhan_score

    selected = viterbi_select(df["Time"], half_score, xhan_score, cfg)
    df["selected_method"] = np.where(selected == 0, "HALF", "XHan")
    df["fused_keep"] = np.where(selected == 0, df["half_keep"].to_numpy(), df["xhan_keep"].to_numpy())
    return assign_segments(df, cfg)
