from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import Config
from .highd_labeling import read_tracks


F1_COLUMNS = [
    "ego_xVelocity", "ego_yVelocity", "ego_xAcceleration", "ego_yAcceleration",
    "ego_width", "ego_height", "ego_laneId", "ego_speed",
    "d_front_same_x", "dv_front_same_x", "d_rear_same_x", "dv_rear_same_x",
    "d_front_left_x", "dv_front_left_x", "d_rear_left_x", "dv_rear_left_x",
    "d_front_right_x", "dv_front_right_x", "d_rear_right_x", "dv_rear_right_x",
    "neighbor_count_30m", "neighbor_count_60m", "min_front_gap", "min_rear_gap",
    "abs_yVelocity", "time_headway_front",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _nearest_in_lane(frame_df: pd.DataFrame, ego: pd.Series, lane: int, direction: str) -> tuple[float, float]:
    """Return longitudinal gap and relative speed for front or rear vehicle in given lane."""
    cand = frame_df[(frame_df["laneId"] == lane) & (frame_df["id"] != ego["id"])].copy()
    if cand.empty:
        return 999.0, 0.0
    cand["dx"] = cand["x"] - ego["x"]
    if direction == "front":
        cand = cand[cand["dx"] > 0]
        if cand.empty:
            return 999.0, 0.0
        row = cand.loc[cand["dx"].idxmin()]
    else:
        cand = cand[cand["dx"] < 0]
        if cand.empty:
            return -999.0, 0.0
        row = cand.loc[cand["dx"].idxmax()]
    gap = _safe_float(row["dx"])
    rel_speed = _safe_float(row["xVelocity"] - ego["xVelocity"])
    return gap, rel_speed


def _features_for_sample(all_tracks: pd.DataFrame, sample: pd.Series) -> dict:
    frame = int(sample["frame"])
    track_id = int(sample["track_id"])
    frame_df = all_tracks[all_tracks["frame"] == frame]
    ego_rows = frame_df[frame_df["id"] == track_id]
    if ego_rows.empty:
        return {col: np.nan for col in F1_COLUMNS}
    ego = ego_rows.iloc[0]
    lane = int(ego["laneId"])

    fs_gap, fs_dv = _nearest_in_lane(frame_df, ego, lane, "front")
    rs_gap, rs_dv = _nearest_in_lane(frame_df, ego, lane, "rear")
    fl_gap, fl_dv = _nearest_in_lane(frame_df, ego, lane - 1, "front")
    rl_gap, rl_dv = _nearest_in_lane(frame_df, ego, lane - 1, "rear")
    fr_gap, fr_dv = _nearest_in_lane(frame_df, ego, lane + 1, "front")
    rr_gap, rr_dv = _nearest_in_lane(frame_df, ego, lane + 1, "rear")

    others = frame_df[frame_df["id"] != track_id].copy()
    if others.empty:
        count_30 = count_60 = 0
    else:
        dx = (others["x"] - ego["x"]).abs()
        dy = (others["y"] - ego["y"]).abs()
        count_30 = int(((dx <= 30) & (dy <= 10)).sum())
        count_60 = int(((dx <= 60) & (dy <= 10)).sum())

    front_gaps = [v for v in [fs_gap, fl_gap, fr_gap] if v < 900]
    rear_gaps = [abs(v) for v in [rs_gap, rl_gap, rr_gap] if v > -900]
    min_front = min(front_gaps) if front_gaps else 999.0
    min_rear = min(rear_gaps) if rear_gaps else 999.0
    ego_speed = float(np.hypot(_safe_float(ego["xVelocity"]), _safe_float(ego["yVelocity"])))
    thw = float(min_front / max(ego_speed, 0.1)) if min_front < 900 else 99.0

    return {
        "ego_xVelocity": _safe_float(ego["xVelocity"]),
        "ego_yVelocity": _safe_float(ego["yVelocity"]),
        "ego_xAcceleration": _safe_float(ego["xAcceleration"]),
        "ego_yAcceleration": _safe_float(ego["yAcceleration"]),
        "ego_width": _safe_float(ego["width"]),
        "ego_height": _safe_float(ego["height"]),
        "ego_laneId": float(lane),
        "ego_speed": ego_speed,
        "d_front_same_x": fs_gap,
        "dv_front_same_x": fs_dv,
        "d_rear_same_x": rs_gap,
        "dv_rear_same_x": rs_dv,
        "d_front_left_x": fl_gap,
        "dv_front_left_x": fl_dv,
        "d_rear_left_x": rl_gap,
        "dv_rear_left_x": rl_dv,
        "d_front_right_x": fr_gap,
        "dv_front_right_x": fr_dv,
        "d_rear_right_x": rr_gap,
        "dv_rear_right_x": rr_dv,
        "neighbor_count_30m": float(count_30),
        "neighbor_count_60m": float(count_60),
        "min_front_gap": float(min_front),
        "min_rear_gap": float(min_rear),
        "abs_yVelocity": abs(_safe_float(ego["yVelocity"])),
        "time_headway_front": thw,
    }


def build_f1_features(cfg: Config, index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    track_cache: dict[str, pd.DataFrame] = {}
    for _, sample in index.iterrows():
        rec = str(sample["recording_id"]).zfill(2)
        if rec not in track_cache:
            track_cache[rec] = read_tracks(cfg, rec)
        feats = _features_for_sample(track_cache[rec], sample)
        row = {
            "sample_id": sample["sample_id"],
            "recording_id": sample["recording_id"],
            "track_id": int(sample["track_id"]),
            "frame": int(sample["frame"]),
            "label": int(sample["label"]),
            "label_name": sample["label_name"],
        }
        row.update(feats)
        rows.append(row)
    f1 = pd.DataFrame(rows)
    for col in F1_COLUMNS:
        if col not in f1.columns:
            f1[col] = np.nan
    f1[F1_COLUMNS] = f1[F1_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f1


def check_feature_alignment(index: pd.DataFrame, f1: pd.DataFrame) -> pd.DataFrame:
    idx_keys = set(index["sample_id"].astype(str))
    f1_keys = set(f1["sample_id"].astype(str))
    rows = [
        {"check_item": "index_sample_count", "value": len(index), "status": "ok"},
        {"check_item": "f1_sample_count", "value": len(f1), "status": "ok"},
        {"check_item": "missing_in_f1", "value": len(idx_keys - f1_keys), "status": "ok" if not (idx_keys - f1_keys) else "error"},
        {"check_item": "extra_in_f1", "value": len(f1_keys - idx_keys), "status": "ok" if not (f1_keys - idx_keys) else "warning"},
        {"check_item": "f1_feature_dim", "value": len(F1_COLUMNS), "status": "ok" if len(F1_COLUMNS) == 26 else "warning"},
    ]
    return pd.DataFrame(rows)
