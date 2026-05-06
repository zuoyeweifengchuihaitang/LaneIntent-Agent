from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from .config import Config
from .utils import ensure_dir, label_name


TRACK_PATTERN = re.compile(r"(\d+)_tracks\.csv$")

REQUIRED_TRACK_COLUMNS = {
    "frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity",
    "xAcceleration", "yAcceleration", "laneId"
}


def discover_recordings(raw_dir: str | Path) -> list[str]:
    raw_dir = Path(raw_dir)
    recs: list[str] = []
    for p in sorted(raw_dir.glob("*_tracks.csv")):
        m = TRACK_PATTERN.search(p.name)
        if m:
            recs.append(m.group(1))
    return recs


def check_recording_files(cfg: Config) -> pd.DataFrame:
    rows = []
    for rec in discover_recordings(cfg.raw_path):
        tracks_path = cfg.raw_path / f"{rec}_tracks.csv"
        tracks_meta_path = cfg.raw_path / f"{rec}_tracksMeta.csv"
        rec_meta_path = cfg.raw_path / f"{rec}_recordingMeta.csv"
        row = {
            "recording_id": rec,
            "tracks_exists": tracks_path.exists(),
            "tracksMeta_exists": tracks_meta_path.exists(),
            "recordingMeta_exists": rec_meta_path.exists(),
            "status": "ok",
            "message": "",
        }
        if not tracks_meta_path.exists() or not rec_meta_path.exists():
            row["status"] = "warning"
            row["message"] = "缺少 tracksMeta 或 recordingMeta，主流程仍会尽量使用 tracks.csv"
        try:
            sample = pd.read_csv(tracks_path, nrows=5)
            missing = sorted(REQUIRED_TRACK_COLUMNS - set(sample.columns))
            if missing:
                row["status"] = "error"
                row["message"] = f"tracks.csv 缺少字段: {missing}"
        except Exception as exc:
            row["status"] = "error"
            row["message"] = repr(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def read_tracks(cfg: Config, recording_id: str) -> pd.DataFrame:
    path = cfg.raw_path / f"{recording_id}_tracks.csv"
    df = pd.read_csv(path)
    missing = REQUIRED_TRACK_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要字段: {sorted(missing)}")
    df = df.copy()
    df["recording_id"] = recording_id
    return df


def _lane_change_label(old_lane: int, new_lane: int, left_when_lane_decreases: bool) -> int:
    delta = int(new_lane) - int(old_lane)
    if delta == 0:
        return 2
    if left_when_lane_decreases:
        return 1 if delta < 0 else 3
    return 1 if delta > 0 else 3


def _select_anchor_frame(track: pd.DataFrame, change_pos: int, cfg: Config) -> int:
    """Select an intention-like anchor before the actual lane-id change.

    Rule:
    - Prefer the earliest frame t before the lane change where lateral velocity is
      high and 1-second future lateral displacement is large.
    - If no strict anchor exists, fallback to 1 second before the lane change.
    """
    future = max(1, int(round(cfg.future_seconds_for_anchor * cfg.fps)))
    frames = track["frame"].to_numpy()
    y = track["y"].to_numpy()
    vy = track["yVelocity"].to_numpy()
    start = max(0, change_pos - future * 2)
    end = max(0, change_pos)
    candidates = []
    for i in range(start, end):
        j = min(i + future, len(track) - 1)
        if abs(vy[i]) >= cfg.vy_threshold and abs(y[j] - y[i]) >= cfg.future_lateral_disp_threshold:
            candidates.append(i)
    if candidates:
        return int(frames[candidates[0]])
    fallback_pos = max(0, change_pos - future)
    return int(frames[fallback_pos])


def _build_lane_change_samples_for_track(track: pd.DataFrame, cfg: Config) -> list[dict]:
    samples: list[dict] = []
    track = track.sort_values("frame").reset_index(drop=True)
    lanes = track["laneId"].to_numpy()
    frames = track["frame"].to_numpy()
    change_positions = np.where(lanes[1:] != lanes[:-1])[0] + 1
    if len(change_positions) == 0:
        return samples

    for pos in change_positions:
        old_lane = int(lanes[pos - 1])
        new_lane = int(lanes[pos])
        label = _lane_change_label(old_lane, new_lane, cfg.left_when_lane_decreases)
        anchor_frame = _select_anchor_frame(track, int(pos), cfg)
        samples.append({
            "recording_id": str(track["recording_id"].iloc[0]),
            "track_id": int(track["id"].iloc[0]),
            "frame": int(anchor_frame),
            "decision_frame": int(frames[pos]),
            "label": int(label),
            "label_name": label_name(label),
            "sample_type": "lane_change",
            "old_lane": old_lane,
            "new_lane": new_lane,
        })
    return samples


def _build_lane_keep_samples_for_track(track: pd.DataFrame, cfg: Config) -> list[dict]:
    samples: list[dict] = []
    track = track.sort_values("frame").reset_index(drop=True)
    lanes = track["laneId"].to_numpy()
    frames = track["frame"].to_numpy()
    future = max(1, int(round(cfg.lane_keep_future_seconds * cfg.fps)))
    stride = max(1, int(cfg.lane_keep_stride_frames))

    for i in range(0, max(1, len(track) - future), stride):
        j = min(i + future, len(track) - 1)
        if np.all(lanes[i:j + 1] == lanes[i]):
            samples.append({
                "recording_id": str(track["recording_id"].iloc[0]),
                "track_id": int(track["id"].iloc[0]),
                "frame": int(frames[i]),
                "decision_frame": int(frames[i]),
                "label": 2,
                "label_name": "直行",
                "sample_type": "lane_keep",
                "old_lane": int(lanes[i]),
                "new_lane": int(lanes[i]),
            })
    return samples


def build_sample_index(cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in discover_recordings(cfg.raw_path):
        tracks = read_tracks(cfg, rec)
        for _, track in tracks.groupby("id"):
            rows.extend(_build_lane_change_samples_for_track(track, cfg))
            rows.extend(_build_lane_keep_samples_for_track(track, cfg))
    if not rows:
        raise RuntimeError("没有构建出任何样本，请检查数据格式与筛选阈值。")
    index = pd.DataFrame(rows)
    index = index.drop_duplicates(subset=["recording_id", "track_id", "frame", "label"]).reset_index(drop=True)
    index.insert(0, "sample_id", [f"S{i:06d}" for i in range(len(index))])
    return index


def balance_index(index: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    counts = index["label"].value_counts()
    available = counts[counts >= cfg.min_samples_per_class]
    if len(available) < 2:
        raise RuntimeError(f"有效类别不足，当前类别计数: {counts.to_dict()}")
    target = int(available.min())
    parts = []
    for label, group in index.groupby("label"):
        if len(group) < cfg.min_samples_per_class:
            continue
        parts.append(group.sample(n=target, random_state=cfg.balance_random_seed))
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=cfg.balance_random_seed).reset_index(drop=True)
    out["sample_id"] = [f"B{i:06d}" for i in range(len(out))]
    return out


def save_label_distribution(index: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    dist = (
        index.groupby(["label", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("label")
    )
    dist["ratio"] = dist["count"] / dist["count"].sum()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(path, index=False, encoding="utf-8-sig")
    return dist


def plot_label_distribution(dist: pd.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    xlabels = dist["label"].map({1: "Left LC", 2: "Keep", 3: "Right LC"}).fillna(dist["label_name"])
    ax.bar(xlabels, dist["count"])
    ax.set_title(title)
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    for i, v in enumerate(dist["count"]):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
