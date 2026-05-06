from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _make_vehicle_track(
    rng: np.random.Generator,
    track_id: int,
    start_frame: int,
    n_frames: int,
    lane_id: int,
    lc_type: str | None,
    x0: float,
    speed: float,
) -> pd.DataFrame:
    frames = np.arange(start_frame, start_frame + n_frames)
    t = np.arange(n_frames)
    x = x0 + speed * t / 25.0 + rng.normal(0, 0.05, n_frames)
    y_base = lane_id * 3.5
    y = np.full(n_frames, y_base, dtype=float)
    lane = np.full(n_frames, lane_id, dtype=int)

    if lc_type in {"left", "right"} and n_frames > 90:
        change_center = rng.integers(45, n_frames - 35)
        direction = -1 if lc_type == "left" else 1
        target_lane = lane_id + direction
        transition = np.clip((t - (change_center - 12)) / 24.0, 0, 1)
        smooth = transition * transition * (3 - 2 * transition)
        y = y_base + direction * 3.5 * smooth + rng.normal(0, 0.04, n_frames)
        lane[t >= change_center] = target_lane
    else:
        y = y + rng.normal(0, 0.03, n_frames)

    x_velocity = np.gradient(x) * 25.0
    y_velocity = np.gradient(y) * 25.0
    x_acc = np.gradient(x_velocity) * 25.0
    y_acc = np.gradient(y_velocity) * 25.0

    return pd.DataFrame({
        "frame": frames,
        "id": track_id,
        "x": x,
        "y": y,
        "width": 4.6 + rng.normal(0, 0.05, n_frames),
        "height": 1.8 + rng.normal(0, 0.03, n_frames),
        "xVelocity": x_velocity,
        "yVelocity": y_velocity,
        "xAcceleration": x_acc,
        "yAcceleration": y_acc,
        "laneId": lane,
    })


def create_demo_data(raw_dir: str | Path, n_recordings: int = 2, seed: int = 7) -> None:
    """Create a small highD-like demo dataset.

    The demo is intentionally small. It is only used to prove that the Agent
    pipeline can run end to end without requiring the original highD dataset.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for rec in range(1, n_recordings + 1):
        all_tracks = []
        meta_rows = []
        for i in range(36):
            track_id = i + 1
            lane_id = int(rng.choice([2, 3, 4]))
            lc_type = None
            if i % 6 == 0 and lane_id > 1:
                lc_type = "left"
            elif i % 6 == 1 and lane_id < 5:
                lc_type = "right"
            n_frames = int(rng.integers(105, 170))
            start_frame = int(rng.integers(0, 30))
            speed = float(rng.uniform(18, 33))
            x0 = float(rng.uniform(-40, 40) + i * 12)
            df = _make_vehicle_track(rng, track_id, start_frame, n_frames, lane_id, lc_type, x0, speed)
            all_tracks.append(df)
            meta_rows.append({
                "id": track_id,
                "width": float(df["width"].mean()),
                "height": float(df["height"].mean()),
                "initialFrame": int(df["frame"].min()),
                "finalFrame": int(df["frame"].max()),
                "numFrames": int(len(df)),
                "drivingDirection": 1,
            })

        tracks = pd.concat(all_tracks, ignore_index=True)
        tracks.to_csv(raw_dir / f"{rec:02d}_tracks.csv", index=False)
        pd.DataFrame(meta_rows).to_csv(raw_dir / f"{rec:02d}_tracksMeta.csv", index=False)
        pd.DataFrame([{
            "id": rec,
            "frameRate": 25,
            "locationId": 1,
            "speedLimit": 120,
            "numVehicles": len(meta_rows),
        }]).to_csv(raw_dir / f"{rec:02d}_recordingMeta.csv", index=False)

    print(f"[DEMO] 已生成演示数据: {raw_dir.resolve()}")


if __name__ == "__main__":
    create_demo_data(Path("demo_data/raw_data"))
