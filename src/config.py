from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Config:
    raw_data_dir: str
    output_dir: str
    fps: int = 25
    left_when_lane_decreases: bool = True
    future_seconds_for_anchor: float = 1.0
    vy_threshold: float = 0.2
    future_lateral_disp_threshold: float = 0.4
    lane_keep_stride_frames: int = 25
    lane_keep_future_seconds: float = 1.0
    balance_random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.0
    model_random_seed: int = 42
    min_samples_per_class: int = 5
    save_plots: bool = True
    enable_xgboost: bool = False

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_data_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Config(**data)
