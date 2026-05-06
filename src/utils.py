from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_banner(text: str) -> None:
    line = "=" * max(30, len(text) + 8)
    print(f"\n{line}\n{text}\n{line}")


def write_json(obj: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def label_name(label: int) -> str:
    mapping = {1: "左换道", 2: "直行", 3: "右换道"}
    return mapping.get(int(label), f"未知{label}")
