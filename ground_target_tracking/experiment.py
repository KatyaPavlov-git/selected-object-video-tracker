"""experiment.py — per-run experiment logging for reproducible experiments.

A `--save` run creates `LOGS_DIR/<RUN_DIR_PREFIX>NNN/` (auto-incrementing) and writes:
    config.json    snapshot of all config params + the CLI args used
    frame_log.csv  per-frame rows: frame_idx, x, y, n_points, mean_error, source, step_ms
    output.mp4     the annotated frames
    stats.json     final performance summary (from evaluation.PerfStats)

`logs/` is git-ignored, so runs stay local. The tracking columns (n_points,
mean_error, source) are placeholders in Batch 1 (no tracker yet) and become
meaningful from M6.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Optional

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config, utils
except ImportError:  # fallback: run from inside the package folder
    import config
    import utils

_FRAME_FIELDS = ["frame_idx", "x", "y", "n_points", "mean_error", "source",
                 "confidence", "state", "step_ms"]


class RunLogger:
    """Owns one experiment run directory and its artifact files."""

    def __init__(self, base_dir: Optional[str] = None, prefix: Optional[str] = None) -> None:
        base_dir = base_dir or config.LOGS_DIR
        prefix = prefix or config.RUN_DIR_PREFIX
        self.run_dir = self._next_run_dir(base_dir, prefix)
        os.makedirs(self.run_dir, exist_ok=True)
        self._csv_file = open(os.path.join(self.run_dir, "frame_log.csv"), "w", newline="")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=_FRAME_FIELDS)
        self._csv.writeheader()
        self._writer = None  # cv2.VideoWriter, opened lazily by open_video()
        self.n_rows = 0

    @staticmethod
    def _next_run_dir(base_dir: str, prefix: str) -> str:
        used = []
        if os.path.isdir(base_dir):
            for name in os.listdir(base_dir):
                tail = name[len(prefix):]
                if name.startswith(prefix) and tail.isdigit():
                    used.append(int(tail))
        n = (max(used) + 1) if used else 1
        return os.path.join(base_dir, f"{prefix}{n:03d}")

    def save_config(self, cli_args: Optional[dict] = None) -> None:
        """Snapshot all UPPERCASE config params (+ CLI args) to config.json."""
        snap = {}
        for name in dir(config):
            if not name.isupper():
                continue
            val = getattr(config, name)
            if isinstance(val, (bool, int, float, str)):
                snap[name] = val
            elif isinstance(val, (tuple, list)):
                snap[name] = list(val)
        payload = {"config": snap, "cli_args": cli_args or {}}
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def open_video(self, fps: float, frame_size) -> None:
        self._writer = utils.make_writer(
            os.path.join(self.run_dir, "output.mp4"), fps, frame_size
        )

    def write_frame(self, frame_bgr) -> None:
        if self._writer is not None and frame_bgr is not None:
            self._writer.write(frame_bgr)

    def log_frame(self, frame_idx: int, point=None, n_points=None, mean_error=None,
                  source: str = "fixed", step_ms=None, confidence=None,
                  state: str = "") -> None:
        x, y = ("", "")
        if point is not None:
            x, y = round(point.x, 2), round(point.y, 2)
        self._csv.writerow({
            "frame_idx": frame_idx,
            "x": x,
            "y": y,
            "n_points": "" if n_points is None else n_points,
            "mean_error": "" if mean_error is None else round(mean_error, 4),
            "source": source,
            "confidence": "" if confidence is None else round(confidence, 4),
            "state": state,
            "step_ms": "" if step_ms is None else round(step_ms, 3),
        })
        self.n_rows += 1

    def finish(self, stats: dict) -> None:
        with open(os.path.join(self.run_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        self.close()

    def close(self) -> None:
        try:
            if self._csv_file and not self._csv_file.closed:
                self._csv_file.flush()
                self._csv_file.close()
        finally:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
