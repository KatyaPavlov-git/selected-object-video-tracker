"""preprocessing.py — frame/patch preprocessing pipelines (Milestone 4).

DESIGN PRINCIPLE:
    The preprocessing pipeline is part of the tracking method. Different tracking
    methods may require different preprocessing pipelines, so each method selects
    the pipeline that suits it. There is no single global pipeline, and NO HSV /
    color segmentation — the target is a user-selected ground point, not a color.

Each step is `np.ndarray -> np.ndarray`; pipelines are built by `compose(*steps)`.
Step parameters default to the values in `config` (single source of truth), and
the optional steps are gated by `config.USE_GAUSSIAN_BLUR` / `config.USE_CLAHE` /
`config.GAMMA`. `PREPROC_RESIZE_SCALE` is intentionally NOT wired into the method
pipelines (resizing changes the coordinate space the tracker works in); `resize`
is provided as a standalone helper only.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config
except ImportError:  # fallback: run from inside the package folder
    import config

Pipeline = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- #
# Individual steps (ndarray -> ndarray)
# --------------------------------------------------------------------------- #
def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR -> single-channel gray; pass through if already 2-D."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian(img: np.ndarray, ksize=None, sigma=None) -> np.ndarray:
    """Gaussian blur. `ksize` is forced odd. Defaults from config."""
    ksize = config.GAUSSIAN_KSIZE if ksize is None else ksize
    sigma = config.GAUSSIAN_SIGMA if sigma is None else sigma
    k = int(ksize) | 1  # force odd; cv2 requires odd kernel sizes
    return cv2.GaussianBlur(img, (k, k), float(sigma))


def clahe(img: np.ndarray, clip=None, grid=None) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization (expects/produces gray)."""
    clip = config.CLAHE_CLIP_LIMIT if clip is None else clip
    grid = config.CLAHE_TILE_GRID if grid is None else grid
    gray = to_gray(img)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(grid))
    return op.apply(gray)


def gamma(img: np.ndarray, g=None) -> np.ndarray:
    """Gamma correction via LUT. Identity when g == 1.0."""
    g = config.GAMMA if g is None else g
    if abs(float(g) - 1.0) < 1e-6:
        return img
    inv = 1.0 / float(g)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def resize(img: np.ndarray, scale=None) -> np.ndarray:
    """Resize by a scale factor. Identity when scale == 1.0 (standalone helper)."""
    scale = config.PREPROC_RESIZE_SCALE if scale is None else scale
    if abs(float(scale) - 1.0) < 1e-6:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def compose(*steps: Pipeline) -> Pipeline:
    """Return a pipeline f(img) = stepN(...step2(step1(img)))."""
    def pipeline(img: np.ndarray) -> np.ndarray:
        out = img
        for step in steps:
            out = step(out)
        return out
    return pipeline


# --------------------------------------------------------------------------- #
# Per-method pipeline factories
# --------------------------------------------------------------------------- #
def optical_flow_pipeline(cfg=config) -> Pipeline:
    """Lucas-Kanade prep: gray (+ optional gamma / blur / CLAHE).

    LK relies on intensity gradients/corners, so this stays minimal by default
    (gray only); blur/CLAHE are opt-in via config.
    """
    steps = [to_gray]
    if abs(float(cfg.GAMMA) - 1.0) > 1e-6:
        steps.append(gamma)
    if cfg.USE_GAUSSIAN_BLUR:
        steps.append(gaussian)
    if cfg.USE_CLAHE:
        steps.append(clahe)
    return compose(*steps)


def orb_pipeline(cfg=config) -> Pipeline:
    """ORB/AKAZE feature prep: gray -> CLAHE (always) -> optional gamma / blur.

    CLAHE is integral here because it is the single best lever for coaxing
    descriptors out of low-texture ground patches.
    """
    steps = [to_gray]
    if abs(float(cfg.GAMMA) - 1.0) > 1e-6:
        steps.append(gamma)
    steps.append(clahe)
    if cfg.USE_GAUSSIAN_BLUR:
        steps.append(gaussian)
    return compose(*steps)


def template_pipeline(cfg=config) -> Pipeline:
    """Template-matching prep: gray (+ optional gamma / CLAHE). Reserved for M9."""
    steps = [to_gray]
    if abs(float(cfg.GAMMA) - 1.0) > 1e-6:
        steps.append(gamma)
    if cfg.USE_CLAHE:
        steps.append(clahe)
    return compose(*steps)


def get_pipeline(method: str, cfg=config) -> Pipeline:
    """Dispatch a pipeline by method name: 'of' | 'orb' | 'template'."""
    key = (method or "").lower()
    if key == "of":
        return optical_flow_pipeline(cfg)
    if key == "orb":
        return orb_pipeline(cfg)
    if key == "template":
        return template_pipeline(cfg)
    raise ValueError(f"Unknown preprocessing method '{method}' (expected of|orb|template).")
