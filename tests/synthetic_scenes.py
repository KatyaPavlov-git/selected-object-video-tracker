"""Synthetic exit-absence-return scenes (Stage 2A benchmark builder).

Builds a controlled scene — textured background, a distinct textured target
patch, and a visually similar decoy with a parameterized correlation — plus an
absolute transform schedule that translates the scene rigidly so the selected
target point leaves the frame, stays absent, and returns. Ground truth is
analytic via evaluation.synthetic_gt_sequence; nothing here reads tracker
state. Used by tests/test_benchmark_exit_return.py (Stage 2A form) and, from
Stage 2D on, by the reacquisition benchmark; write_mp4 feeds the same scene
through the real main.py pipeline.
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")  # repo root: make the package importable under discovery

from ground_target_tracking import utils


def _textured(h, w, seed, blur=5):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (blur, blur), 0)


def build_scene(w=640, h=480, seed=7, target_xy=(560, 240), decoy_xy=(160, 240),
                patch_px=41, decoy_blend=0.85):
    """Base frame with a distinct target patch and a correlated decoy.

    decoy_blend is the fraction of TARGET texture in the decoy (1.0 = an
    identical twin), so the decoy's visual similarity is a controlled
    parameter. Positions are chosen so the decoy stays fully in-frame across
    the whole exit_return_schedule while the target leaves through the right
    border.
    """
    base = _textured(h, w, seed)
    target = _textured(patch_px, patch_px, seed + 1, blur=3)
    noise = _textured(patch_px, patch_px, seed + 2, blur=3)
    decoy = (decoy_blend * target.astype(np.float32)
             + (1.0 - decoy_blend) * noise.astype(np.float32)).astype(np.uint8)
    half = patch_px // 2
    tx, ty = target_xy
    dx, dy = decoy_xy
    base[ty - half:ty + half + 1, tx - half:tx + half + 1] = target
    base[dy - half:dy + half + 1, dx - half:dx + half + 1] = decoy
    return (base, utils.Point2D(float(tx), float(ty)),
            utils.Point2D(float(dx), float(dy)))


def exit_return_schedule(n_visible=60, n_exit=30, n_absent=80, n_return=30,
                         n_tail=50, drift_dx=-0.5, exit_dx=6.0):
    """Absolute 2x3 translations: gentle drift -> exit right -> hold absent ->
    symmetric return -> hold. Phase boundaries are NOT returned — derive them
    from the analytic ground-truth points (the honest source), not from the
    schedule's intent.
    """
    txs = []
    tx = 0.0
    for _ in range(n_visible):
        tx += drift_dx
        txs.append(tx)
    for _ in range(n_exit):
        tx += exit_dx
        txs.append(tx)
    for _ in range(n_absent):
        txs.append(tx)
    for _ in range(n_return):
        tx -= exit_dx
        txs.append(tx)
    for _ in range(n_tail):
        txs.append(tx)
    return [np.array([[1, 0, t], [0, 1, 0]], np.float32) for t in txs]


def fully_visible(gt_point, w, h, margin=25.0):
    """True when the ground-truth point is at least `margin` px inside the
    frame — the definition used for 'last fully visible' / 'first fully
    visible return' in the benchmark contract."""
    return (margin <= gt_point.x < w - margin
            and margin <= gt_point.y < h - margin)


def write_mp4(path, base, transforms, fps=30.0):
    """Encode the scene as a real video (frame 0 = base) so the identical
    benchmark can be driven through the actual main.py pipeline (2D/2E)."""
    h, w = base.shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    vw.write(base)
    for M in transforms:
        vw.write(cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT))
    vw.release()


# --------------------------------------------------------------------------- #
# In-frame identity-loss scene (Amendment A1 §15.8) — built in Stage 2B,
# consumed by the Stage 2D integration benchmark. The true target stays visible
# the whole time; a higher-contrast LOOK-ALIKE distractor sweeps THROUGH the
# target's neighbourhood to pull a real LK tracker off the target (emergent
# drift). Nothing here forces a reacquisition location or reads tracker state —
# frames + analytic ground truth only.
# --------------------------------------------------------------------------- #
def _contrast(patch, gain):
    """Scale contrast about the patch mean (a distractor LK prefers to lock)."""
    m = float(patch.mean())
    return np.clip((patch.astype(np.float32) - m) * gain + m, 0, 255).astype(np.uint8)


def build_distractor_scene(w=640, h=480, seed=11, target_xy=(320, 240),
                           patch_px=41, decoy_blend=0.85, decoy_contrast=1.5,
                           n_frames=140, sweep_x=(60, 580)):
    """A stationary, always-visible target and a look-alike distractor that
    sweeps horizontally through the target's row (overlapping it mid-run).

    Returns (frames, gt_points): the target ground-truth point is constant (it
    never moves and is always visible), so any committed point that ends up on
    the moving decoy is unambiguous drift. `decoy_blend` is the fraction of
    TARGET texture in the decoy (1.0 = identical twin); `decoy_contrast` makes
    the decoy the higher-contrast lock the LK corners are drawn to.
    """
    base = _textured(h, w, seed)
    target = _textured(patch_px, patch_px, seed + 1, blur=3)
    noise = _textured(patch_px, patch_px, seed + 2, blur=3)
    decoy = (decoy_blend * target.astype(np.float32)
             + (1.0 - decoy_blend) * noise.astype(np.float32)).astype(np.uint8)
    decoy = _contrast(decoy, decoy_contrast)
    half = patch_px // 2
    tx, ty = target_xy

    def paste(img, patch, cx, cy):
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + patch_px, y0 + patch_px
        ix0, iy0 = max(0, x0), max(0, y0)
        ix1, iy1 = min(w, x1), min(h, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        img[iy0:iy1, ix0:ix1] = patch[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]

    x0, x1 = sweep_x
    frames, gt = [], []
    for i in range(n_frames):
        f = base.copy()
        paste(f, target, tx, ty)                       # true target (always visible)
        dx = int(round(x0 + (x1 - x0) * i / max(1, n_frames - 1)))
        paste(f, decoy, dx, ty)                        # distractor sweeps its row
        frames.append(f)
        gt.append(utils.Point2D(float(tx), float(ty)))
    return frames, gt
