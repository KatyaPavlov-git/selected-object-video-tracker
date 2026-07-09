"""utils.py — shared helpers for the ground-target-tracking pipeline.

This module imports only `config` (the deepest dependency sink), so every other
module can import from it without circular-import risk. All tunable values come
from config rather than being hardcoded here.

Milestone 1 provides the small data types and the video I/O used by the loader.
ROI extraction and richer drawing helpers are added in Milestones 2–3.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # works as `python3 -m ground_target_tracking.main`
    from . import config
except ImportError:  # fallback: run from inside the package folder
    import config


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Point2D:
    """A 2-D point in full-resolution frame coordinates."""

    x: float
    y: float

    def as_int(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def dist(self, other: "Point2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class VideoMeta:
    """Basic properties of an opened video."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    def __str__(self) -> str:
        return (
            f"{os.path.basename(self.path)}  {self.width}x{self.height}  "
            f"fps={self.fps:.2f}  frames={self.frame_count}"
        )


@dataclass
class Patch:
    """A local ROI / patch of visual context around the target point (M3).

    Semantics: `center` is the user-selected ground target (the predicted impact
    point); the patch is the local context around it. `size` is the requested
    (configured) side length; the cropped `image` may be smaller than `size x size`
    when the target sits near a frame border (border-safe clipping).

    NOTE: in M2/M3 `center` is a fixed pixel location only because tracking is not
    implemented yet. It denotes the same real-world ground point, so once a tracker
    exists (M6+) its pixel coordinates must be updated per frame as the camera/scene
    moves — the fixed-pixel value is a placeholder, not the final behavior.
    """

    image: np.ndarray       # the cropped patch pixels (a copy)
    top_left: Point2D       # clamped ROI top-left in full-resolution coords
    size: int               # requested square side length (config.PATCH_SIZE)
    center: Point2D         # the selected target point (ROI center)

    def bbox(self) -> Tuple[int, int, int, int]:
        """Actual clamped ROI as (x0, y0, x1, y1) in full-resolution coords."""
        x0, y0 = self.top_left.as_int()
        h, w = self.image.shape[:2]
        return (x0, y0, x0 + w, y0 + h)

    @property
    def actual_size(self) -> Tuple[int, int]:
        """Actual (width, height) of the cropped patch (<= size near borders)."""
        h, w = self.image.shape[:2]
        return (w, h)


@dataclass
class TrackResult:
    """One frame's tracking outcome (produced by trackers, consumed by main).

    `point` is the current best estimate of the ground target (full-res px).
    `ok` is False when the measurement is unreliable (too few live points). The
    later Kalman wrapper (M7) will set `source="predict"` while coasting.
    `confidence` is the tracker's own [0..1] estimate of measurement quality,
    computed from its internal signals (M8); session-level signals (patch
    similarity, edge proximity) are combined on top of it by TrackingSession.
    """

    point: "Point2D"
    ok: bool
    n_points: int = 0
    mean_error: float = 0.0
    redetected: bool = False
    source: str = "measure"
    raw_point: Optional["Point2D"] = None  # pre-filter measurement (M7 --show-measurement)
    confidence: float = 1.0                # tracker-computed measurement quality (M8)


# --------------------------------------------------------------------------- #
# Video I/O (Milestone 1)
# --------------------------------------------------------------------------- #
def is_stream_url(path: str) -> bool:
    """True if `path` is a network/stream URL OpenCV can open directly
    (http/https/rtsp/rtmp), rather than a local filesystem path."""
    return str(path).lower().startswith(
        ("http://", "https://", "rtsp://", "rtmp://")
    )


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoMeta]:
    """Open and validate a video source: a local file path OR a network stream
    URL (http/https/rtsp/rtmp) handed straight to OpenCV.

    Raises ValueError if a local file is missing or OpenCV cannot open the
    source. Returns the live capture handle and a VideoMeta snapshot.
    Note: URL/stream playback depends on OpenCV's ffmpeg backend, the network,
    and the codec; local files are the primary tested path.
    """
    if not is_stream_url(path) and not os.path.isfile(path):
        raise ValueError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    meta = VideoMeta(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, meta


def read_frames(cap: cv2.VideoCapture) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr_frame) until the stream ends or a read fails."""
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield idx, frame
        idx += 1


def get_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    """Return a specific frame by index (used to pick an initialization frame)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {index}")
    return frame


def make_writer(path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an mp4 writer (mp4v) at `fps` and full-res `frame_size` (w, h).

    Creates the parent directory if needed. Raises ValueError if the writer cannot
    be opened. Used by the experiment logger to save annotated output.mp4.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        path, fourcc, float(fps) if fps and fps > 0 else config.DEFAULT_FPS,
        (int(frame_size[0]), int(frame_size[1])),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open VideoWriter for: {path}")
    return writer


# --------------------------------------------------------------------------- #
# ROI / Patch (Milestone 3)
# --------------------------------------------------------------------------- #
def clamp_roi(cx: float, cy: float, size: int,
              frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Return a border-safe square ROI (x0, y0, x1, y1) centered on (cx, cy).

    The box is clipped to the frame, so near an edge or corner it becomes smaller
    rather than wrapping or producing negative/out-of-range indices. Guarantees
    0 <= x0 < x1 <= frame_w and 0 <= y0 < y1 <= frame_h for an in-frame center.
    """
    half = size // 2
    icx, icy = int(round(cx)), int(round(cy))
    x0 = max(0, icx - half)
    y0 = max(0, icy - half)
    x1 = min(frame_w, x0 + size)
    y1 = min(frame_h, y0 + size)
    # If clipping on the right/bottom shrank the box, pull the top-left back in so
    # we keep as much of the requested size as the frame allows.
    x0 = max(0, min(x0, x1 - 1))
    y0 = max(0, min(y0, y1 - 1))
    return x0, y0, x1, y1


def extract_patch(frame: np.ndarray, center: "Point2D",
                  size: Optional[int] = None) -> Patch:
    """Crop a border-safe square patch around `center`. `size` defaults to config.PATCH_SIZE.

    `center` is the fixed ground target; the returned Patch holds the local
    context, its clamped top-left, the requested size, and the center point.
    """
    if size is None:
        size = config.PATCH_SIZE
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = clamp_roi(center.x, center.y, size, w, h)
    crop = frame[y0:y1, x0:x1].copy()
    return Patch(image=crop, top_left=Point2D(x0, y0), size=size, center=center)


def build_overlay_mask(frame_bgr: np.ndarray, cfg=config,
                       kind: str = "seed") -> Optional[np.ndarray]:
    """Binary ignore mask (255 = fixed HUD overlay) for feature seeding/scoring.

    `kind` selects the crosshair-region treatment (the pillarbox part is shared):
      * "seed" (default, UNCHANGED): conservative filled disc
        (OVERLAY_CROSSHAIR_RADIUS_PX) — used for LK seeding, ignore masks,
        dilation, known-occlusion. Over-masking is desirable here.
      * "identity": geometry-faithful STROKES only — thin diagonals + "+" ticks +
        centre dot — so the clean scene BETWEEN the strokes stays valid. Used for
        M8 local appearance similarity, where discarding the between-stroke scene
        (the 45px filled disc did) destroys the identity signal (RC1).

    Two INDEPENDENT parts (their coupling was the geometry defect found on the
    official sample video):

    1) Pillarbox stripes — genuinely black side columns (every pixel of the
       column <= OVERLAY_LETTERBOX_MAX_INTENSITY) plus a margin band at the
       content edge. Detection only decides where black bars are; it does NOT
       anchor the HUD.
    2) HUD X + crosshair — the drone OSD draws these anchored to the FULL
       FRAME centre, not to the content box: diagonals follow
       x(y) = cx +/- OVERLAY_X_SLOPE * (y - cy) with (cx, cy) = (w/2, h/2),
       endpoints derived by extending each line to the top and bottom frame
       rows (cv2.line clips as needed); the crosshair disc sits at (cx, cy).
       OVERLAY_X_SLOPE is dimensionless (dx/dy), so the model scales with any
       resolution of the same OSD. Measured on the sample video across frames
       0/100/400/800: slope 1.3321..1.3337 (~4/3), crossing (960.0, 542.4+-0.3)
       on 1920x1080 — the <=3px residual centre offset is absorbed by the
       band half-width (OVERLAY_DIAG_THICKNESS_PX / 2).

    Coordinates are zero-based (x, y) in full-resolution source pixels.
    Purely advisory: display code never uses it. Returns None when
    OVERLAY_MASK_ENABLED is off.
    """
    if not getattr(cfg, "OVERLAY_MASK_ENABLED", False):
        return None
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), np.uint8)
    # --- 1) pillarbox stripes (independent of the HUD geometry) ----------- #
    col_max = gray.max(axis=0)
    lit = np.nonzero(col_max > cfg.OVERLAY_LETTERBOX_MAX_INTENSITY)[0]
    cx0, cx1 = (int(lit[0]), int(lit[-1]) + 1) if len(lit) else (0, w)
    if cx0 > 0 or cx1 < w:  # bars + their boundary edges
        m = int(cfg.OVERLAY_LETTERBOX_MARGIN_PX)
        mask[:, :min(w, cx0 + m)] = 255
        mask[:, max(0, cx1 - m):] = 255
    # --- 2) HUD X + crosshair, anchored to the frame centre --------------- #
    cx, cy = w / 2.0, h / 2.0
    s = float(cfg.OVERLAY_X_SLOPE)
    if kind == "identity":
        # Geometry-faithful STROKES only (no filled disc): thin diagonals + "+"
        # ticks + centre dot, on a sub-mask that is dilated (anti-alias margin)
        # WITHOUT dilating the pillarbox, then OR-ed in.
        strokes = np.zeros((h, w), np.uint8)
        td = int(getattr(cfg, "OVERLAY_IDENTITY_DIAG_THICKNESS_PX", 8))
        cv2.line(strokes, (int(round(cx - s * cy)), 0),
                 (int(round(cx + s * (h - 1 - cy))), h - 1), 255, td)
        cv2.line(strokes, (int(round(cx + s * cy)), 0),
                 (int(round(cx - s * (h - 1 - cy))), h - 1), 255, td)
        icx, icy = int(round(cx)), int(round(cy))
        g = int(getattr(cfg, "OVERLAY_TICK_GAP_PX", 5))
        L = int(getattr(cfg, "OVERLAY_TICK_LEN_PX", 17))
        tt = int(getattr(cfg, "OVERLAY_TICK_THICKNESS_PX", 3))
        for sg in (-1, 1):
            cv2.line(strokes, (icx + sg * g, icy), (icx + sg * (g + L), icy), 255, tt)
            cv2.line(strokes, (icx, icy + sg * g), (icx, icy + sg * (g + L)), 255, tt)
        cv2.circle(strokes, (icx, icy),
                   int(getattr(cfg, "OVERLAY_CENTER_DOT_RADIUS_PX", 3)), 255, -1)
        d = int(getattr(cfg, "OVERLAY_IDENTITY_DILATE_PX", 3))
        if d > 0:
            strokes = cv2.dilate(strokes, np.ones((d, d), np.uint8))
        mask[strokes > 0] = 255
        return mask
    # kind == "seed" (default): conservative diagonals + filled disc (UNCHANGED).
    t = int(cfg.OVERLAY_DIAG_THICKNESS_PX)
    cv2.line(mask, (int(round(cx - s * cy)), 0),
             (int(round(cx + s * (h - 1 - cy))), h - 1), 255, t)
    cv2.line(mask, (int(round(cx + s * cy)), 0),
             (int(round(cx - s * (h - 1 - cy))), h - 1), 255, t)
    cv2.circle(mask, (int(round(cx)), int(round(cy))),
               int(cfg.OVERLAY_CROSSHAIR_RADIUS_PX), 255, -1)
    return mask


# --------------------------------------------------------------------------- #
# Video discovery / selection helpers
# --------------------------------------------------------------------------- #
def list_videos(directory: Optional[str] = None,
                exts: Optional[Sequence[str]] = None) -> List[str]:
    """Return a sorted list of supported video files directly inside `directory`.

    Non-recursive. Returns an empty list if the directory does not exist or holds
    no supported videos. `directory` defaults to config.VIDEOS_DIR and `exts` to
    config.SUPPORTED_VIDEO_EXTENSIONS.
    """
    if directory is None:
        directory = config.VIDEOS_DIR
    if exts is None:
        exts = config.SUPPORTED_VIDEO_EXTENSIONS
    exts = tuple(e.lower() for e in exts)
    if not os.path.isdir(directory):
        return []
    found = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(exts) and os.path.isfile(os.path.join(directory, name))
    ]
    return sorted(found)


def pick_video_via_dialog(exts: Optional[Sequence[str]] = None) -> Optional[str]:
    """Open a native file-picker dialog; return the chosen path or None.

    On macOS this uses a native AppleScript (osascript) dialog; otherwise (or if
    that fails) it falls back to Tk's filedialog. Returns None if the user cancels
    or no GUI backend is available.
    """
    if exts is None:
        exts = config.SUPPORTED_VIDEO_EXTENSIONS
    if sys.platform == "darwin":
        path = _pick_via_osascript(exts)
        if path is not None:
            return path
    return _pick_via_tk(exts)


def _pick_via_osascript(exts: Sequence[str]) -> Optional[str]:
    """macOS native 'choose file' dialog via osascript. None on cancel/failure."""
    type_list = ", ".join('"%s"' % e.lstrip(".") for e in exts)
    script = (
        "try\n"
        '  set theFile to choose file with prompt "Select a video to track" '
        "of type {%s}\n"
        "  POSIX path of theFile\n"
        "on error number -128\n"  # -128 = user pressed Cancel
        '  return ""\n'
        "end try" % type_list
    )
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    path = proc.stdout.strip()
    return path or None


def _pick_via_tk(exts: Sequence[str]) -> Optional[str]:
    """Cross-platform Tk file dialog fallback. None on cancel/failure."""
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:
        return None
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.update()
        patterns = " ".join("*%s" % e for e in exts)
        path = filedialog.askopenfilename(
            title="Select a video to track",
            filetypes=[("Videos", patterns), ("All files", "*.*")],
        )
        root.destroy()
    except Exception:
        return None
    return path or None


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def fit_display(frame: np.ndarray, max_h: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Downscale a tall frame so it fits on screen.

    `max_h` defaults to config.MAX_DISPLAY_HEIGHT. Returns (display_frame, scale).
    To map a click in display coordinates back to the original frame:
    original = display_coord / scale. The portrait 576x1024 clips are taller than
    most laptop screens, so interactive windows show a scaled copy while all
    tracking math stays in full-resolution coordinates.
    """
    if max_h is None:
        max_h = config.MAX_DISPLAY_HEIGHT
    h = frame.shape[0]
    if h <= max_h:
        return frame, 1.0
    scale = max_h / float(h)
    disp = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return disp, scale


def video_summary(meta: "VideoMeta", title: Optional[str] = None) -> str:
    """Build the console summary banner printed once a video is selected.

    `title` defaults to config.WINDOW_TITLE; the rule width is config.BANNER_WIDTH.
    """
    if title is None:
        title = config.WINDOW_TITLE
    bar = "-" * config.BANNER_WIDTH
    return "\n".join([
        bar,
        title,
        bar,
        f"Video: {os.path.basename(meta.path)}",
        f"Resolution: {meta.width}x{meta.height}",
        f"FPS: {meta.fps:.2f}",
        f"Frames: {meta.frame_count}",
        bar,
    ])


def draw_point(img: np.ndarray, point: "Point2D", color=None,
               radius: Optional[int] = None, label: Optional[str] = None) -> np.ndarray:
    """Draw a filled marker with a crosshair at `point`. Colors/radius from config. In place."""
    if color is None:
        color = config.TARGET_POINT_COLOR
    if radius is None:
        radius = config.TARGET_POINT_RADIUS
    cx, cy = point.as_int()
    reach = radius + 6
    cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.line(img, (cx - reach, cy), (cx + reach, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - reach), (cx, cy + reach), color, 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (cx + reach, cy - reach), config.FONT_FACE,
                    config.FONT_SCALE, color, config.HUD_TEXT_THICKNESS, cv2.LINE_AA)
    return img


def draw_keypoints(img: np.ndarray, keypoints, color=None) -> np.ndarray:
    """Draw detected keypoints on a copy of `img` (M5.5 feature visualization).

    Accepts gray or BGR input; returns a BGR image with rich keypoints drawn.
    """
    if color is None:
        color = config.HUD_TEXT_COLOR
    base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.drawKeypoints(
        base, list(keypoints), None, color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_roi(img: np.ndarray, bbox: Tuple[int, int, int, int],
             color=None, thickness: Optional[int] = None) -> np.ndarray:
    """Draw the ROI rectangle (x0, y0, x1, y1). Color/thickness from config. In place."""
    if color is None:
        color = config.ROI_BOX_COLOR
    if thickness is None:
        thickness = config.ROI_BOX_THICKNESS
    x0, y0, x1, y1 = bbox
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    return img


def scale_bbox(bbox: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    """Scale a full-resolution bbox into display coordinates."""
    x0, y0, x1, y1 = bbox
    return (int(round(x0 * scale)), int(round(y0 * scale)),
            int(round(x1 * scale)), int(round(y1 * scale)))


def draw_trajectory(img: np.ndarray, points: Sequence["Point2D"], scale: float = 1.0,
                    color=None, thickness: Optional[int] = None) -> np.ndarray:
    """Draw the tracked-point trajectory as a polyline (M6). No-op for < 2 points."""
    if color is None:
        color = config.TRAJECTORY_COLOR
    if thickness is None:
        thickness = config.TRAJECTORY_THICKNESS
    if points is None or len(points) < 2:
        return img
    pts = np.array([[int(round(p.x * scale)), int(round(p.y * scale))] for p in points],
                   dtype=np.int32)
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)
    return img


def draw_hud(img: np.ndarray, lines: Sequence[str], color=None) -> np.ndarray:
    """Draw HUD text lines (with a dark outline for legibility) top-left. In place.

    Colors, font and thicknesses are sourced from config.VISUALIZATION settings.
    """
    if color is None:
        color = config.HUD_TEXT_COLOR
    y = 24
    for line in lines:
        cv2.putText(img, line, (10, y), config.FONT_FACE, config.FONT_SCALE,
                    (0, 0, 0), config.HUD_OUTLINE_THICKNESS, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), config.FONT_FACE, config.FONT_SCALE,
                    color, config.HUD_TEXT_THICKNESS, cv2.LINE_AA)
        y += 26
    return img
