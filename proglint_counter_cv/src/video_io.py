"""Video reading/writing, frame timestamps, camera-cut detection and view fingerprints."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoError(RuntimeError):
    pass


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def to_dict(self) -> dict:
        return asdict(self)


def probe_video(path: str | Path) -> VideoMeta:
    path = Path(path)
    if not path.exists():
        raise VideoError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"OpenCV cannot open {path.name} (unsupported codec/container or corrupt file)")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError(f"{path.name} opened but no frame could be decoded")
        h, w = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0 or fps > 1000:
            fps = 30.0  # unknown FPS; timestamps fall back to this nominal rate and the report says so
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoMeta(str(path), w, h, fps, n, n / fps if n else 0.0)
    finally:
        cap.release()


def iter_frames(meta: VideoMeta, stride: int = 1, max_frames: int | None = None
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, frame).

    Timestamps come from the container (CAP_PROP_POS_MSEC) so variable-frame-rate files keep their
    real timeline. If the container timestamps are missing or go backwards, index / fps is used.
    """
    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        raise VideoError(f"Cannot open {meta.path}")
    idx, last_t, emitted = -1, -1.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000.0 if ms and np.isfinite(ms) else idx / meta.fps
            if idx > 0 and (t <= last_t or abs(t - idx / meta.fps) > 2.0):
                t = idx / meta.fps  # container timestamp unusable for this frame
            last_t = t
            if idx % stride:
                continue
            yield idx, t, frame
            emitted += 1
            if max_frames and emitted >= max_frames:
                break
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise VideoError(f"Cannot read frame {index} of {path}")
        return frame
    finally:
        cap.release()


# ----------------------------------------------------------------------------- scene / view checks

def _hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    cv2.normalize(h, h, 1.0, 0.0, cv2.NORM_L1)
    return h


class SceneCutDetector:
    """Flags hard cuts in edited footage (fixed CCTV cameras normally have none).

    A cut is declared when the HSV histogram distance between consecutive analysed frames exceeds
    the threshold. People moving in a fixed view change the histogram only slightly.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._prev = None
        self.last_distance = 0.0

    def update(self, frame: np.ndarray) -> bool:
        h = _hist(frame)
        cut = False
        if self._prev is not None:
            self.last_distance = float(cv2.compareHist(self._prev, h, cv2.HISTCMP_BHATTACHARYYA))
            cut = self.last_distance > self.threshold
        self._prev = h
        return cut


FINGERPRINT_SIZE = (32, 18)


def view_fingerprint(frame: np.ndarray) -> list[int]:
    """Tiny grayscale thumbnail used to check that a calibration belongs to this camera view."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 3), FINGERPRINT_SIZE, interpolation=cv2.INTER_AREA)
    return [int(v) for v in g.flatten()]


def fingerprint_similarity(a: list[int], b: list[int]) -> float:
    """Normalised cross-correlation of two fingerprints in [-1, 1]."""
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.shape != y.shape or x.size == 0:
        return -1.0
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------- writing

def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class AnnotatedVideoWriter:
    """Writes the annotated video as browser-playable H.264.

    Frames are piped straight into the bundled ffmpeg (libx264), so there is no intermediate file and no
    second encode. Without ffmpeg (or if the pipe fails to start) OpenCV's mp4v writer is used and a
    warning says the file may not play in a browser.
    ``close()`` verifies the result can be decoded and has the expected frame count, and raises
    VideoError otherwise - a failed write is never reported as success.
    """

    def __init__(self, out_path: Path, fps: float, size: tuple[int, int], h264: bool = True):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps, self.size, self.h264 = fps, size, h264
        self.frames = 0
        self.warning: str | None = None
        self.proc: subprocess.Popen | None = None
        self.writer = None
        exe = ffmpeg_exe() if h264 else None
        if h264 and exe is None:
            self.warning = "ffmpeg not found: video kept as MPEG-4 Part 2 (may not play in a browser)"
        if exe:
            w, h = size
            # libx264 + yuv420p needs even dimensions: pad by one pixel when the source is odd-sized
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.6f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(self.out_path)]
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.codec = "h264"
            except OSError as exc:
                self.warning = f"could not start ffmpeg ({exc}); video kept as MPEG-4 Part 2"
        if self.proc is None:
            self.writer = cv2.VideoWriter(str(self.out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.writer.isOpened():
                raise VideoError(f"cv2.VideoWriter could not be opened for {self.out_path}")
            self.codec = "mp4v"

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise VideoError(f"frame size {frame.shape[1]}x{frame.shape[0]} != writer size {self.size}")
        if self.proc is not None:
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except (BrokenPipeError, OSError) as exc:
                err = self.proc.stderr.read().decode(errors="replace").strip()[:200] if self.proc.stderr else ""
                raise VideoError(f"ffmpeg stopped while writing the annotated video: {err or exc}") from exc
        else:
            self.writer.write(frame)
        self.frames += 1

    def close(self) -> Path:
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="replace").strip()
            if self.proc.wait() != 0:
                raise VideoError(f"H.264 encoding failed: {err[:300]}")
        else:
            self.writer.release()
        if self.frames == 0:
            self.out_path.unlink(missing_ok=True)
            raise VideoError("no frames were written")
        n = _count_frames(self.out_path)
        if n == 0:
            raise VideoError(f"annotated video {self.out_path} cannot be decoded after writing")
        if abs(n - self.frames) > 2:
            raise VideoError(f"annotated video has {n} frames, expected {self.frames}")
        return self.out_path


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n
