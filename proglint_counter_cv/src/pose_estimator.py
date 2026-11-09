"""Person detection + pose keypoints + multi-object tracking (Ultralytics YOLO pose + ByteTrack).

The pretrained model only provides boxes, track IDs and 17 COCO keypoints with confidences.
It does NOT classify hand contact; all interaction logic lives in the other modules.

Keypoint order is the COCO-17 order documented by Ultralytics for ``*-pose.pt`` checkpoints.
Ultralytics reports undetected keypoints as (0, 0) with low confidence: such keypoints are never
treated as real positions here (see ``keypoint_valid``).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# COCO-17 keypoint indices (Ultralytics pose models)
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
ARM = {  # side -> (shoulder, elbow, wrist)
    "L": (KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"]),
    "R": (KP["right_shoulder"], KP["right_elbow"], KP["right_wrist"]),
}


class ModelLoadError(RuntimeError):
    pass


@dataclass
class PersonDetection:
    """One detected person in one frame.

    ``track_id`` is None when the tracker has not (yet) confirmed the detection.
    ``keypoints`` is a (17, 3) array of x, y, confidence in original-frame pixels.
    """
    track_id: int | None
    box: np.ndarray        # x1, y1, x2, y2
    score: float
    keypoints: np.ndarray  # (17, 3)

    def to_json(self) -> dict:
        return {
            "id": self.track_id,
            "box": [round(float(v), 1) for v in self.box],
            "score": round(float(self.score), 4),
            "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 4)] for x, y, c in self.keypoints],
        }

    @staticmethod
    def from_json(d: dict) -> "PersonDetection":
        return PersonDetection(
            track_id=d["id"], box=np.asarray(d["box"], float), score=float(d["score"]),
            keypoints=np.asarray(d["kp"], float).reshape(17, 3),
        )


def keypoint_valid(kp: np.ndarray, idx: int, min_conf: float, frame_size: tuple[int, int],
                   edge_margin: float = 2.0) -> bool:
    """A keypoint is usable only if confident, finite, not the (0,0) sentinel and inside the frame."""
    x, y, c = kp[idx]
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)) or c < min_conf:
        return False
    if x == 0.0 and y == 0.0:
        return False
    w, h = frame_size
    return edge_margin <= x <= w - 1 - edge_margin and edge_margin <= y <= h - 1 - edge_margin


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch import problems are reported by ultralytics later
        return "cpu"


def write_tracker_yaml(tracking_cfg, fps: float, out_dir: Path) -> Path:
    """Write an Ultralytics tracker YAML from our config (track buffer converted from seconds to frames)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer_frames = max(1, int(round(tracking_cfg.track_buffer_s * max(fps, 1.0))))
    data = {
        "tracker_type": tracking_cfg.tracker_type,
        "track_high_thresh": tracking_cfg.track_high_thresh,
        "track_low_thresh": tracking_cfg.track_low_thresh,
        "new_track_thresh": tracking_cfg.new_track_thresh,
        "track_buffer": buffer_frames,
        "match_thresh": tracking_cfg.match_thresh,
        "fuse_score": tracking_cfg.fuse_score,
    }
    if tracking_cfg.tracker_type == "botsort":  # BoT-SORT needs these keys; ReID disabled (fixed camera baseline)
        data.update({"gmc_method": "sparseOptFlow", "proximity_thresh": 0.5,
                     "appearance_thresh": 0.25, "with_reid": False, "model": "auto"})
    text = "\n".join(f"{k}: {json.dumps(v)}" for k, v in data.items()) + "\n"
    path = out_dir / f"tracker_{hashlib.md5(text.encode()).hexdigest()[:10]}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def letterbox_shape(width: int, height: int, imgsz: int, stride: int = 32) -> tuple[int, int]:
    """(h, w) network input that Ultralytics' rectangular letterbox gives a ``width`` x ``height`` frame:
    long side scaled to ``imgsz``, short side padded up to a multiple of ``stride``."""
    r = imgsz / max(width, height)
    nw, nh = round(width * r), round(height * r)
    return math.ceil(nh / stride) * stride, math.ceil(nw / stride) * stride


_EXPORT_LOCK = threading.Lock()


def openvino_model_for(model_path: Path, shape: tuple[int, int]) -> Path:
    """OpenVINO export of ``model_path`` for one static input shape, created once and cached under
    models/openvino/. The weights are the same checkpoint; only the runtime changes."""
    h, w = shape
    target = model_path.parent / "openvino" / f"{model_path.stem}_{h}x{w}_openvino_model"  # suffix required by Ultralytics
    if (target / f"{model_path.stem}.xml").exists():
        return target
    with _EXPORT_LOCK:
        if (target / f"{model_path.stem}.xml").exists():
            return target
        from ultralytics import YOLO
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / model_path.name
            shutil.copy2(model_path, src)
            out = Path(YOLO(str(src)).export(format="openvino", imgsz=(h, w), verbose=False))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(out, target)
    return target


class PoseTracker:
    """Runs YOLO pose with persistent tracking on consecutive frames of ONE video.

    Create a new instance for every video so that tracker state never leaks between videos.
    With ``backend="openvino"`` (CPU) the same checkpoint runs through an OpenVINO export made for this
    video's input shape; if that export cannot be made, it falls back to PyTorch and says so in ``warning``.
    """

    def __init__(self, model_cfg, tracking_cfg, fps: float, work_dir: Path, model_path: Path,
                 frame_size: tuple[int, int] | None = None):
        if not model_path.exists():
            raise ModelLoadError(
                f"Pose checkpoint not found: {model_path}\n"
                f"Run `python main.py setup-model` (downloads {model_path.name} into models/)."
            )
        self.cfg = model_cfg
        self.device = resolve_device(model_cfg.device)
        self.imgsz: int | tuple[int, int] = model_cfg.imgsz
        self.backend = "torch"
        self.warning: str | None = None
        load_path, task = model_path, None
        if getattr(model_cfg, "backend", "torch") == "openvino" and self.device == "cpu" and frame_size:
            shape = letterbox_shape(frame_size[0], frame_size[1], model_cfg.imgsz)
            try:
                load_path, task = openvino_model_for(model_path, shape), "pose"
                self.imgsz, self.backend = shape, "openvino"
            except Exception as exc:  # missing openvino package, export error, ... -> same model on PyTorch
                self.warning = f"OpenVINO export failed ({exc}); using PyTorch inference instead (slower)"
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(load_path), task=task)
        except Exception as exc:  # corrupt file, incompatible ultralytics version, ...
            raise ModelLoadError(f"Could not load pose model {load_path}: {exc}") from exc
        if getattr(self.model, "task", None) != "pose":
            raise ModelLoadError(f"{model_path.name} is a '{self.model.task}' model, a pose model is required")
        self.tracker_yaml = write_tracker_yaml(tracking_cfg, fps, work_dir)

    def process(self, frame: np.ndarray) -> list[PersonDetection]:
        res = self.model.track(
            frame, persist=True, tracker=str(self.tracker_yaml), imgsz=self.imgsz,
            conf=self.cfg.detection_conf, iou=self.cfg.iou, device=self.device,
            classes=[0], verbose=False,
        )[0]
        return detections_from_result(res)

    def reset(self) -> None:
        """Drop all tracks (used at camera cuts)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()


def detections_from_result(res) -> list[PersonDetection]:
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return []
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(boxes)
    xy = res.keypoints.xy.cpu().numpy()
    kc = res.keypoints.conf
    conf = kc.cpu().numpy() if kc is not None else np.zeros(xy.shape[:2])
    out = []
    for i in range(len(boxes)):
        kp = np.concatenate([xy[i], conf[i][:, None]], axis=1)
        tid = None if ids[i] is None else int(ids[i])
        out.append(PersonDetection(tid, boxes[i], float(scores[i]), kp))
    return out


# ---------------------------------------------------------------------------- detection cache
# Pose inference dominates run time on CPU. Raw per-frame detections are cached so that
# thresholds can be re-tuned and the interaction logic re-run in seconds.

def cache_key(video_path: Path, model_cfg, tracking_cfg, stride: int, max_frames) -> str:
    st = video_path.stat()
    h = hashlib.md5()
    with open(video_path, "rb") as fh:
        h.update(fh.read(1 << 20))
    backend = getattr(model_cfg, "backend", "torch")
    tag = "" if backend == "torch" else f"{backend}|"   # torch keys unchanged: older caches stay valid
    h.update(f"{st.st_size}|{model_cfg.path}|{tag}{model_cfg.imgsz}|{model_cfg.detection_conf}|"
             f"{model_cfg.iou}|{sorted(vars(tracking_cfg).items())}|{stride}|{max_frames}".encode())
    return h.hexdigest()[:16]


def save_cache(path: Path, meta: dict, frames: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"meta": meta, "frames": frames}, fh)


def load_cache(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["meta"], data["frames"]


def iter_cached_detections(frames: list[dict]) -> Iterator[tuple[int, float, bool, list[PersonDetection]]]:
    for f in frames:
        yield f["i"], f["t"], f.get("cut", False), [PersonDetection.from_json(d) for d in f["p"]]
