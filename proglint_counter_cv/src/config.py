"""Loading and validation of the global processing configuration (config/default.yaml).

Every threshold used by the interaction logic is read from here; nothing is hard-coded in the
logic modules. Validation runs at startup so that a bad value fails loudly with a clear message
instead of silently producing wrong events.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


@dataclass
class ModelCfg:
    path: str = "models/yolo26s-pose.pt"
    imgsz: int = 640
    detection_conf: float = 0.30
    iou: float = 0.60
    device: str = "cpu"
    backend: str = "openvino"   # "openvino" (CPU-optimised export of the same checkpoint) | "torch"


@dataclass
class TrackingCfg:
    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.40
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.45
    track_buffer_s: float = 1.0
    match_thresh: float = 0.80
    fuse_score: bool = True


@dataclass
class KeypointCfg:
    wrist_conf: float = 0.50
    elbow_conf: float = 0.40
    shoulder_conf: float = 0.40
    edge_margin_px: float = 2
    hand_extension_ratio: float = 0.35
    fingertip_ratio: float = 0.74
    smoothing_alpha: float = 0.60
    forearm_plausibility: list = field(default_factory=lambda: [0.35, 2.5])
    scale_history: int = 30


@dataclass
class RolesCfg:
    min_person_height_frac: float | str = "auto"
    background_height_ratio: float = 0.35
    staff_rule: str = "activity"
    staff_rule_min_margin: float = 0.5
    confirm_frames: int = 3
    window_frames: int = 15


@dataclass
class CrossingCfg:
    point: str = "hand"
    hysteresis_ratio: float = 0.15
    hysteresis_min_px: float = 4
    persist_s: float = 0.10
    max_gap_s: float = 0.30
    rearm_s: float = 1.0
    interaction_rearm_s: float = 10.0
    approach_ratio: float = 1.0
    approach_hold_s: float = 0.3
    max_episode_s: float = 60


@dataclass
class ContactCfg:
    enter_ratio: float = 0.60
    exit_ratio: float = 0.90
    min_contact_s: float = 0.50
    min_close_ratio: float = 0.70
    max_relative_speed: float = 3.0
    separation_s: float = 0.40
    missing_tolerance_s: float = 0.70
    cooldown_s: float = 1.0
    uncertain_min_s: float = 0.20
    partial_near_ratio: float = 2.25
    ambiguity_ratio: float = 1.25
    approach_ratio: float = 2.0
    touch_enter_ratio: float = 0.20
    touch_exit_ratio: float = 0.30
    touch_tip_min: float = 0.85
    touch_near_ratio: float = 0.35
    visible_conf: float = 0.95
    touch_hold_max_speed: float = 0.5
    max_episode_s: float = 120
    engagement_window_s: float = 2.0
    engagement_lag_s: float = 0.3
    engagement_min_history_s: float = 0.5
    min_engagement: float = 0.03
    engaged_reach: float = 0.30
    engaged_prior_reach: float = 0.45


@dataclass
class HandshakeCfg:
    score_threshold: float = 0.60
    duration_start_s: float = 0.50
    duration_full_s: float = 1.00
    w_duration: float = 0.30
    w_pumping: float = 0.35
    w_comovement: float = 0.15
    w_extension: float = 0.20
    min_reversals: int = 2
    min_amplitude_ratio: float = 0.06
    max_relative_std: float = 0.40
    min_extension_ratio: float = 1.3
    extension_min_observed: float = 0.30
    min_motion_reversals: int = 2
    clasp_distance: float = 0.20
    min_clasp_frac: float = 0.30
    clasp_window_s: float = 0.5


@dataclass
class ClassifierCfg:
    enabled: bool = True
    model_path: str = "models/interaction_model.json"
    contact_threshold: float = 0.5
    handshake_threshold: float = 0.5
    uncertain_min_prob: float = 0.25
    one_sided_factor: float = 0.3
    stable_s: float = 0.3


@dataclass
class InteractionsCfg:
    join_gap_s: float = 1.0
    link_paired_crossings: bool = True


@dataclass
class IdentityCfg:
    switch_window_s: float = 1.0
    switch_radius_ratio: float = 2.0


@dataclass
class SceneCfg:
    cut_detection: bool = True
    cut_threshold: float = 0.45
    view_similarity_min: float = 0.55


@dataclass
class VideoCfg:
    stride: int = 1
    max_frames: int | None = None


@dataclass
class OutputCfg:
    annotated_dir: str = "outputs/annotated"
    events_dir: str = "outputs/events"
    reports_dir: str = "outputs/reports"
    cache_dir: str = "outputs/cache"
    h264: bool = True
    trail_frames: int = 20


@dataclass
class AppConfig:
    model: ModelCfg = field(default_factory=ModelCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    keypoints: KeypointCfg = field(default_factory=KeypointCfg)
    roles: RolesCfg = field(default_factory=RolesCfg)
    crossing: CrossingCfg = field(default_factory=CrossingCfg)
    contact: ContactCfg = field(default_factory=ContactCfg)
    handshake: HandshakeCfg = field(default_factory=HandshakeCfg)
    classifier: ClassifierCfg = field(default_factory=ClassifierCfg)
    interactions: InteractionsCfg = field(default_factory=InteractionsCfg)
    identity: IdentityCfg = field(default_factory=IdentityCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve_path(self, p: str | Path) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _build(cls, data: dict | None, where: str):
    """Instantiate dataclass ``cls`` from ``data``, rejecting unknown keys (typos)."""
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{where}': {sorted(unknown)}")
    defaults = cls()  # every config dataclass has defaults for all fields
    kwargs = {}
    for name in known:
        if name not in data:
            continue
        default = getattr(defaults, name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), data[name], f"{where}.{name}")
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_from_dict(data: dict) -> AppConfig:
    cfg = _build(AppConfig, data, "config")
    validate_config(cfg)
    return cfg


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> AppConfig:
    """Load YAML config, apply optional overrides (e.g. from a camera file or the UI), validate."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = deep_merge(data, overrides)
    return config_from_dict(data)


def _num(value: Any, name: str, lo: float | None = None, hi: float | None = None,
         lo_inclusive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if lo is not None and (value < lo or (value == lo and not lo_inclusive)):
        op = ">=" if lo_inclusive else ">"
        raise ConfigError(f"{name} must be {op} {lo}, got {value}")
    if hi is not None and value > hi:
        raise ConfigError(f"{name} must be <= {hi}, got {value}")


def validate_config(cfg: AppConfig) -> None:
    """Range and consistency checks. Raises ConfigError with a readable message."""
    m = cfg.model
    _num(m.imgsz, "model.imgsz", 32)
    if m.imgsz % 32:
        raise ConfigError(f"model.imgsz must be a multiple of 32, got {m.imgsz}")
    _num(m.detection_conf, "model.detection_conf", 0, 1)
    _num(m.iou, "model.iou", 0, 1)
    if not str(m.path).endswith(".pt"):
        raise ConfigError(f"model.path must point to a .pt pose checkpoint, got {m.path!r}")
    if m.backend not in ("openvino", "torch"):
        raise ConfigError(f"model.backend must be 'openvino' or 'torch', got {m.backend!r}")

    t = cfg.tracking
    if t.tracker_type not in ("bytetrack", "botsort"):
        raise ConfigError(f"tracking.tracker_type must be 'bytetrack' or 'botsort', got {t.tracker_type!r}")
    for n in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        _num(getattr(t, n), f"tracking.{n}", 0, 1)
    if t.track_low_thresh > t.track_high_thresh:
        raise ConfigError("tracking.track_low_thresh must not exceed track_high_thresh")
    _num(t.track_buffer_s, "tracking.track_buffer_s", 0, lo_inclusive=False)

    k = cfg.keypoints
    for n in ("wrist_conf", "elbow_conf", "shoulder_conf"):
        _num(getattr(k, n), f"keypoints.{n}", 0, 1)
    _num(k.hand_extension_ratio, "keypoints.hand_extension_ratio", 0, 1.5)
    _num(k.smoothing_alpha, "keypoints.smoothing_alpha", 0, 1, lo_inclusive=False)
    fp = k.forearm_plausibility
    if not (isinstance(fp, (list, tuple)) and len(fp) == 2 and 0 < fp[0] < 1 < fp[1]):
        raise ConfigError(f"keypoints.forearm_plausibility must be [low<1, high>1], got {fp}")
    _num(k.scale_history, "keypoints.scale_history", 1)

    r = cfg.roles
    _num(r.background_height_ratio, "roles.background_height_ratio", 0, 1)
    if r.min_person_height_frac != "auto":
        _num(r.min_person_height_frac, "roles.min_person_height_frac", 0, 1)
    _num(r.confirm_frames, "roles.confirm_frames", 1)
    _num(r.window_frames, "roles.window_frames", r.confirm_frames)

    c = cfg.crossing
    if c.point not in ("hand", "wrist"):
        raise ConfigError(f"crossing.point must be 'hand' or 'wrist', got {c.point!r}")
    _num(c.hysteresis_ratio, "crossing.hysteresis_ratio", 0)
    _num(c.hysteresis_min_px, "crossing.hysteresis_min_px", 0)
    for n in ("persist_s", "max_gap_s", "rearm_s", "approach_ratio"):
        _num(getattr(c, n), f"crossing.{n}", 0)
    _num(c.max_episode_s, "crossing.max_episode_s", 0, lo_inclusive=False)

    ct = cfg.contact
    _num(ct.enter_ratio, "contact.enter_ratio", 0, lo_inclusive=False)
    _num(ct.exit_ratio, "contact.exit_ratio", 0, lo_inclusive=False)
    if ct.exit_ratio < ct.enter_ratio:
        raise ConfigError("contact.exit_ratio must be >= contact.enter_ratio (hysteresis)")
    _num(ct.min_contact_s, "contact.min_contact_s", 0, lo_inclusive=False)
    _num(ct.min_close_ratio, "contact.min_close_ratio", 0, 1)
    for n in ("max_relative_speed", "separation_s", "missing_tolerance_s", "cooldown_s",
              "uncertain_min_s", "partial_near_ratio", "approach_ratio"):
        _num(getattr(ct, n), f"contact.{n}", 0)
    _num(ct.ambiguity_ratio, "contact.ambiguity_ratio", 1)
    _num(ct.max_episode_s, "contact.max_episode_s", 0, lo_inclusive=False)
    if ct.uncertain_min_s > ct.min_contact_s:
        raise ConfigError("contact.uncertain_min_s must not exceed contact.min_contact_s")

    h = cfg.handshake
    _num(h.score_threshold, "handshake.score_threshold", 0, 1, lo_inclusive=False)
    _num(h.duration_start_s, "handshake.duration_start_s", 0)
    _num(h.duration_full_s, "handshake.duration_full_s", h.duration_start_s, lo_inclusive=False)
    weights = [h.w_duration, h.w_pumping, h.w_comovement, h.w_extension]
    for n, w in zip(("w_duration", "w_pumping", "w_comovement", "w_extension"), weights):
        _num(w, f"handshake.{n}", 0, 1)
    if abs(sum(weights) - 1.0) > 0.01:
        raise ConfigError(f"handshake weights must sum to 1, got {sum(weights):.2f}")
    _num(h.min_reversals, "handshake.min_reversals", 1)
    _num(h.min_amplitude_ratio, "handshake.min_amplitude_ratio", 0)
    _num(h.max_relative_std, "handshake.max_relative_std", 0, lo_inclusive=False)
    _num(h.min_extension_ratio, "handshake.min_extension_ratio", 0)
    _num(h.extension_min_observed, "handshake.extension_min_observed", 0, 1)

    k = cfg.classifier
    for n in ("contact_threshold", "handshake_threshold", "uncertain_min_prob"):
        _num(getattr(k, n), f"classifier.{n}", 0, 1)
    if k.uncertain_min_prob > k.contact_threshold:
        raise ConfigError("classifier.uncertain_min_prob must not exceed contact_threshold")
    _num(cfg.interactions.join_gap_s, "interactions.join_gap_s", 0, 30)
    if cfg.contact.touch_exit_ratio < cfg.contact.touch_enter_ratio:
        raise ConfigError("contact.touch_exit_ratio must be >= contact.touch_enter_ratio (hysteresis)")
    if cfg.roles.staff_rule not in ("activity", "position"):
        raise ConfigError(f"roles.staff_rule must be 'activity' or 'position', got {cfg.roles.staff_rule!r}")
    _num(cfg.roles.staff_rule_min_margin, "roles.staff_rule_min_margin", 0)
    _num(cfg.identity.switch_window_s, "identity.switch_window_s", 0)
    _num(cfg.identity.switch_radius_ratio, "identity.switch_radius_ratio", 0)
    _num(cfg.scene.cut_threshold, "scene.cut_threshold", 0, 1, lo_inclusive=False)
    _num(cfg.scene.view_similarity_min, "scene.view_similarity_min", -1, 1)
    _num(cfg.video.stride, "video.stride", 1)
    if cfg.video.max_frames is not None:
        _num(cfg.video.max_frames, "video.max_frames", 1)
    _num(cfg.output.trail_frames, "output.trail_frames", 0)
