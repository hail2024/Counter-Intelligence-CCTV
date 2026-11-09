"""End-to-end processing of one video. Used by both the CLI (main.py) and the UI (app.py).

Pass 1  pose + tracking on every analysed frame (independent of the counter line) -> cached detections.
Setup   counter line + customer/staff sides: a saved camera calibration if given, otherwise derived
        automatically from the two main participants tracked through the whole video.
Pass 2  interaction engine + annotated video, then event ledger export, run report and evaluation.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from .calibration import CameraCalibration, auto_calibrate, now_iso
from .config import AppConfig, config_from_dict, deep_merge, load_config
from .engine import FrameResult, InteractionEngine
from .interaction_state import STATE_PRIORITY, State
from .evaluation import evaluate, load_labels, write_report
from .event_logger import EventLedger
from .pose_estimator import PersonDetection, PoseTracker, cache_key, load_cache, save_cache
from .video_io import (AnnotatedVideoWriter, SceneCutDetector, VideoMeta, fingerprint_similarity,
                       iter_frames, probe_video, read_frame, view_fingerprint)
from .visualization import Renderer, draw_calibration

Progress = Callable[[float, str], None]
DETECT_SHARE = 0.8  # share of the progress bar used by pass 1 when inference is needed


@dataclass
class RunResult:
    video: str
    run_name: str
    meta: dict
    ledger: EventLedger
    calibration: CameraCalibration
    auto_calibrated: bool
    calibration_preview: Path | None
    annotated_path: Path | None
    events_csv: Path
    events_json: Path
    interactions_csv: Path
    report_json: Path
    report_md: Path
    evaluation: dict | None = None
    evaluation_md: Path | None = None
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


FINAL_STATE = {"LIKELY_HAND_CONTACT": State.LIKELY_HAND_CONTACT, "LIKELY_HANDSHAKE": State.LIKELY_HANDSHAKE,
               "UNCERTAIN": State.UNCERTAIN}


def apply_final(res: FrameResult, final_types: dict[str, str], final_episodes: list[dict] | None = None,
                validate_s: float = 0.5) -> FrameResult:
    """Show each episode's FINAL classification while it is on screen.

    An episode's record can be corrected when it ends (the decision on its complete evidence). The video is drawn
    in a second pass that already knows those final types, so a frame never says LIKELY_HANDSHAKE for an episode the
    event log reports as LIKELY_HAND_CONTACT (or the reverse). Only decided states are replaced; approaching,
    crossing, POSSIBLE_HAND_CONTACT (validation still in progress) and in-frame UNCERTAIN evidence stay as they were.
    """
    if not final_types and not final_episodes:
        return res
    for pv in res.pairs:
        ft = final_types.get(pv.event_id) if pv.event_id else None
        if ft is not None and pv.state in (State.LIKELY_HAND_CONTACT, State.LIKELY_HANDSHAKE):
            pv.state, pv.event_type = FINAL_STATE[ft], ft
        elif pv.state == State.POSSIBLE_HAND_CONTACT and final_episodes:
            # an episode that ended UNCERTAIN: once its validation window has passed without confirmation, show it
            for ep in final_episodes:
                if (ep["customer_track_id"] == pv.customer_id and ep["staff_track_id"] == pv.staff_id
                        and ep["outcome"] == "UNCERTAIN" and ep["start"] + validate_s <= res.t <= ep["end"] + 0.05):
                    pv.state, pv.event_type, pv.event_id = State.UNCERTAIN, "UNCERTAIN", ep["event_id"]
                    break
    states = [pv.state for pv in res.pairs]
    states += [State.BOUNDARY_CROSSED if v.crossing == "crossed" else State.HAND_APPROACHING
               for v in res.persons if v.crossing != "idle"]
    if not res.paused:
        res.overall_state = min(states, key=STATE_PRIORITY.index) if states else State.NO_ACTIVE_INTERACTION
    return res


def _logic_pass(frames: list[dict], cfg: AppConfig, cal: CameraCalibration, meta: VideoMeta, video_path: Path,
                auto: bool) -> EventLedger:
    """Interaction logic only (no drawing), to learn every episode's final classification before rendering."""
    ledger = EventLedger()
    engine = InteractionEngine(cfg, cal, meta.fps, cfg.video.stride, ledger)
    for f in frames:
        if f.get("cut"):
            sim = None
            if not auto and cal.fingerprint:
                sim = fingerprint_similarity(view_fingerprint(read_frame(video_path, f["i"])), cal.fingerprint)
            engine.scene_cut(f["i"], f["t"], sim is None or sim >= cfg.scene.view_similarity_min, sim)
        engine.process(f["i"], f["t"], [PersonDetection.from_json(d) for d in f["p"]], meta.size)
    engine.finish(frames[-1]["i"], frames[-1]["t"])
    return ledger


def effective_config(cal: CameraCalibration | None, base_path: Path | None = None,
                     ui_overrides: dict | None = None) -> AppConfig:
    """Global config + per-camera overrides + UI overrides (later wins)."""
    merged = deep_merge((cal.overrides if cal else None) or {}, ui_overrides or {})
    return load_config(base_path, overrides=merged)


def detect_people(video_path: Path, meta: VideoMeta, cfg: AppConfig, use_cache: bool,
                  progress: Progress) -> tuple[list[dict], bool, float, dict]:
    """Pass 1: pose + tracking for every analysed frame.
    Returns (frames, from_cache, ms_per_frame, info) where info names the inference backend actually used."""
    stride = cfg.video.stride
    cache_dir = cfg.resolve_path(cfg.output.cache_dir)
    key = cache_key(video_path, cfg.model, cfg.tracking, stride, cfg.video.max_frames)
    cpath = cache_dir / f"{video_path.stem}_{key}.json.gz"
    if use_cache and cpath.exists():
        cmeta, frames = load_cache(cpath)
        return frames, True, 0.0, {"backend": cmeta.get("backend", "torch"), "warning": None}
    progress(0.0, "loading the pose model (first use of a new video shape prepares a fast CPU copy, ~30 s)")
    tracker = PoseTracker(cfg.model, cfg.tracking, meta.fps / stride, cache_dir, cfg.resolve_path(cfg.model.path),
                          meta.size)
    cut_det = SceneCutDetector(cfg.scene.cut_threshold) if cfg.scene.cut_detection else None
    total = (meta.frame_count // stride) or 0
    if cfg.video.max_frames:
        total = min(total or cfg.video.max_frames, cfg.video.max_frames)
    frames, infer, t0 = [], 0.0, time.perf_counter()
    for idx, t, frame in iter_frames(meta, stride, cfg.video.max_frames):
        cut = bool(cut_det.update(frame)) if cut_det is not None else False
        if cut:
            tracker.reset()
        s = time.perf_counter()
        dets = tracker.process(frame)
        infer += time.perf_counter() - s
        frames.append({"i": idx, "t": round(t, 4), "cut": cut, "p": [d.to_json() for d in dets]})
        n = len(frames)
        if n % 5 == 0 or n == total:
            rate = n / (time.perf_counter() - t0)
            progress(DETECT_SHARE * (n / total if total else 0.0),
                     f"detecting people & poses: frame {n}/{total or '?'} ({rate:.1f} fps)")
    if not frames:
        raise RuntimeError("no frames could be read from the video")
    backend, warning = getattr(tracker, "backend", "torch"), getattr(tracker, "warning", None)
    save_cache(cpath, {"video": video_path.name, "model": cfg.model.path, "fps": meta.fps, "backend": backend,
                       "imgsz": getattr(tracker, "imgsz", cfg.model.imgsz)}, frames)
    return frames, False, 1000 * infer / len(frames), {"backend": backend, "warning": warning}


def run_video(video_path: str | Path, cal: CameraCalibration | None, cfg: AppConfig, run_name: str | None = None,
              labels_path: str | Path | None = None, render: bool = True, use_cache: bool = True,
              progress: Progress | None = None, tolerance_s: float = 0.5, swap_roles: bool = False) -> RunResult:
    """Process one video. ``cal=None`` -> the counter line and sides are set up automatically."""
    video_path = Path(video_path)
    progress = progress or (lambda f, m: None)
    meta = probe_video(video_path)
    run_name = run_name or video_path.stem
    warnings: list[str] = []
    t_start = time.perf_counter()

    frames, cached, ms_pose, infer_info = detect_people(video_path, meta, cfg, use_cache, progress)
    if infer_info["warning"]:
        warnings.append(infer_info["warning"])

    # ------------------------------------------------------------------ scene setup
    first = read_frame(video_path, frames[0]["i"])
    auto = cal is None
    setup_info: dict = {}
    if auto:
        progress(DETECT_SHARE, "placing the counter line between the two main people")
        cal, setup_info = auto_calibrate(frames, meta.size, f"{run_name} (auto)", cfg.keypoints.shoulder_conf,
                                         swap_roles, cfg.roles.background_height_ratio, cfg.roles.staff_rule,
                                         cfg.roles.staff_rule_min_margin)
        cal.fingerprint, cal.source_video, cal.created = view_fingerprint(first), video_path.name, now_iso()
    else:
        cal = cal.adapted_to(meta.size)   # raises if the aspect ratio does not match
        if swap_roles:
            cal.customer_region, cal.staff_region = cal.staff_region, cal.customer_region
        sim0 = fingerprint_similarity(view_fingerprint(first), cal.fingerprint) if cal.fingerprint else None
        if sim0 is None:
            warnings.append("calibration has no reference fingerprint: cannot verify it matches this camera view")
        elif sim0 < cfg.scene.view_similarity_min:
            warnings.append(f"first frame differs from the saved calibration (similarity {sim0:.2f}); "
                            "check it belongs to this camera, or use automatic setup")
    over = dict(cal.overrides or {})
    if cfg.roles.min_person_height_frac == "auto" and "min_person_height_frac" not in over.get("roles", {}):
        try:  # saved/manual line: still filter background people using the tracked participants
            _, info = auto_calibrate(frames, meta.size, "size-filter", cfg.keypoints.shoulder_conf,
                                     background_ratio=cfg.roles.background_height_ratio)
            over = deep_merge(over, {"roles": {"min_person_height_frac": round(info["min_person_height_px"] / meta.height, 3)}})
        except Exception:
            over = deep_merge(over, {"roles": {"min_person_height_frac": 0.0}})
    cfg = config_from_dict(deep_merge(cfg.to_dict(), over))

    rep_dir = cfg.resolve_path(cfg.output.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    preview = rep_dir / f"{run_name}_setup.png"
    mid = read_frame(video_path, frames[len(frames) // 2]["i"])
    cv2.imwrite(str(preview), draw_calibration(mid, cal))
    if auto:
        cal.save(rep_dir / f"{run_name}_auto_calibration.yaml")

    # ------------------------------------------------------------------ pass 2: interaction logic + video
    stride = cfg.video.stride
    final = _logic_pass(frames, cfg, cal, meta, video_path, auto) if render else None
    final_types = {e.event_id: e.event_type for e in final.events} if final is not None else {}
    ledger = EventLedger()
    engine = InteractionEngine(cfg, cal, meta.fps, stride, ledger)
    renderer = Renderer(cal, meta.size, cfg, final) if render else None
    writer = AnnotatedVideoWriter(cfg.resolve_path(cfg.output.annotated_dir) / f"{run_name}_annotated.mp4",
                                  meta.fps / stride, meta.size, cfg.output.h264) if render else None
    frame_iter = iter_frames(meta, stride, cfg.video.max_frames) if render else None
    cuts = []
    last_idx, last_t = frames[0]["i"], frames[0]["t"]
    base = 0.0 if cached else DETECT_SHARE
    annotated = None
    try:
        for n, f in enumerate(frames, 1):
            idx, t = f["i"], f["t"]
            frame = next(frame_iter)[2] if frame_iter is not None else None
            if f.get("cut"):
                sim = None
                if not auto and cal.fingerprint and frame is not None:
                    sim = fingerprint_similarity(view_fingerprint(frame), cal.fingerprint)
                ok = sim is None or sim >= cfg.scene.view_similarity_min
                engine.scene_cut(idx, t, ok, sim)
                cuts.append({"frame": idx, "t": round(t, 3), "view_similarity": sim, "analysis": "on" if ok else "paused"})
            res = apply_final(engine.process(idx, t, [PersonDetection.from_json(d) for d in f["p"]], meta.size),
                              final_types, final.episodes if final is not None else None, cfg.contact.min_contact_s)
            ledger.record_states(t, res.overall_state.value,
                                 {f"C{pv.customer_id}-S{pv.staff_id}": pv.state.value for pv in res.pairs})
            if writer is not None:
                writer.write(renderer.draw(frame, res, ledger, cfg.keypoints.wrist_conf))
            last_idx, last_t = idx, t
            if n % 10 == 0 or n == len(frames):
                progress(base + (1 - base) * 0.95 * n / len(frames), f"analysing interactions: frame {n}/{len(frames)}")
    finally:
        engine.finish(last_idx, last_t)
        if writer is not None:
            progress(0.98, "encoding annotated video")
            try:
                annotated = writer.close()
            except Exception as exc:  # report, never pretend the video exists
                warnings.append(f"annotated video NOT saved: {exc}")
            if writer.warning:
                warnings.append(writer.warning)
    elapsed = time.perf_counter() - t_start

    stats = {
        "frames_processed": len(frames), "stride": stride, "elapsed_s": round(elapsed, 2),
        "throughput_fps": round(len(frames) / elapsed, 2) if elapsed > 0 else None,
        "pose_inference_ms_per_frame": round(ms_pose, 1) if not cached else None,
        "detections_from_cache": cached, "scene_cuts": cuts, "id_switches": engine.id_switches,
        "untracked_detections": engine.registry.untracked_detections,
        "setup": "automatic" if auto else f"saved calibration '{cal.name}'", "setup_info": setup_info,
        "device": cfg.model.device, "inference_backend": infer_info["backend"], "platform": platform.platform(),
    }
    run_meta = {"video": video_path.name, "video_meta": meta.to_dict(), "camera": cal.name,
                "model": cfg.model.path, "config": cfg.to_dict(), "stats": stats, "warnings": warnings}
    ev_dir = cfg.resolve_path(cfg.output.events_dir)
    events_json = ledger.write_json(ev_dir / f"{run_name}_events.json",
                                    {k: run_meta[k] for k in ("video", "camera", "model", "stats", "warnings")})
    events_csv = ledger.write_csv(ev_dir / f"{run_name}_events.csv")
    interactions_csv = ledger.write_interactions_csv(ev_dir / f"{run_name}_interactions.csv")

    evaluation = eval_md = None
    if labels_path:
        evaluation = evaluate(ledger.records(), load_labels(labels_path), tolerance_s)
        eval_md = rep_dir / f"{run_name}_evaluation.md"
        write_report(evaluation, rep_dir / f"{run_name}_evaluation.json", eval_md, video_path.name)

    report_json = rep_dir / f"{run_name}_run.json"
    report_json.write_text(json.dumps({**run_meta, "counts": ledger.counts(),
                                       "interaction_counts": ledger.interaction_counts(), "notes": ledger.notes,
                                       "calibration": cal.to_dict() | {"fingerprint": "..."},
                                       "annotated_video": str(annotated) if annotated else None,
                                       "evaluation": evaluation}, indent=2, default=str), encoding="utf-8")
    report_md = rep_dir / f"{run_name}_run.md"
    report_md.write_text(_run_markdown(run_meta, ledger, annotated, cal), encoding="utf-8")
    progress(1.0, "done")
    return RunResult(str(video_path), run_name, run_meta, ledger, cal, auto, preview, annotated, events_csv,
                     events_json, interactions_csv, report_json, report_md, evaluation, eval_md, warnings, stats)


def _run_markdown(meta: dict, ledger: EventLedger, annotated, cal: CameraCalibration) -> str:
    vm, st = meta["video_meta"], meta["stats"]
    c = ledger.counts()
    ic = ledger.interaction_counts()
    lines = [
        f"# Run report - {meta['video']}", "",
        f"- Video: {vm['width']}x{vm['height']}, {vm['fps']:.2f} FPS, {vm['frame_count']} frames ({vm['duration_s']:.1f} s)",
        f"- Scene setup: {st['setup']} - {cal.notes}",
        f"- Frames analysed: {st['frames_processed']} (stride {st['stride']}), wall time {st['elapsed_s']} s"
        + (f", pose {st['pose_inference_ms_per_frame']} ms/frame" if st['pose_inference_ms_per_frame'] else " (detections from cache)"),
        f"- Inference backend: {st.get('inference_backend', 'n/a')}",
        f"- Annotated video: {annotated or 'NOT SAVED'}", "",
        f"**{ic['total']} interaction(s)**: {ic['LIKELY_HANDSHAKE']} likely handshake, {ic['LIKELY_HAND_CONTACT']} likely "
        f"hand contact, {ic['UNCERTAIN']} uncertain, {ic['BOUNDARY_CROSSING']} reach without contact.", "",
        "## Interactions (counting unit)", "",
        "| ID | Outcome | Start | End | Crossings | Rows | What it looks like |", "|---|---|---|---|---|---|---|",
    ]
    for i in ledger.interactions():
        end = "-" if i["end_timestamp"] is None else f"{i['end_timestamp']:.2f}"
        lines.append(f"| {i['interaction_id']} | {i['outcome']} | {i['start_timestamp']:.2f} | {end} | "
                     f"{i['crossing_count']} | {', '.join(i['event_ids'])} | {i['interpretation']} |")
    lines += [
        "", "## Observation rows (event log)", "",
        "| BOUNDARY_CROSSING | LIKELY_HAND_CONTACT | LIKELY_HANDSHAKE | UNCERTAIN | suppressed candidates |",
        "|---|---|---|---|---|",
        f"| {c['BOUNDARY_CROSSING']} | {c['LIKELY_HAND_CONTACT']} | {c['LIKELY_HANDSHAKE']} | {c['UNCERTAIN']} | {len(ledger.suppressed)} |",
        "", "| ID | Interaction | Type | Start | End | Handshake % | What it looks like |", "|---|---|---|---|---|---|---|",
    ]
    for e in ledger.records():
        lines.append(f"| {e['event_id']} | {e['interaction_id']} | {e['event_type']} | {e['start_timestamp']:.2f} | {e['end_timestamp']:.2f} | "
                     f"{'-' if e['handshake_score'] is None else e['handshake_score']} | {e['interpretation']} |")
    if meta["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in meta["warnings"]]
    if ledger.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in ledger.notes]
    return "\n".join(lines) + "\n"


def make_calibration_from_line(name: str, frame, p0, p1, source_video: str, swap_roles: bool = False) -> CameraCalibration:
    """A calibration from a single user-drawn counter line: each side of the line is one role's region.

    The customer side is the left side (or the top side for a mostly horizontal line); swap_roles flips it.
    """
    import numpy as np

    from .calibration import _clip_segment_to_rect
    from .geometry import clip_polygon_halfplane, frame_rect, signed_distance_to_polyline
    h, w = frame.shape[:2]
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    if np.hypot(*d) < 10:
        raise ValueError("the two points are too close together - click two points along the counter")
    seg = _clip_segment_to_rect(p0 - d * 100, p1 + d * 100, w, h)  # extend the line across the frame
    q0, q1 = seg
    rect = frame_rect(w, h)
    left = clip_polygon_halfplane(rect, q0, q1, keep_left=True)
    right = clip_polygon_halfplane(rect, q0, q1, keep_left=False)
    probe = (0.0, h / 2) if abs(d[1]) >= abs(d[0]) else (w / 2, 0.0)   # left edge, or top edge for a flat line
    cust, staff = (left, right) if signed_distance_to_polyline(probe, [q0, q1]) > 0 else (right, left)
    if swap_roles:
        cust, staff = staff, cust
    cal = CameraCalibration(name=name, reference_size=(w, h), boundary=[list(q0), list(q1)], customer_region=cust,
                            staff_region=staff, fingerprint=view_fingerprint(frame), source_video=source_video,
                            created=now_iso(), notes="counter line drawn by the user")
    cal.validate()
    return cal
