"""Command-line interface.

Examples
  python main.py setup-model
  python main.py run --video any_video.mp4                                   # automatic counter line & roles
  python main.py run --video data/input/real_test.mp4 --labels data/labels/real_test_gt.csv
  python main.py run --video cam1.mp4 --camera config/cameras/cam1.yaml      # saved camera setup
  python main.py preview --video cam1.mp4 --camera config/cameras/cam1.yaml
  python main.py evaluate --events outputs/events/real_test_events.json --labels data/labels/real_test_gt.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.calibration import CalibrationError, CameraCalibration
from src.config import PROJECT_ROOT, ConfigError, load_config
from src.evaluation import LabelError, evaluate, load_labels, report_markdown
from src.pose_estimator import ModelLoadError
from src.video_io import VideoError, read_frame


def _progress(frac: float, msg: str) -> None:
    bar = "#" * int(frac * 30)
    print(f"\r[{bar:<30}] {frac * 100:5.1f}%  {msg:<60}", end="", flush=True)


def cmd_run(a) -> int:
    from src.pipeline import effective_config, run_video
    cal = CameraCalibration.load(a.camera) if a.camera else None
    overrides = {"video": {}}
    if a.max_frames:
        overrides["video"]["max_frames"] = a.max_frames
    if a.stride:
        overrides["video"]["stride"] = a.stride
    cfg = effective_config(cal, Path(a.config) if a.config else None, overrides)
    res = run_video(a.video, cal, cfg, a.name, a.labels, render=not a.no_video, use_cache=not a.no_cache,
                    progress=_progress, tolerance_s=a.tolerance, swap_roles=a.swap_roles)
    print()
    c = res.ledger.counts()
    print(f"Frames analysed  : {res.stats['frames_processed']}  "
          f"({'cached detections' if res.stats['detections_from_cache'] else str(res.stats['pose_inference_ms_per_frame']) + ' ms/frame pose'})")
    print(f"Scene setup      : {res.stats['setup']} - {res.calibration.notes}")
    print(f"Setup preview    : {res.calibration_preview}")
    ic = res.ledger.interaction_counts()
    print(f"Inference        : {res.stats.get('inference_backend', 'n/a')} backend")
    print(f"Interactions     : {ic['total']}  (handshake {ic['LIKELY_HANDSHAKE']}, contact {ic['LIKELY_HAND_CONTACT']}, "
          f"uncertain {ic['UNCERTAIN']}, reach only {ic['BOUNDARY_CROSSING']})")
    for i in res.ledger.interactions():
        end = "open" if i["end_timestamp"] is None else f"{i['end_timestamp']:6.2f}"
        print(f"  {i['interaction_id']} {i['outcome']:<20} {i['start_timestamp']:6.2f}-{end}s "
              f"rows {','.join(i['event_ids'])}  {i['interpretation']}")
    print(f"Event rows       : {c}")
    for e in res.ledger.records():
        print(f"  {e['event_id']} [{e['interaction_id']}] {e['event_type']:<20} {e['start_timestamp']:6.2f}-"
              f"{e['end_timestamp']:6.2f}s C={e['customer_track_id']} S={e['staff_track_id']}")
    print(f"Annotated video  : {res.annotated_path or 'NOT SAVED'}")
    print(f"Event log        : {res.events_csv}\n                   {res.events_json}\n"
          f"                   {res.interactions_csv}")
    print(f"Run report       : {res.report_md}")
    if res.evaluation:
        print(f"Evaluation       : {res.evaluation_md}")
        for cat, r in res.evaluation["categories"].items():
            print(f"  {cat:<10} TP={r['tp']} FP={r['fp']} FN={r['fn']} P={r['precision']} R={r['recall']} F1={r['f1']}")
    for w in res.warnings:
        print(f"WARNING: {w}")
    return 0 if res.annotated_path or a.no_video else 2


def cmd_preview(a) -> int:
    import cv2

    from src.video_io import fingerprint_similarity, view_fingerprint
    from src.visualization import draw_calibration
    cal = CameraCalibration.load(a.camera)
    frame = read_frame(a.video, a.frame)
    h, w = frame.shape[:2]
    cal = cal.adapted_to((w, h))
    out = Path(a.out or PROJECT_ROOT / "outputs" / "reports" / f"{Path(a.video).stem}_calibration.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), draw_calibration(frame, cal)):
        raise VideoError(f"could not write {out}")
    if cal.fingerprint:
        print(f"view similarity to calibration reference: {fingerprint_similarity(view_fingerprint(frame), cal.fingerprint):.2f}")
    print(f"calibration preview written to {out}")
    return 0


def cmd_evaluate(a) -> int:
    data = json.loads(Path(a.events).read_text(encoding="utf-8"))
    print(report_markdown(evaluate(data["events"], load_labels(a.labels), a.tolerance), Path(a.events).name))
    return 0


def cmd_setup_model(a) -> int:
    cfg = load_config(a.config)
    target = cfg.resolve_path(cfg.model.path)
    if target.exists():
        print(f"{target} already present")
        return 0
    from ultralytics import YOLO
    target.parent.mkdir(parents=True, exist_ok=True)
    YOLO(target.name)  # Ultralytics downloads official release weights by name into the CWD
    src = Path.cwd() / target.name
    if not src.exists():
        print(f"download failed: {target.name} not found after YOLO() call")
        return 1
    shutil.move(str(src), str(target))
    print(f"downloaded {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Proglint Counter Intelligence - counter crossing & hand-contact analytics")
    p.add_argument("--config", help="global config YAML (default config/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a video")
    r.add_argument("--video", required=True)
    r.add_argument("--camera", help="saved camera calibration YAML (default: automatic setup from the video)")
    r.add_argument("--swap-roles", action="store_true", help="the right/bottom person is the customer")
    r.add_argument("--labels", help="ground-truth CSV for evaluation")
    r.add_argument("--name", help="output name (default: video file stem)")
    r.add_argument("--max-frames", type=int)
    r.add_argument("--stride", type=int)
    r.add_argument("--tolerance", type=float, default=0.5, help="evaluation timestamp tolerance (s)")
    r.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    r.add_argument("--no-cache", action="store_true", help="re-run pose inference even if cached")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("preview", help="draw a saved calibration on a video frame")
    v.add_argument("--video", required=True)
    v.add_argument("--camera", required=True)
    v.add_argument("--frame", type=int, default=0)
    v.add_argument("--out")
    v.set_defaults(fn=cmd_preview)
    e = sub.add_parser("evaluate", help="score an events JSON against labels")
    e.add_argument("--events", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--tolerance", type=float, default=0.5)
    e.set_defaults(fn=cmd_evaluate)
    m = sub.add_parser("setup-model", help="download the configured pose checkpoint into models/")
    m.set_defaults(fn=cmd_setup_model)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (ConfigError, CalibrationError, VideoError, ModelLoadError, LabelError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
