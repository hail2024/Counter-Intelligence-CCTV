"""Run every labelled video (data/labels/<video>_gt.csv) with automatic scene setup and score it.

    python tools/benchmark.py              # uses cached pose detections when available
    python tools/benchmark.py --render     # also writes annotated videos

Prints per-video TP/FP/FN for CROSSING, CONTACT, HANDSHAKE and INTERACTION (the counting unit shown in the
dashboard), the interaction count error, and pooled totals, and writes outputs/reports/benchmark.json. No video names or expected results are hard-coded: the list of
videos is whatever has a label file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluation import evaluate, load_labels  # noqa: E402
from src.pipeline import run_video  # noqa: E402

CATS = ("CROSSING", "CONTACT", "HANDSHAKE", "INTERACTION")


def find_video(stem: str) -> Path | None:
    for p in (PROJECT_ROOT / "data" / "input").glob("**/*"):
        if p.stem == stem and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--only", nargs="*", help="video stems to run")
    a = ap.parse_args()
    cfg = load_config()
    total = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATS}
    count = {"predicted": 0, "labelled": 0, "abs_error": 0, "exact_videos": 0, "videos": 0}
    rows = []
    for lab in sorted((PROJECT_ROOT / "data" / "labels").glob("*_gt.csv")):
        stem = lab.name[:-len("_gt.csv")]
        if a.only and stem not in a.only:
            continue
        video = find_video(stem)
        if video is None:
            continue
        res = run_video(video, None, cfg, run_name=stem, labels_path=lab, render=a.render)
        # score EVERY category on every video: a contact predicted where none happened is a false positive
        ev = evaluate(res.ledger.records(), load_labels(lab), 0.5, categories=CATS)["categories"]
        it = ev["INTERACTION"]
        count["predicted"] += it["count_predicted"]
        count["labelled"] += it["count_labelled"]
        count["abs_error"] += it["count_error"]
        count["exact_videos"] += it["count_error"] == 0
        count["videos"] += 1
        line = {"video": stem, "counts": res.ledger.counts(), "interaction_counts": res.ledger.interaction_counts(),
                "interaction_count": [it["count_predicted"], it["count_labelled"]],
                "backend": res.stats.get("inference_backend"),
                "pose_ms_per_frame": res.stats.get("pose_inference_ms_per_frame"),
                "elapsed_s": res.stats.get("elapsed_s"), "events": [
            (e["event_id"], e["event_type"], round(e["start_timestamp"], 2), round(e["end_timestamp"], 2),
             e["handshake_score"]) for e in res.ledger.records()]}
        for c in CATS:
            r = ev.get(c)
            line[c] = None if r is None else (r["tp"], r["fp"], r["fn"])
            if r:
                for k in ("tp", "fp", "fn"):
                    total[c][k] += r[k]
        rows.append(line)
        cells = "  ".join(f"{c[:5]} {'-' if line[c] is None else '%d/%d/%d' % line[c]}" for c in CATS)
        print(f"{stem:<18} {cells}  count {it['count_predicted']}/{it['count_labelled']}   "
              + ", ".join(f"{i['interaction_id']}:{i['outcome'].replace('LIKELY_', '')[:9]}@{i['start_timestamp']}"
                          for i in res.ledger.interactions()), flush=True)
    print("\nTP/FP/FN totals:")
    for c in CATS:
        t = total[c]
        p = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else None
        r = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else None
        print(f"  {c:<10} TP={t['tp']} FP={t['fp']} FN={t['fn']}  precision={p if p is None else round(p, 2)}  "
              f"recall={r if r is None else round(r, 2)}")
    print(f"  interaction count: predicted {count['predicted']} vs labelled {count['labelled']}, "
          f"sum |error| {count['abs_error']}, exact on {count['exact_videos']}/{count['videos']} videos")
    out = PROJECT_ROOT / "outputs" / "reports" / "benchmark.json"
    out.write_text(json.dumps({"videos": rows, "totals": total, "interaction_count": count}, indent=2),
                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
