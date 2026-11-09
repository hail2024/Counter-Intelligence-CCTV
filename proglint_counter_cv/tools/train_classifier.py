"""Train the episode classifier on labelled videos and validate it leave-one-video-out.

    python tools/train_classifier.py

1. Every video with a label file (data/labels/<video>_gt.csv) is processed WITHOUT a model (cached poses,
   no rendering) to collect candidate contact episodes and their pose features.
2. Each episode gets a training label from the ground truth it overlaps:
      handshake  - overlaps a HANDSHAKE label
      contact    - overlaps a CONTACT label (not a handshake)
      ambiguous  - overlaps only an AMBIGUOUS_CONTACT label (e.g. an object hand-over)
      none       - overlaps no contact label (one-sided reach, pass, near-miss)
   contact model:   (handshake + contact) vs none          (ambiguous episodes are not used)
   handshake model: handshake vs (contact + ambiguous)
3. Leave-one-video-out: for each video, both models are trained on the OTHER videos only, the full pipeline
   is run on the held-out video with that model, and its events are scored against its labels.
   These held-out numbers are the honest accuracy estimate.
4. Final models are trained on all videos and saved to models/interaction_model.json.
Reports: outputs/reports/training_report.md / .json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import (CONTACT_FEATURES, HANDSHAKE_FEATURES, MIN_TRAIN_CLOSE_S,  # noqa: E402
                            InteractionClassifier, fit_logistic, fit_tree)
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluation import evaluate, load_labels  # noqa: E402
from src.pipeline import run_video  # noqa: E402

CATS = ("CROSSING", "CONTACT", "HANDSHAKE")
TOL = 0.3
KIND = "logistic"   # "logistic" (shipped; validated better) or "tree" (shallow decision tree); set with --model
BACKEND = None      # inference backend override (--backend openvino|torch); None = config/default.yaml
OUT = PROJECT_ROOT / "models" / "interaction_model.json"   # --out PATH writes the final model elsewhere


def find_video(stem: str) -> Path | None:
    for p in (PROJECT_ROOT / "data" / "input").glob("**/*"):
        if p.stem == stem and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            return p
    return None


def label_of(ep: dict, labels) -> str:
    def hit(cat):
        return any(g.category == cat and min(ep["end"] + TOL, g.end) - max(ep["start"] - TOL, g.start) > 0 for g in labels)
    if hit("HANDSHAKE"):
        return "handshake"
    if hit("CONTACT"):
        return "contact"
    if hit("AMBIGUOUS_CONTACT"):
        return "ambiguous"
    return "none"


def _over(extra: dict) -> dict:
    return {**extra, "model": {"backend": BACKEND}} if BACKEND else extra


def collect() -> list[dict]:
    cfg = load_config(overrides=_over({"classifier": {"enabled": False}}))
    rows = []
    for lab in sorted((PROJECT_ROOT / "data" / "labels").glob("*_gt.csv")):
        stem = lab.name[:-len("_gt.csv")]
        video = find_video(stem)
        if video is None:
            continue
        res = run_video(video, None, cfg, run_name=f"_train_{stem}", render=False)
        labels = load_labels(lab)
        for ep in res.ledger.episodes:
            if ep["features"]:
                rows.append({"video": stem, "label": label_of(ep, labels), **ep})
        print(f"  {stem:<18} {sum(1 for r in rows if r['video'] == stem)} candidate episodes")
    return rows


def train(rows: list[dict], exclude: str | None = None) -> InteractionClassifier:
    tr = [r for r in rows if r["video"] != exclude and r["features"]["close_time"] >= MIN_TRAIN_CLOSE_S]
    c_rows = [r for r in tr if r["label"] in ("handshake", "contact", "none")]
    Xc = np.array([[r["features"][k] for k in CONTACT_FEATURES] for r in c_rows])
    yc = np.array([r["label"] != "none" for r in c_rows], int)
    h_rows = [r for r in tr if r["label"] in ("handshake", "contact", "ambiguous")]
    Xh = np.array([[r["features"][k] for k in HANDSHAKE_FEATURES] for r in h_rows])
    yh = np.array([r["label"] == "handshake" for r in h_rows], int)
    meta = {"trained": dt.datetime.now().isoformat(timespec="seconds"),
            "videos": sorted({r["video"] for r in tr}),
            "contact_examples": {"positive": int(yc.sum()), "negative": int(len(yc) - yc.sum())},
            "handshake_examples": {"positive": int(yh.sum()), "negative": int(len(yh) - yh.sum())}}
    fit = fit_tree if KIND == "tree" else fit_logistic
    meta["model_type"] = KIND
    return InteractionClassifier(fit(Xc, yc, CONTACT_FEATURES), fit(Xh, yh, HANDSHAKE_FEATURES), meta)


def score_video(stem: str, model_path: Path | None) -> dict:
    over = {"classifier": {"model_path": str(model_path)}} if model_path else {}
    cfg = load_config(overrides=_over(over))
    lab = PROJECT_ROOT / "data" / "labels" / f"{stem}_gt.csv"
    res = run_video(find_video(stem), None, cfg, run_name=f"_eval_{stem}", render=False)
    ev = evaluate(res.ledger.records(), load_labels(lab), 0.5, categories=CATS)["categories"]
    return {"video": stem, "events": [(e["event_id"], e["event_type"], round(e["start_timestamp"], 2),
                                       e["handshake_score"], e["contact_score"]) for e in res.ledger.records()],
            **{c: (ev[c]["tp"], ev[c]["fp"], ev[c]["fn"]) for c in CATS}}


def totals(results: list[dict]) -> dict:
    out = {}
    for c in CATS:
        tp, fp, fn = (sum(r[c][i] for r in results) for i in range(3))
        out[c] = {"tp": tp, "fp": fp, "fn": fn,
                  "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    return out


def main() -> int:
    global KIND, BACKEND, OUT
    if "--model" in sys.argv:
        KIND = sys.argv[sys.argv.index("--model") + 1]
    if "--backend" in sys.argv:
        BACKEND = sys.argv[sys.argv.index("--backend") + 1]
    if "--out" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
    print(f"1) collecting candidate episodes (no model)  [model type: {KIND}, backend: {BACKEND or 'config'}]")
    rows = collect()
    counts = {k: sum(r["label"] == k for r in rows) for k in ("handshake", "contact", "ambiguous", "none")}
    print(f"   episode labels: {counts}")
    videos = sorted({r["video"] for r in rows} | {p.name[:-7] for p in (PROJECT_ROOT / "data" / "labels").glob("*_gt.csv")
                                                  if find_video(p.name[:-7])})

    print("2) leave-one-video-out validation (model never sees the video it is scored on)")
    lovo = []
    with tempfile.TemporaryDirectory() as tmp:
        for v in videos:
            mp = train(rows, exclude=v).save(Path(tmp) / f"m_{v}.json")
            r = score_video(v, mp)
            lovo.append(r)
            print(f"   {v:<18} " + "  ".join(f"{c[:5]} {r[c][0]}/{r[c][1]}/{r[c][2]}" for c in CATS)
                  + "   " + ", ".join(f"{e[1].replace('LIKELY_', '')[:9]}@{e[2]}" for e in r["events"]))
    lovo_tot = totals(lovo)

    print("3) final model on all videos")
    final = train(rows)
    final.meta["leave_one_video_out"] = lovo_tot
    final.meta["inference_backend"] = BACKEND or load_config().model.backend
    out = final.save(OUT)
    print(f"   saved {out}")

    rep_dir = PROJECT_ROOT / "outputs" / "reports"
    suffix = "" if OUT == PROJECT_ROOT / "models" / "interaction_model.json" else f"_{OUT.stem}"
    (rep_dir / f"training_report{suffix}.json").write_text(json.dumps(
        {"episode_label_counts": counts, "leave_one_video_out": {"per_video": lovo, "totals": lovo_tot},
         "episodes": [{k: r[k] for k in ("video", "label", "start", "end", "features")} for r in rows],
         "model": json.loads(out.read_text())}, indent=2), encoding="utf-8")
    md = ["# Classifier training report", "",
          f"Trained {final.meta['trained']} on {len(videos)} labelled videos, {len(rows)} candidate episodes "
          f"({counts}).", "",
          "## Leave-one-video-out results (each video scored by a model that never saw it)", "",
          "| Video | Crossing TP/FP/FN | Contact TP/FP/FN | Handshake TP/FP/FN |", "|---|---|---|---|"]
    md += [f"| {r['video']} | {'/'.join(map(str, r['CROSSING']))} | {'/'.join(map(str, r['CONTACT']))} | "
           f"{'/'.join(map(str, r['HANDSHAKE']))} |" for r in lovo]
    md += ["", "| Category | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    md += [f"| {c} | {t['tp']} | {t['fp']} | {t['fn']} | {t['precision']} | {t['recall']} |" for c, t in lovo_tot.items()]
    md += ["", f"## Final model ({KIND})", ""]
    for name, m in (("contact", final.contact), ("handshake", final.handshake)):
        if hasattr(m, "rules"):
            md.append(f"- **{name}** (decision tree):")
            md += [f"  - {r}" for r in m.rules()]
        else:
            md.append(f"- **{name}**: " + ", ".join(f"{f} {c:+.2f}" for f, c in zip(m.features, m.coef))
                      + f", intercept {m.intercept:+.2f}")
    (rep_dir / f"training_report{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nLeave-one-video-out totals:")
    for c, t in lovo_tot.items():
        print(f"   {c:<10} TP={t['tp']} FP={t['fp']} FN={t['fn']}  precision={t['precision']}  recall={t['recall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
