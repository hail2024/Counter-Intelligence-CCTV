"""Event-level evaluation of predicted episodes against manually labelled intervals.

Label file (CSV): ``category,start_timestamp,end_timestamp,role,notes`` plus OPTIONAL columns
``video_id, interaction, customer_track_id, staff_track_id, direction, wrist_visibility, expected_result``
(see data/README.md). Optional columns may be absent or empty; old label files stay valid.

  category  CROSSING            a boundary-crossing reach by one person (optional role customer/staff)
            CONTACT             a customer-staff hand-contact episode (handshakes included)
            HANDSHAKE           a handshake episode (also label it as CONTACT)
            AMBIGUOUS_CONTACT   annotator cannot tell whether the hands touched (e.g. an object
                                hand-over); CONTACT-family predictions overlapping it are excluded
                                from scoring instead of counted as false positives
  role      optional, for CROSSING: customer | staff

Prediction -> category mapping
  BOUNDARY_CROSSING                         -> CROSSING
  LIKELY_HAND_CONTACT, LIKELY_HANDSHAKE     -> CONTACT
  LIKELY_HANDSHAKE                          -> HANDSHAKE
  UNCERTAIN                                 -> not scored; listed with what it overlapped

INTERACTION (the counting unit shown in the dashboard)
  ground truth  label rows sharing an ``interaction`` value form one interaction; rows without that column are
                grouped by time overlap. A group with a CROSSING / CONTACT / HANDSHAKE row is REQUIRED; a group
                of only AMBIGUOUS_* rows is OPTIONAL (an exchange whose type cannot be judged, e.g. an object
                hand-over): a prediction matching it is neither a TP nor an FP, and missing it is not an FN.
  prediction    one per ``interaction_id`` in the event log, spanning its rows.
  count error   |predicted interactions - labelled interactions (required + optional)|, the number a user sees.

Matching rule (per category): a prediction and a label are compatible when their intervals overlap
after widening the prediction by ``tolerance_s`` on both sides (and, for CROSSING with a role on both,
the roles agree). Pairs are matched one-to-one greedily by largest overlap. Unmatched predictions
are false positives, unmatched labels false negatives. Detection delay = prediction confirmation
time - label start.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("CROSSING", "CONTACT", "HANDSHAKE")
VALID_LABELS = CATEGORIES + ("AMBIGUOUS_CONTACT", "AMBIGUOUS_CROSSING")
OPTIONAL_COLUMNS = ("video_id", "interaction", "customer_track_id", "staff_track_id", "direction",
                    "wrist_visibility", "expected_result")
EXPECTED_RESULTS = ("", "event", "uncertain", "no_event")
PRED_MAP = {
    "BOUNDARY_CROSSING": ("CROSSING",),
    "LIKELY_HAND_CONTACT": ("CONTACT",),
    "LIKELY_HANDSHAKE": ("CONTACT", "HANDSHAKE"),
}


class LabelError(ValueError):
    pass


@dataclass
class Interval:
    category: str
    start: float
    end: float
    role: str | None = None
    ref: str = ""        # event id or label row
    confirmed: float | None = None
    interaction: str | None = None              # label: annotator's interaction group
    extra: dict | None = None                   # label: the other optional annotation columns


def load_labels(path: str | Path) -> list[Interval]:
    path = Path(path)
    if not path.exists():
        raise LabelError(f"label file not found: {path}")
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(line for line in fh if not line.lstrip().startswith("#"))]
    for i, r in enumerate(rows, start=2):
        cat = (r.get("category") or "").strip().upper()
        if cat not in VALID_LABELS:
            raise LabelError(f"{path.name} line {i}: unknown category {cat!r} (expected {VALID_LABELS})")
        try:
            s, e = float(r["start_timestamp"]), float(r["end_timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LabelError(f"{path.name} line {i}: bad timestamps") from exc
        if e < s:
            raise LabelError(f"{path.name} line {i}: end before start")
        role = (r.get("role") or "").strip().lower() or None
        extra = {k: (r.get(k) or "").strip() for k in OPTIONAL_COLUMNS if (r.get(k) or "").strip()}
        if extra.get("expected_result", "").lower() not in EXPECTED_RESULTS:
            raise LabelError(f"{path.name} line {i}: expected_result must be one of {EXPECTED_RESULTS[1:]}")
        out.append(Interval(cat, s, e, role, f"L{i}", interaction=extra.pop("interaction", None) or None,
                            extra=extra or None))
    return out


def gt_interactions(labels: list[Interval]) -> list[tuple[Interval, bool]]:
    """Ground-truth interactions as (span, required). Explicit ``interaction`` groups first, then the remaining
    rows merged by time overlap."""
    groups: dict[str, list[Interval]] = {}
    loose = []
    for g in labels:
        (groups.setdefault(g.interaction, []) if g.interaction else loose).append(g)
    merged: list[list[Interval]] = []
    for g in sorted(loose, key=lambda x: x.start):
        if merged and g.start <= max(x.end for x in merged[-1]):
            merged[-1].append(g)
        else:
            merged.append([g])
    clusters = list(groups.values()) + merged
    out = []
    for n, rows in enumerate(clusters, 1):
        required = any(r.category in CATEGORIES for r in rows)
        span = Interval("INTERACTION", min(r.start for r in rows), max(r.end for r in rows),
                        ref="+".join(r.ref for r in rows))
        out.append((span, required))
    out.sort(key=lambda x: x[0].start)
    return out


def predicted_interactions(events: list[dict]) -> list[Interval]:
    spans: dict[str, Interval] = {}
    for e in events:
        iid = e.get("interaction_id") or e["event_id"]
        end = e["end_timestamp"] if e["end_timestamp"] is not None else e["start_timestamp"]
        conf = e.get("confirmed_timestamp")
        if iid not in spans:
            spans[iid] = Interval("INTERACTION", e["start_timestamp"], end, None, iid, conf)
        else:
            sp = spans[iid]
            sp.start, sp.end = min(sp.start, e["start_timestamp"]), max(sp.end, end)
            if conf is not None:
                sp.confirmed = conf if sp.confirmed is None else min(sp.confirmed, conf)
    return sorted(spans.values(), key=lambda x: x.start)


def evaluate_interactions(events: list[dict], labels: list[Interval], tolerance_s: float) -> dict:
    gts = gt_interactions(labels)
    required = [g for g, req in gts if req]
    optional = [g for g, req in gts if not req]
    preds = predicted_interactions(events)
    pairs, fps, fns = match(preds, required, tolerance_s)
    # unmatched predictions that overlap an OPTIONAL (ambiguous) exchange are not errors
    opt_pairs, fps, _ = match(fps, optional, tolerance_s)
    tp = len(pairs)
    prec = tp / (tp + len(fps)) if tp + len(fps) else None
    rec = tp / (tp + len(fns)) if tp + len(fns) else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return {
        "labels": len(required), "labels_optional": len(optional), "predictions": len(preds), "tp": tp,
        "fp": len(fps), "fn": len(fns), "excluded_ambiguous": len(opt_pairs),
        "precision": None if prec is None else round(prec, 3), "recall": None if rec is None else round(rec, 3),
        "f1": None if f1 is None else round(f1, 3),
        "detection_delay_s": [round(p.confirmed - g.start, 3) for p, g in pairs if p.confirmed is not None],
        "count_predicted": len(preds), "count_labelled": len(gts), "count_error": abs(len(preds) - len(gts)),
        "matches": [{"pred": p.ref, "label": g.ref, "pred_interval": [round(p.start, 2), round(p.end, 2)],
                     "label_interval": [g.start, g.end]} for p, g in pairs],
        "false_positives": [{"pred": p.ref, "interval": [round(p.start, 2), round(p.end, 2)]} for p in fps],
        "false_negatives": [{"label": g.ref, "interval": [g.start, g.end]} for g in fns],
        "excluded": [p.ref for p, _ in opt_pairs],
    }


def predictions_from_events(events: list[dict]) -> tuple[dict[str, list[Interval]], list[Interval]]:
    by_cat: dict[str, list[Interval]] = {c: [] for c in CATEGORIES}
    uncertain = []
    for e in events:
        end = e["end_timestamp"] if e["end_timestamp"] is not None else e["start_timestamp"]
        role = {"customer_to_staff": "customer", "staff_to_customer": "staff"}.get(e.get("crossing_direction"))
        if e["event_type"] == "UNCERTAIN":
            uncertain.append(Interval("UNCERTAIN", e["start_timestamp"], end, None, e["event_id"],
                                      e.get("confirmed_timestamp")))
        for cat in PRED_MAP.get(e["event_type"], ()):
            by_cat[cat].append(Interval(cat, e["start_timestamp"], end, role, e["event_id"],
                                        e.get("confirmed_timestamp")))
    return by_cat, uncertain


def _overlap(a: Interval, b: Interval, tol: float) -> float:
    return min(a.end + tol, b.end) - max(a.start - tol, b.start)


def match(preds: list[Interval], gts: list[Interval], tol: float) -> tuple[list, list, list]:
    cands = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p.role and g.role and p.role != g.role:
                continue
            ov = _overlap(p, g, tol)
            if ov > 0:
                cands.append((ov, i, j))
    cands.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for ov, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((preds[i], gts[j]))
    fps = [p for i, p in enumerate(preds) if i not in used_p]
    fns = [g for j, g in enumerate(gts) if j not in used_g]
    return pairs, fps, fns


def evaluate(events: list[dict], labels: list[Interval], tolerance_s: float = 0.5,
             categories: tuple[str, ...] | None = None) -> dict:
    """Return per-category TP/FP/FN, precision, recall, F1 and delays.

    Only categories that have at least one label (or are listed in ``categories``) are scored, so a
    clip without crossing labels does not report crossing "precision" from nothing.
    """
    by_cat, uncertain = predictions_from_events(events)
    ambiguous = [g for g in labels if g.category == "AMBIGUOUS_CONTACT"]
    scored = (tuple(c for c in categories if c in CATEGORIES) if categories
              else tuple(c for c in CATEGORIES if any(g.category == c for g in labels)))
    report = {"tolerance_s": tolerance_s, "categories": {}, "uncertain_predictions": []}
    for cat in scored:
        gts = [g for g in labels if g.category == cat]
        preds = by_cat[cat]
        excluded = []
        # Predictions overlapping an interval where the truth is undecidable are excluded, not scored:
        #  - contact/handshake predictions inside AMBIGUOUS_CONTACT intervals
        #  - crossing predictions inside labelled contact/ambiguous intervals (hands meeting at the line may cross it)
        if cat == "CROSSING":
            excl_zones = [g for g in labels if g.category in ("CONTACT", "AMBIGUOUS_CONTACT", "AMBIGUOUS_CROSSING")]
        else:
            excl_zones = ambiguous if cat == "CONTACT" else []  # no object hand-over is a handshake
        if excl_zones:
            keep = []
            for p in preds:
                overlaps_gt = any(_overlap(p, g, tolerance_s) > 0 for g in gts)
                if not overlaps_gt and any(_overlap(p, a, tolerance_s) > 0 for a in excl_zones):
                    excluded.append(p)
                else:
                    keep.append(p)
            preds = keep
        pairs, fps, fns = match(preds, gts, tolerance_s)
        tp = len(pairs)
        prec = tp / (tp + len(fps)) if tp + len(fps) else None
        rec = tp / (tp + len(fns)) if tp + len(fns) else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
        delays = [round(p.confirmed - g.start, 3) for p, g in pairs if p.confirmed is not None]
        report["categories"][cat] = {
            "labels": len(gts), "predictions": len(preds) + len(excluded), "tp": tp, "fp": len(fps),
            "fn": len(fns), "excluded_ambiguous": len(excluded),
            "precision": None if prec is None else round(prec, 3),
            "recall": None if rec is None else round(rec, 3),
            "f1": None if f1 is None else round(f1, 3),
            "detection_delay_s": delays,
            "matches": [{"pred": p.ref, "label": g.ref, "pred_interval": [round(p.start, 2), round(p.end, 2)],
                         "label_interval": [g.start, g.end]} for p, g in pairs],
            "false_positives": [{"pred": p.ref, "interval": [round(p.start, 2), round(p.end, 2)]} for p in fps],
            "false_negatives": [{"label": g.ref, "interval": [g.start, g.end]} for g in fns],
            "excluded": [p.ref for p in excluded],
        }
    for u in uncertain:
        over = [g.ref + ":" + g.category for g in labels if _overlap(u, g, tolerance_s) > 0]
        report["uncertain_predictions"].append({"pred": u.ref, "interval": [round(u.start, 2), round(u.end, 2)],
                                                "overlaps_labels": over})
    if labels and (categories is None or "INTERACTION" in categories):
        report["categories"]["INTERACTION"] = evaluate_interactions(events, labels, tolerance_s)
    report["not_scored"] = [c for c in CATEGORIES if c not in scored]
    return report


def report_markdown(report: dict, video_name: str) -> str:
    lines = [f"# Evaluation - {video_name}", "",
             f"Event-level matching, tolerance ±{report['tolerance_s']} s. "
             "Small label sets: treat these numbers as preliminary.", "",
             "| Category | Labels | Pred | TP | FP | FN | Excl. | Precision | Recall | F1 | Delay (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
    for cat, r in report["categories"].items():
        d = r["detection_delay_s"]
        lines.append(f"| {cat} | {r['labels']} | {r['predictions']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                     f"{r['excluded_ambiguous']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | "
                     f"{', '.join(f'{x:.2f}' for x in d) or '-'} |")
    it = report["categories"].get("INTERACTION")
    if it:
        lines += ["", f"**Interaction count:** {it['count_predicted']} predicted vs {it['count_labelled']} labelled "
                      f"({it['labels']} required + {it['labels_optional']} ambiguous) - count error {it['count_error']}."]
    if report["not_scored"]:
        lines += ["", f"Not scored (no labels): {', '.join(report['not_scored'])}"]
    for cat, r in report["categories"].items():
        for fp in r["false_positives"]:
            lines.append(f"- FP {cat}: {fp['pred']} {fp['interval']}")
        for fn in r["false_negatives"]:
            lines.append(f"- FN {cat}: label {fn['label']} {fn['interval']}")
    if report["uncertain_predictions"]:
        lines += ["", "UNCERTAIN predictions (not scored):"]
        for u in report["uncertain_predictions"]:
            lines.append(f"- {u['pred']} {u['interval']} overlaps {u['overlaps_labels'] or 'no label'}")
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_json: Path, out_md: Path, video_name: str) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(report_markdown(report, video_name), encoding="utf-8")
