"""Trained episode classifier: P(hand contact) and P(handshake) from pose-only episode features.

Two small logistic-regression models (scikit-learn at training time, plain NumPy at run time) are stored
in one JSON file (``models/interaction_model.json``) so they are inspectable and need no pickle:

  contact   : likely hand contact (incl. handshakes)  vs  no contact (one-sided reach, pass, near-miss)
  handshake : likely handshake  vs  other hand contact (touch, giving/taking an item)

Training data are the candidate episodes produced by the state machine on labelled videos
(``tools/train_classifier.py``). The features use the pose keypoints only - no objects are detected.
If the model file is missing, the state machine falls back to its rule-based scores.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Kept deliberately small (the training set is small). min_d - how close the two hands get - is the strongest
# contact cue: overlapping hands come within ~0.25 forearm lengths, a hand reaching past someone's resting hand or
# an object held between two hands keeps them ~0.4-0.6 apart.
CONTACT_FEATURES = ["close_time", "close_ratio", "min_d", "mean_d", "engage_min"]
HANDSHAKE_FEATURES = ["close_time", "min_d", "reach_sym", "comove", "reversals"]
MIN_TRAIN_CLOSE_S = 0.3   # shorter fragments (e.g. a person cut out by occlusion) are not used as training examples


def episode_features(ep, hs_parts: dict | None = None, clasp_distance: float = 0.20) -> dict:
    """Features of a (possibly still open) contact episode. Missing values are filled conservatively."""
    def med(x):
        return float(np.median(x)) if len(x) else math.nan

    rc, rs = med(ep.reach_c), med(ep.reach_s)
    base_c = ep.base_c if ep.base_c is not None else (min(ep.reach_c) if ep.reach_c else math.nan)
    base_s = ep.base_s if ep.base_s is not None else (min(ep.reach_s) if ep.reach_s else math.nan)
    eng_c, eng_s = rc - base_c, rs - base_s
    hs = hs_parts or {}
    f = {
        "close_time": float(ep.close_time),
        "close_ratio": float(ep.close_ratio),
        "mean_d": float(np.mean(ep.d_close)) if ep.d_close else 1.0,
        "min_d": float(np.min(ep.d_close)) if ep.d_close else 1.0,
        "clasp_frac": float(np.mean(np.asarray(ep.d_close) < clasp_distance)) if ep.d_close else 0.0,
        "engage_c": eng_c, "engage_s": eng_s,
        "engage_min": float(np.nanmin([eng_c, eng_s])) if not (math.isnan(eng_c) and math.isnan(eng_s)) else 0.0,
        "reach_c": rc, "reach_s": rs,
        "reach_min": float(np.nanmin([rc, rs])) if not (math.isnan(rc) and math.isnan(rs)) else 0.0,
        "reach_sym": abs(rc - rs) if not (math.isnan(rc) or math.isnan(rs)) else 1.0,
        "rel_speed": float(ep.median_speed) if ep.median_speed is not None else 0.0,
        "reversals": float(hs.get("reversals", 0)),
        "comove": float(hs.get("comovement", 0.0)),
        "extension": float(hs.get("extension", 0.5)),
        "conf": float(np.mean(ep.confs)) if ep.confs else 0.0,
        "meet_shift": _meet_shift(ep.meets),
        "vertical": _vertical_share(ep.mids),
    }
    return {k: (0.0 if isinstance(v, float) and math.isnan(v) else v) for k, v in f.items()}


def _meet_shift(meets: list) -> float:
    """How far the meeting point of the two hands drifted along the line between the people while the hands
    were together (fraction of the distance between them). An item handed over travels toward the receiver;
    clasped hands in a handshake stay where they met."""
    if len(meets) < 6:
        return 0.0
    k = max(2, len(meets) // 4)
    return abs(float(np.median(meets[-k:])) - float(np.median(meets[:k])))


def _vertical_share(mids: list) -> float:
    """Share of the joined hands' (drift-removed) motion that is vertical in the image: a handshake pumps up and
    down (seen from the side), a hand-over moves the hands sideways from giver to receiver."""
    if len(mids) < 5:
        return 0.5
    m = np.asarray(mids, float)
    t = np.arange(len(m), dtype=float)
    res = [m[:, i] - np.polyval(np.polyfit(t, m[:, i], 1), t) for i in range(2)]
    sx, sy = float(np.std(res[0])), float(np.std(res[1]))
    return sy / (sx + sy) if sx + sy > 1e-9 else 0.5


class LogisticModel:
    def __init__(self, features, mean, scale, coef, intercept):
        self.features = list(features)
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.coef = np.asarray(coef, float)
        self.intercept = float(intercept)

    def prob(self, feats: dict) -> float:
        x = (np.array([feats[k] for k in self.features], float) - self.mean) / self.scale
        z = float(x @ self.coef + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))

    def contributions(self, feats: dict) -> dict:
        x = (np.array([feats[k] for k in self.features], float) - self.mean) / self.scale
        return {k: round(float(v), 2) for k, v in zip(self.features, x * self.coef)}

    def to_dict(self) -> dict:
        return {"features": self.features, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @staticmethod
    def from_dict(d: dict) -> "LogisticModel":
        return LogisticModel(d["features"], d["mean"], d["scale"], d["coef"], d["intercept"])


class TreeModel:
    """A shallow decision tree stored as plain arrays. prob() = Laplace-smoothed positive fraction of the leaf."""

    def __init__(self, features, feature_idx, threshold, left, right, pos, neg):
        self.features = list(features)
        self.feature_idx, self.threshold = list(feature_idx), list(threshold)
        self.left, self.right, self.pos, self.neg = list(left), list(right), list(pos), list(neg)

    def _leaf(self, feats: dict) -> int:
        n = 0
        while self.left[n] != -1:
            f = self.features[self.feature_idx[n]]
            n = self.left[n] if feats[f] <= self.threshold[n] else self.right[n]
        return n

    def prob(self, feats: dict) -> float:
        n = self._leaf(feats)
        return (self.pos[n] + 1.0) / (self.pos[n] + self.neg[n] + 2.0)

    def rules(self) -> list[str]:
        """Human-readable root-to-leaf rules."""
        out = []

        def walk(n, conds):
            if self.left[n] == -1:
                p = (self.pos[n] + 1.0) / (self.pos[n] + self.neg[n] + 2.0)
                out.append(f"IF {' AND '.join(conds) or 'always'} THEN p={p:.2f} ({self.pos[n]}+/{self.neg[n]}-)")
                return
            f, t = self.features[self.feature_idx[n]], self.threshold[n]
            walk(self.left[n], conds + [f"{f} <= {t:.3f}"])
            walk(self.right[n], conds + [f"{f} > {t:.3f}"])
        walk(0, [])
        return out

    def to_dict(self) -> dict:
        return {"type": "tree", "features": self.features, "feature_idx": self.feature_idx,
                "threshold": self.threshold, "left": self.left, "right": self.right, "pos": self.pos, "neg": self.neg}

    @staticmethod
    def from_dict(d: dict) -> "TreeModel":
        return TreeModel(d["features"], d["feature_idx"], d["threshold"], d["left"], d["right"], d["pos"], d["neg"])


def fit_tree(X: np.ndarray, y: np.ndarray, features: list[str], max_depth: int = 2, min_leaf: int = 2) -> TreeModel:
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0)
    clf.fit(X, y)
    t = clf.tree_
    pos, neg = [], []
    # leaf counts from the training samples (sklearn stores fractions in recent versions)
    leaves = clf.apply(X)
    for n in range(t.node_count):
        m = leaves == n
        pos.append(int(y[m].sum()))
        neg.append(int(m.sum() - y[m].sum()))
    return TreeModel(features, [int(i) for i in t.feature], [float(v) for v in t.threshold],
                     [int(v) for v in t.children_left], [int(v) for v in t.children_right], pos, neg)


def model_from_dict(d: dict):
    return TreeModel.from_dict(d) if d.get("type") == "tree" else LogisticModel.from_dict(d)


class InteractionClassifier:
    def __init__(self, contact: LogisticModel, handshake: LogisticModel, meta: dict):
        self.contact, self.handshake, self.meta = contact, handshake, meta

    @staticmethod
    def load(path: str | Path) -> "InteractionClassifier | None":
        path = Path(path)
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return InteractionClassifier(model_from_dict(d["contact"]), model_from_dict(d["handshake"]), d.get("meta", {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"contact": self.contact.to_dict(), "handshake": self.handshake.to_dict(),
                                    "meta": self.meta}, indent=2), encoding="utf-8")
        return path


def fit_logistic(X: np.ndarray, y: np.ndarray, features: list[str], C: float = 0.5) -> LogisticModel:
    """Standardise, then L2-regularised logistic regression with balanced classes (small data)."""
    from sklearn.linear_model import LogisticRegression
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-6] = 1.0
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit((X - mean) / scale, y)
    return LogisticModel(features, mean, scale, clf.coef_[0], clf.intercept_[0])
