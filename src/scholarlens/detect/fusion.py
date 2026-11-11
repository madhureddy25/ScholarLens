"""Lens fusion detector: calibrated gradient boosting over statistical, stylometric
and sentence-transition features.

Robust training: AI texts that were paraphrased, and human texts that were
language-edited, are included in training with their *true* labels, so the
model learns that paraphrasing does not make AI text human, and that editing
does not make human text AI. ``exclude_families`` holds attack families out of
training to measure generalisation to unseen attacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scholarlens.bench.schema import LABEL_AI, Record
from scholarlens.detect.featurize import Featurizer


def _make_estimator(kind: str, seed: int):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.15, random_state=seed)


class FusionDetector:
    name = "lens-fusion"

    def __init__(self, featurizer: Featurizer, *, estimator: str = "hgb", calibration: str = "isotonic",
                 seed: int = 13) -> None:
        self.featurizer = featurizer
        self.estimator_kind, self.calibration, self.seed = estimator, calibration, seed
        self.model = None
        self.feature_names: list[str] = []
        self.train_info: dict = {}

    def _xy(self, records: list[Record], on_progress=None):
        X, names = self.featurizer.featurize([(r.id, r.text) for r in records], on_progress=on_progress)
        y = np.array([1 if r.label == LABEL_AI else 0 for r in records])
        return X, y, names

    def fit(self, train: list[Record], val: list[Record] | None = None, *, exclude_families: tuple[str, ...] = (),
            sample_weight: dict[str, float] | None = None, on_progress=None) -> FusionDetector:
        train = [r for r in train if r.family not in exclude_families]
        X, y, names = self._xy(train, on_progress)
        self.feature_names = names
        w = np.array([sample_weight.get(r.id, 1.0) for r in train]) if sample_weight else None
        base = _make_estimator(self.estimator_kind, self.seed).fit(X, y, sample_weight=w)
        self.model = base
        if val and self.calibration:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.frozen import FrozenEstimator

            val = [r for r in val if r.family not in exclude_families]
            Xv, yv = self._matrix(val)
            if 0 < yv.sum() < len(yv):
                self.model = CalibratedClassifierCV(FrozenEstimator(base), method=self.calibration).fit(Xv, yv)
        self.train_info = {"n_train": len(train), "n_val": len(val or []), "excluded_families": list(exclude_families),
                           "families": sorted({r.family for r in train}), "featurizer": self.featurizer.signature,
                           "estimator": self.estimator_kind, "calibration": self.calibration}
        return self

    def _matrix(self, records: list[Record]):
        X, y, names = self._xy(records)
        return self._align(X, names), y

    def _align(self, X: np.ndarray, names: list[str]) -> np.ndarray:
        idx = {n: i for i, n in enumerate(names)}
        out = np.full((X.shape[0], len(self.feature_names)), np.nan)
        for j, n in enumerate(self.feature_names):
            if n in idx:
                out[:, j] = X[:, idx[n]]
        return out

    def predict_proba(self, records: list[Record]) -> np.ndarray:
        X, _ = self._matrix(records)
        return self.model.predict_proba(X)[:, 1]

    def score_texts(self, texts: list[str], ids: list[str] | None = None) -> np.ndarray:
        ids = ids or [f"adhoc:{hash(t)}" for t in texts]
        X, names = self.featurizer.featurize(list(zip(ids, texts, strict=True)))
        return self.model.predict_proba(self._align(X, names))[:, 1]

    def importance(self, records: list[Record], n_repeats: int = 5) -> dict[str, float]:
        from sklearn.inspection import permutation_importance

        X, y = self._matrix(records)
        r = permutation_importance(self.model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=self.seed)
        return dict(sorted(zip(self.feature_names, r.importances_mean.round(4).tolist(), strict=True),
                           key=lambda kv: -kv[1]))

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        (path / "meta.json").write_text(json.dumps({"feature_names": self.feature_names, **self.train_info}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, featurizer: Featurizer) -> FusionDetector:
        import joblib

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        det = cls(featurizer, estimator=meta.get("estimator", "hgb"), calibration=meta.get("calibration", "isotonic"))
        det.model = joblib.load(path / "model.joblib")
        det.feature_names = meta["feature_names"]
        det.train_info = meta
        return det
