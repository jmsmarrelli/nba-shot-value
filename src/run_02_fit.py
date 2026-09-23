"""Stage 2 - fit both expected-value models and score every shot out of fold.

    python src/run_02_fit.py

Writes data/processed/xpps_geo.npy and xpps_full.npy (one out-of-fold
expectation per shot, aligned row-for-row with shots.csv.gz), plus
output/model_report.csv and output/calibration.csv.

Both specifications are fit here rather than one, because they answer different
questions - see the note at the top of model.py. Expect roughly fifteen minutes
on a laptop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT, PROC, SEED  # noqa: E402
import model as M  # noqa: E402

LABELS = {"geo": "geometry only", "full": "geometry + action + clock"}


def _oof_from(est_factory, X, y, groups):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        est = est_factory()
        est.fit(X.iloc[tr], y[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return oof


def main() -> None:
    t0 = time.time()
    shots = pd.read_csv(PROC / "shots.csv.gz", low_memory=False)
    print(f"loaded {len(shots):,} shots in {time.time() - t0:.0f}s")

    y = shots["made"].to_numpy().astype(int)
    groups = shots["game_id"].to_numpy()
    rows, cal_frames = [], []

    for name, spec in M.SPECS.items():
        print(f"\n{name}: {', '.join(M.features(spec))}")
        oof, _ = M.fit_out_of_fold(shots, spec)
        np.save(PROC / f"xpps_{name}.npy", oof)
        rows.append({"model": LABELS[name], "spec": name,
                     "auc": roc_auc_score(y, oof), "log_loss": log_loss(y, oof),
                     "brier": brier_score_loss(y, oof)})
        cal = M.calibration_table(oof, y)
        cal.insert(0, "spec", name)
        cal_frames.append(cal)
        gap = (cal["actual"] - cal["predicted"]).abs().max()
        print(f"  AUC {rows[-1]['auc']:.4f}   largest calibration gap {gap:.4f}")

    print("\nreference points")
    p_base = _oof_from(lambda: DummyClassifier(strategy="prior"),
                       shots[["dist"]], y, groups)
    rows.append({"model": "league make rate", "spec": "base", "auc": 0.5,
                 "log_loss": log_loss(y, p_base),
                 "brier": brier_score_loss(y, p_base)})
    p_dist = _oof_from(
        lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                               random_state=SEED),
        shots[["dist"]], y, groups)
    rows.append({"model": "distance only", "spec": "dist",
                 "auc": roc_auc_score(y, p_dist), "log_loss": log_loss(y, p_dist),
                 "brier": brier_score_loss(y, p_dist)})

    report = pd.DataFrame(rows).round(5)
    report.to_csv(OUT / "model_report.csv", index=False)
    pd.concat(cal_frames).to_csv(OUT / "calibration.csv", index=False)
    print("\n" + report.to_string(index=False))

    for name in M.SPECS:
        oof = np.load(PROC / f"xpps_{name}.npy")
        xp = M.expected_points(oof, shots["is_three"].to_numpy())
        print(f"{name}: mean xPPS {xp.mean():.4f}  vs actual {shots['points'].mean():.4f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
