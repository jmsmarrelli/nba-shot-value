"""Stage 7 - try to break the headline.

    python src/run_07_robustness.py

The claim is that most of the league's twenty-two-year efficiency gain is
conversion, not location. It gets rebuilt six more ways.

The headline uses a five-zone league-average yardstick and no model, because zone
labels are the only description of a shot that means the same thing at both ends
of the sample. Everything else in the table is a check on that:

1. **The geometry model**, out of fold, across all seasons.
2. **The geometry model on 2010-11 onward**, where shot coordinates are recorded
   consistently, alongside the zone yardstick on the same window so the two are
   compared like for like.
3. **Era-anchored**: a geometry model trained on 2012-13 to 2014-15 only and
   applied out of era, to test whether fitting on the modern shot mix is doing
   the work. Restricted to the consistent-coordinate era for the same reason.
4. **Distance alone**, the crudest defensible geometry.
5. **The full model**, included precisely because it *disagrees*. It attributes
   the whole change to selection, and the reason is figure 06: the action labels
   it uses were redefined mid-sample, so it reads a change in the league's
   scorekeeping as a change in the league's shots. It is the right model within a
   season and the wrong one across eras, and this row is the demonstration.

Every specification that avoids a feature which drifted puts the location share
between a quarter and a half, and conversion in the majority. The two that
disagree are the two that use a drifting feature, and the table names them.

Writes output/robustness.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COORD_ERA_START, OUT, PROC, SEED  # noqa: E402
import model as M  # noqa: E402
from run_03_league import load  # noqa: E402

ANCHOR = (2013, 2015)


def decomposition(shots: pd.DataFrame, xpps: np.ndarray) -> dict:
    df = pd.DataFrame({"season": shots["season"].to_numpy(),
                       "points": shots["points"].to_numpy(), "xpps": xpps})
    g = df.groupby("season").agg(pps=("points", "mean"), xpps=("xpps", "mean"))
    d_total = g["pps"].iloc[-1] - g["pps"].iloc[0]
    d_location = g["xpps"].iloc[-1] - g["xpps"].iloc[0]
    return {"d_total": d_total, "d_location": d_location,
            "d_conversion": d_total - d_location,
            "location_share": d_location / d_total}


def era_anchored(shots: pd.DataFrame) -> np.ndarray:
    """Fit on three middle seasons, score every season in the window."""
    train = shots["season"].between(*ANCHOR).to_numpy()
    est = M.new_estimator()
    est.fit(M.design_matrix(shots[train], M.GEO), shots.loc[train, "made"].astype(int))
    p = est.predict_proba(M.design_matrix(shots, M.GEO))[:, 1]
    return M.expected_points(p, shots["is_three"].to_numpy())


def zone_lookup(shots: pd.DataFrame) -> np.ndarray:
    rate = shots.groupby("zone5", observed=True)["made"].mean()
    p = shots["zone5"].map(rate).to_numpy(dtype=float)
    return M.expected_points(p, shots["is_three"].to_numpy())


def distance_only(shots: pd.DataFrame) -> np.ndarray:
    X = shots[["dist"]].copy()
    X["is_three"] = shots["is_three"].astype(int)
    y = shots["made"].to_numpy().astype(int)
    oof = np.full(len(shots), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, shots["game_id"].to_numpy()):
        est = M.new_estimator(random_state=SEED)
        est.fit(X.iloc[tr], y[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return M.expected_points(oof, shots["is_three"].to_numpy())


def main() -> None:
    shots = load("geo")
    value = np.where(shots["is_three"], 3.0, 2.0)
    rows = []

    zone_rate = shots.groupby("zone5", observed=True)["points"].mean()
    xz = shots["zone5"].map(zone_rate).to_numpy(dtype=float)
    rows.append({"spec": "headline: five-zone yardstick, no model",
                 **decomposition(shots, xz)})
    rows.append({"spec": "geometry model, out of fold, all seasons",
                 **decomposition(shots, shots["xpps"].to_numpy())})

    modern = shots["season"] >= COORD_ERA_START
    rows.append({"spec": f"geometry model, {COORD_ERA_START}- only",
                 **decomposition(shots[modern], shots.loc[modern, "xpps"].to_numpy())})
    rows.append({"spec": f"five-zone yardstick, {COORD_ERA_START}- only",
                 **decomposition(shots[modern], xz[modern.to_numpy()])})

    print(f"era-anchored geometry model ({ANCHOR[0]}-{ANCHOR[1]})")
    md = shots[modern].reset_index(drop=True)
    rows.append({"spec": f"geometry trained on {ANCHOR[0]}-{ANCHOR[1]}, "
                         f"{COORD_ERA_START}- only",
                 **decomposition(md, era_anchored(md))})

    print("distance-only model")
    rows.append({"spec": f"distance and shot value only, {COORD_ERA_START}- only",
                 **decomposition(md, distance_only(md))})

    print("full model (shown to fail, see figure 06)")
    full = np.load(PROC / "xpps_full.npy") * value
    rows.append({"spec": "full model, contaminated by label drift",
                 **decomposition(shots, full)})

    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "robustness.csv", index=False)
    print()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
