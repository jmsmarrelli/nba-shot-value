"""Stage 6 - the same split, from the other side of the ball.

    python src/run_06_defense.py

Every shot in the file has a defence attached to it: the other team in the game.
Reversing the identity gives each team-season a shot quality *allowed* and a
shot making *allowed*.

The result is less tidy than the popular version of this argument, and more
useful. Inside a single season both halves of a team's defence look stable,
because a team-season is roughly 7,000 opponent attempts and almost anything is
stable at that sample. Across seasons they come apart: the map a defence forces
opponents into carries, and the percentage those opponents shot does not. That
gap is the whole practical point, because the number in the standings is the
one that does not carry.

Writes output/team_defense.csv, output/reliability_team.csv,
output/defense_carryover.csv and two figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COORD_ERA_START, FIG, LAST_SEASON, OUT  # noqa: E402
import style as S  # noqa: E402
from run_03_league import load  # noqa: E402

SIDES = [("Offence", "shot quality"), ("Offence", "shot making"),
         ("Defence", "shot quality allowed"), ("Defence", "shot making allowed")]


def attach_defence(shots: pd.DataFrame) -> pd.DataFrame:
    pairs = shots[["game_id", "team_abbrev"]].drop_duplicates()
    pairs = pairs.assign(slot=pairs.groupby("game_id").cumcount())
    wide = pairs[pairs["slot"] < 2].pivot(index="game_id", columns="slot",
                                          values="team_abbrev")
    if wide.isna().any().any():
        bad = wide.index[wide.isna().any(axis=1)]
        shots = shots[~shots["game_id"].isin(bad)].copy()
        wide = wide.dropna()
    t0 = shots["game_id"].map(wide[0]).to_numpy()
    t1 = shots["game_id"].map(wide[1]).to_numpy()
    off = shots["team_abbrev"].astype(str).to_numpy()
    shots = shots.copy()
    shots["def_team"] = np.where(off == t0, t1, t0)
    return shots


def side_table(shots: pd.DataFrame, team_col: str, side: str) -> pd.DataFrame:
    g = shots.groupby(["season", team_col], observed=True)
    out = g.agg(fga=("points", "size"), pps=("points", "mean"),
                sq=("xpps", "mean"), sm=("sm", "mean")).reset_index()
    out = out.rename(columns={team_col: "team"})
    league = shots.groupby("season")["sm"].mean()
    out["sm"] = out["sm"] - out["season"].map(league)
    out["side"] = side
    return out


def split_half(shots: pd.DataFrame, team_col: str) -> dict:
    """Odd games against even games, within each season."""
    games = np.sort(shots["game_id"].unique())
    parity = pd.Series(np.arange(len(games)) % 2, index=games)
    s = shots.assign(half=shots["game_id"].map(parity))
    agg = (s.groupby(["season", team_col, "half"], observed=True)
           .agg(sq=("xpps", "mean"), sm=("sm", "mean")).unstack("half"))
    out = {}
    for metric in ("sq", "sm"):
        block = agg[metric].dropna()
        block = block.sub(block.groupby(level="season").transform("mean"))
        r = block[0].corr(block[1])
        out[metric] = {"r_half": r, "r_full": 2 * r / (1 + r), "n": len(block)}
    return out


def carryover(tab: pd.DataFrame, side: str) -> pd.DataFrame:
    """Correlate a season with the next one, for the same franchise.

    Everything is centred within its own season first. Without that, the
    twenty-year league-wide climb in shot quality is common to every team in a
    season pair and shows up as a spurious 0.95 correlation that is really just
    the calendar.
    """
    a = tab[tab["side"] == side][["season", "team", "sq", "sm", "pps"]].copy()
    for col in ("sq", "sm", "pps"):
        a[col] = a[col] - a.groupby("season")[col].transform("mean")
    b = a.copy()
    b["season"] -= 1
    m = a.merge(b, on=["season", "team"], suffixes=("", "_next"))
    rows = []
    for metric in ("sq", "sm", "pps"):
        rows.append({"side": side, "metric": metric, "n": len(m),
                     "r_next": m[metric].corr(m[f"{metric}_next"]),
                     "r_next_pps": m[metric].corr(m["pps_next"]),
                     "r_same_pps": m[metric].corr(m["pps"])})
    return pd.DataFrame(rows)


def next_season_regression(tab: pd.DataFrame, side: str) -> pd.DataFrame:
    """Standardised coefficients on next season's points per shot.

    Correlations taken one at a time cannot separate two correlated inputs.
    Both halves in the same regression can, and the coefficients are in units of
    standard deviations so they are directly comparable.
    """
    a = tab[tab["side"] == side][["season", "team", "sq", "sm", "pps"]].copy()
    for col in ("sq", "sm", "pps"):
        a[col] = a[col] - a.groupby("season")[col].transform("mean")
    b = a.copy()
    b["season"] -= 1
    m = a.merge(b, on=["season", "team"], suffixes=("", "_next"))

    X = np.column_stack([
        (m["sq"] - m["sq"].mean()) / m["sq"].std(),
        (m["sm"] - m["sm"].mean()) / m["sm"].std(),
        np.ones(len(m)),
    ])
    y = ((m["pps_next"] - m["pps_next"].mean()) / m["pps_next"].std()).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var()
    # Standard errors from the usual OLS sandwich-free formula.
    dof = len(m) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / dof)
    return pd.DataFrame({
        "side": side,
        "term": ["shot quality", "shot making", "intercept"],
        "beta": beta, "se": se, "t": beta / se,
        "r2": r2, "n": len(m),
    })


def fig_dumbbell(rel: pd.DataFrame) -> None:
    rel = rel.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.4, 3.7))
    for i, row in rel.iterrows():
        ax.plot([row["r_next"], row["r_full"]], [i, i], color=S.GRID, lw=2.6, zorder=1)
    ax.scatter(rel["r_full"], rel.index, s=86, color=S.SERIES[1],
               edgecolors=S.SURFACE, lw=2, zorder=3)
    ax.scatter(rel["r_next"], rel.index, s=86, color=S.SERIES[0],
               edgecolors=S.SURFACE, lw=2, zorder=3)
    for i, row in rel.iterrows():
        ax.text(row["r_full"] + 0.022, i, f"{row['r_full']:.2f}", va="center",
                fontsize=8.8, color=S.INK_2)
        ax.text(row["r_next"] - 0.022, i, f"{row['r_next']:.2f}", va="center",
                ha="right", fontsize=8.8, color=S.INK_2)
    ax.set_yticks(rel.index, rel["label"])
    ax.set_ylim(-0.62, len(rel) - 0.38)
    ax.set_xlim(0, 1.06)
    ax.grid(axis="y", visible=False)
    S.strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", labelsize=9.2)
    ax.set_xlabel("Correlation")
    ax.set_title("Everything looks like a skill inside one season", loc="left", pad=30)
    ax.text(0.0, 1.035, "Orange: two halves of the same season.   "
                        "Blue: this season against the next one.",
            transform=ax.transAxes, fontsize=9, color=S.INK_2)
    S.caption(fig, f"Team-seasons, {COORD_ERA_START - 1}-{str(COORD_ERA_START)[2:]} to 2024-25. Within-season figures are odd games "
                   "against even games, corrected to full-season length. Both are centred "
                   "within season.")
    fig.tight_layout()
    fig.savefig(FIG / "11_team_reliability.png")
    plt.close(fig)


def fig_carry(carry: pd.DataFrame) -> None:
    d = carry[carry["side"] == "Defence"].set_index("metric")
    order = ["pps", "sq", "sm"]
    names = {"pps": "Points allowed per shot", "sq": "Shot quality allowed",
             "sm": "Shot making allowed"}
    # The same-season correlation of points allowed with itself is 1 by
    # construction, so it is left off rather than drawn as a full-width bar.
    same = [np.nan if m == "pps" else d.loc[m, "r_same_pps"] for m in order]
    nxt = [d.loc[m, "r_next_pps"] for m in order]

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    y = np.arange(len(order))[::-1]
    h = 0.34
    ax.barh(y + h / 2 + 0.02, same, height=h, color=S.SERIES[1])
    ax.barh(y - h / 2 - 0.02, nxt, height=h, color=S.SERIES[0])
    for yy, v in zip(y + h / 2 + 0.02, same):
        if not np.isnan(v):
            ax.text(v + 0.015, yy, f"{v:.2f}", va="center", fontsize=8.8, color=S.INK_2)
    for yy, v in zip(y - h / 2 - 0.02, nxt):
        ax.text(v + 0.015, yy, f"{v:.2f}", va="center", fontsize=8.8, color=S.INK_2)
    ax.set_yticks(y, [names[m] for m in order])
    ax.set_xlim(0, 1.06)
    ax.grid(axis="y", visible=False)
    S.strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", labelsize=9.2)
    ax.set_xlabel("Correlation with points allowed per shot")
    ax.set_title("What this season's defence says about the next one", loc="left", pad=30)
    ax.text(0.0, 1.035, "Orange: the same season.   Blue: the following season.",
            transform=ax.transAxes, fontsize=9, color=S.INK_2)
    S.caption(fig, f"Consecutive team-seasons, {COORD_ERA_START - 1}-{str(COORD_ERA_START)[2:]} to 2024-25, centred within season. Shot "
                   "making allowed is the larger driver of this year's number and still the "
                   "better single predictor of next year's, which is not what the usual "
                   "'defences control location, not conversion' story predicts.")
    fig.tight_layout()
    fig.savefig(FIG / "12_defence_carryover.png")
    plt.close(fig)


def main() -> None:
    shots = attach_defence(load().query("season >= @COORD_ERA_START"))

    off = side_table(shots, "team_abbrev", "Offence")
    dfn = side_table(shots, "def_team", "Defence")
    both = pd.concat([off, dfn], ignore_index=True)
    both.round(5).to_csv(OUT / "team_defense.csv", index=False)

    ro, rd = split_half(shots, "team_abbrev"), split_half(shots, "def_team")
    carry = pd.concat([carryover(both, "Offence"), carryover(both, "Defence")],
                      ignore_index=True)
    carry.round(4).to_csv(OUT / "defense_carryover.csv", index=False)

    within = {("Offence", "sq"): ro["sq"], ("Offence", "sm"): ro["sm"],
              ("Defence", "sq"): rd["sq"], ("Defence", "sm"): rd["sm"]}
    rows = []
    for (side, label), metric in zip(SIDES, ["sq", "sm", "sq", "sm"]):
        w = within[(side, metric)]
        nxt = carry[(carry["side"] == side) & (carry["metric"] == metric)]["r_next"].iloc[0]
        rows.append({"label": f"{side}: {label}", "side": side, "metric": metric,
                     "r_half": w["r_half"], "r_full": w["r_full"],
                     "r_next": nxt, "n": w["n"]})
    rel = pd.DataFrame(rows)
    rel.round(4).to_csv(OUT / "reliability_team.csv", index=False)

    fig_dumbbell(rel)
    fig_carry(carry)

    print(rel[["label", "r_half", "r_full", "r_next", "n"]].round(3).to_string(index=False))
    print()
    print(carry.round(3).to_string(index=False))

    d = both[(both["season"] == LAST_SEASON) & (both["side"] == "Defence")]
    print(f"\n--- {LAST_SEASON - 1}-{str(LAST_SEASON)[2:]} best shot quality allowed ---")
    print(d.nsmallest(8, "sq")[["team", "sq", "sm", "pps"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    dd = both[both["side"] == "Defence"]
    v_sq = dd.groupby("season")["sq"].std().mean()
    v_sm = dd.groupby("season")["sm"].std().mean()
    print(f"\naverage within-season spread across teams (SD):  "
          f"shot quality allowed {v_sq:.4f}, shot making allowed {v_sm:.4f}")

    print("\npredicting next season's points allowed per shot from this season's two halves")
    print(next_season_regression(both, "Defence").round(3).to_string(index=False))
    print(next_season_regression(both, "Offence").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
