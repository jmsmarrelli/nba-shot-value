"""Stage 5 - how much of each number is skill, and how long it takes to know.

    python src/run_05_reliability.py

The evaluation question a front office actually faces is not "who shot well"
but "which of these numbers will still be true next year". Two tests:

* **Split-half at a fixed sample.** Take m attempts from a player's season at
  random, then a disjoint m more, compute the metric on each half and correlate
  across players. Sweeping m traces the reliability curve and locates the
  attempt count where the metric is half signal - the point where it starts
  being worth listening to.

* **Season to season.** The harsher test, because it has to survive a new team,
  a new role, a year of aging and a new set of defences.

Writes output/reliability_curve.csv, output/reliability_yoy.csv, two figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COORD_ERA_START, FIG, MIN_FGA, OUT, SEED  # noqa: E402
import style as S  # noqa: E402
from run_03_league import load  # noqa: E402

M_GRID = [25, 50, 75, 100, 150, 200, 300, 400, 500, 600]
N_REPEATS = 12
METRICS = {"sq": "Shot quality", "sm": "Shot making"}


def spearman_brown(r: float, k: float = 2.0) -> float:
    """Reliability of a test k times as long as the one measured."""
    return k * r / (1 + (k - 1) * r)


def _halves(shots: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.permutation(len(shots))
    s = shots.iloc[idx][["season", "player_id", "xpps", "sm"]].copy()
    s["k"] = s.groupby(["season", "player_id"], observed=True).cumcount()
    s["n"] = s.groupby(["season", "player_id"], observed=True)["k"].transform("size")
    return s


def curve(shots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(SEED + rep)
        s = _halves(shots, rng)
        for m in M_GRID:
            sub = s[s["n"] >= 2 * m]
            if sub.empty:
                continue
            half = np.where(sub["k"] < m, "a", np.where(sub["k"] < 2 * m, "b", None))
            sub = sub.assign(half=half).dropna(subset=["half"])
            agg = (sub.groupby(["season", "player_id", "half"], observed=True)
                   .agg(sq=("xpps", "mean"), sm=("sm", "mean")).unstack("half"))
            # Centre within season so era effects don't inflate the correlation.
            for metric in METRICS:
                block = agg[metric].copy()
                block = block.sub(block.groupby(level="season").transform("mean"))
                r = block["a"].corr(block["b"])
                rows.append({"rep": rep, "m": m, "metric": metric,
                             "n_players": len(block), "r_half": r,
                             "r_full": spearman_brown(r)})
    df = pd.DataFrame(rows)
    return (df.groupby(["metric", "m"])
            .agg(n_players=("n_players", "mean"),
                 r_half=("r_half", "mean"), r_half_sd=("r_half", "std"),
                 r_full=("r_full", "mean"))
            .reset_index())


def half_signal_point(sub: pd.DataFrame) -> float:
    """Attempts at which the split-half correlation reaches 0.50."""
    x, y = sub["m"].to_numpy(float), sub["r_half"].to_numpy(float)
    if y.max() < 0.5:
        return float("nan")
    i = int(np.argmax(y >= 0.5))
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))


def yoy(ps: pd.DataFrame) -> pd.DataFrame:
    a = ps[["season", "player_id", "player", "fga", "sq", "sm", "pps"]].copy()
    b = a.copy()
    b["season"] -= 1
    m = a.merge(b, on=["season", "player_id"], suffixes=("", "_next"))
    rows = []
    for metric, label in [("sq", "Shot quality"), ("sm", "Shot making"),
                          ("pps", "Points per shot")]:
        rows.append({"metric": metric, "label": label, "n": len(m),
                     "r": m[metric].corr(m[f"{metric}_next"])})
    return pd.DataFrame(rows), m


def fig_curve(cur: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axhline(0.5, color=S.GRID, lw=0.9, zorder=1)
    ax.text(M_GRID[0], 0.512, "half signal", fontsize=8, color=S.INK_3, style="italic")
    for i, (metric, label) in enumerate(METRICS.items()):
        sub = cur[cur["metric"] == metric]
        colour = S.SERIES[1] if metric == "sq" else S.SERIES[0]
        ax.fill_between(sub["m"], sub["r_half"] - 2 * sub["r_half_sd"],
                        sub["r_half"] + 2 * sub["r_half_sd"],
                        color=colour, alpha=0.13, lw=0, zorder=2)
        ax.plot(sub["m"], sub["r_half"], color=colour, zorder=3)
        ax.scatter(sub["m"], sub["r_half"], s=22, color=colour,
                   edgecolors=S.SURFACE, lw=1.6, zorder=4)
        ax.annotate(label, (sub["m"].iloc[-1], sub["r_half"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    color=colour, fontsize=9.5, fontweight="semibold")
        m50 = half_signal_point(sub)
        if not np.isnan(m50):
            ax.plot([m50, m50], [0, 0.5], color=colour, lw=1.0, ls=(0, (2, 3)), zorder=2)
            ax.annotate(f"{m50:.0f} attempts", (m50, 0.02), xytext=(4, 0),
                        textcoords="offset points", fontsize=8.4, color=colour)
    S.strip(ax)
    ax.set_xlim(0, 760)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Attempts in each half")
    ax.set_ylabel("Split-half correlation across players")
    ax.set_title("How many shots before the number means anything", loc="left")
    S.caption(fig, "Two disjoint random samples of m attempts from the same player-season, "
                   f"averaged over {N_REPEATS} draws, centred within season, {COORD_ERA_START - 1}-{str(COORD_ERA_START)[2:]} to 2024-25. "
                   "Bands are ±2 standard deviations across draws.")
    fig.tight_layout()
    fig.savefig(FIG / "09_reliability_curve.png")
    plt.close(fig)


def fig_yoy(m: pd.DataFrame, tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
    for ax, (metric, label, colour) in zip(
            axes, [("sq", "Shot quality", S.SERIES[1]), ("sm", "Shot making", S.SERIES[0])]):
        ax.scatter(m[metric], m[f"{metric}_next"], s=9, c=colour, alpha=0.30, lw=0, zorder=2)
        lo = min(m[metric].min(), m[f"{metric}_next"].min())
        hi = max(m[metric].max(), m[f"{metric}_next"].max())
        ax.plot([lo, hi], [lo, hi], color=S.INK_3, lw=1.0, zorder=3)
        b, a = np.polyfit(m[metric], m[f"{metric}_next"], 1)
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, a + b * xs, color=S.INK, lw=1.6, zorder=4)
        r = float(tab.loc[tab["metric"] == metric, "r"].iloc[0])
        ax.set_title(f"{label}   r = {r:.2f}", loc="left")
        ax.set_xlabel("This season")
        ax.set_ylabel("Next season")
        S.strip(ax)
    axes[0].text(0.03, 0.94, "grey line: no change\nblack line: best fit",
                 transform=axes[0].transAxes, fontsize=8, color=S.INK_3, va="top")
    S.caption(fig, f"Consecutive player-seasons with {MIN_FGA}+ attempts in both, "
                   f"n = {len(m):,}. Regression to the mean is the gap between the two lines.")
    fig.tight_layout()
    fig.savefig(FIG / "10_year_over_year.png")
    plt.close(fig)


def main() -> None:
    # Shot quality here is measured from the model, which reads coordinates, so
    # this stage starts where the coordinates become comparable. See config.py.
    shots = load()
    shots = shots[shots["season"] >= COORD_ERA_START]
    cur = curve(shots)
    cur.round(4).to_csv(OUT / "reliability_curve.csv", index=False)
    fig_curve(cur)

    ps = pd.read_csv(OUT / "player_seasons.csv")
    ps = ps[ps["season"] >= COORD_ERA_START]
    tab, merged = yoy(ps)
    tab.round(4).to_csv(OUT / "reliability_yoy.csv", index=False)
    fig_yoy(merged, tab)

    print(cur.round(3).to_string(index=False))
    print()
    for metric, label in METRICS.items():
        sub = cur[cur["metric"] == metric]
        print(f"{label}: half-signal at {half_signal_point(sub):.0f} attempts; "
              f"split-half r at 200 attempts = "
              f"{sub.loc[sub['m'] == 200, 'r_half'].iloc[0]:.3f}")
    print("\nseason to season:")
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
