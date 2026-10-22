"""Stage 4 - player-season shot quality and shot making.

    python src/run_04_players.py

Two numbers per player-season, both in points per attempt:

    SQ  mean expected value of the shots he took
    SM  mean actual minus expected on those shots

SM is centred on the season's league mean so a 2005 shooter and a 2025 shooter
are graded against their own peers rather than against each other. SQ is left
uncentred: the drift in shot quality across eras is the finding, not a nuisance.

Writes output/player_seasons.csv, the leaderboards, and the quadrant figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIG, LAST_SEASON, MIN_FGA, OUT  # noqa: E402
import labels as L  # noqa: E402
import style as S  # noqa: E402
from run_03_league import load  # noqa: E402


def pick_labels(d: pd.DataFrame) -> set:
    """Name the edges of the cloud rather than a hand-picked list, so the chart
    still makes sense when it is rebuilt on a different season."""
    picks = set()
    for col, n in (("sm", 5), ("sq", 4), ("fga", 4)):
        picks |= set(d.nlargest(n, col)["player"])
    for col, n in (("sm", 4), ("sq", 4)):
        picks |= set(d.nsmallest(n, col)["player"])
    return picks


def player_seasons(shots: pd.DataFrame) -> pd.DataFrame:
    g = shots.groupby(["season", "player_id", "player"], observed=True)
    ps = g.agg(fga=("points", "size"), pps=("points", "mean"),
               sq=("xpps", "mean"), sm_raw=("sm", "mean"),
               three_rate=("is_three", "mean"), team=("team_abbrev", "last")).reset_index()
    league_sm = shots.groupby("season")["sm"].mean()
    ps["sm"] = ps["sm_raw"] - ps["season"].map(league_sm)
    ps["points_added"] = ps["sm"] * ps["fga"]
    ps["value_added"] = (ps["sq"] - ps["season"].map(shots.groupby("season")["xpps"].mean())
                         + ps["sm"]) * ps["fga"]
    return ps[ps["fga"] >= MIN_FGA].sort_values(["season", "points_added"],
                                                ascending=[True, False])


def fig_quadrant(ps: pd.DataFrame, season: int) -> None:
    d = ps[ps["season"] == season].copy()
    mx, my = d["sq"].mean(), d["sm"].mean()

    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    ax.axvline(mx, color=S.GRID, lw=0.9, zorder=1)
    ax.axhline(my, color=S.GRID, lw=0.9, zorder=1)
    sizes = 8 + 62 * (d["fga"] - d["fga"].min()) / (d["fga"].max() - d["fga"].min())
    ax.scatter(d["sq"], d["sm"], s=sizes, c=S.NEUTRAL, lw=0, alpha=0.75, zorder=2)

    tag = d[d["player"].isin(pick_labels(d))]
    ax.scatter(tag["sq"], tag["sm"],
               s=8 + 62 * (tag["fga"] - d["fga"].min()) / (d["fga"].max() - d["fga"].min()),
               c=S.SERIES[0], lw=1.6, edgecolors=S.SURFACE, zorder=4)

    for x, y, t, ha, va in [
        (0.02, 0.97, "Hard shots, made anyway", "left", "top"),
        (0.98, 0.97, "Good shots, made", "right", "top"),
        (0.02, 0.03, "Hard shots, missed", "left", "bottom"),
        (0.98, 0.03, "Good shots, missed", "right", "bottom"),
    ]:
        ax.text(x, y, t, transform=ax.transAxes, ha=ha, va=va,
                fontsize=8.4, color=S.INK_3, style="italic")

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.10 * (hi - lo))
    S.strip(ax)
    ax.set_xlabel("Shot quality - expected points per attempt")
    ax.set_ylabel("Shot making - points per attempt above expectation")
    ax.set_title(f"Two ways to be efficient, {season - 1}-{str(season)[2:]}", loc="left")
    fig.tight_layout()
    L.place(ax, tag["sq"], tag["sm"], tag["player"], colour=S.INK)
    S.caption(fig, f"Players with {MIN_FGA}+ field goal attempts. Marker size is attempts. "
                   "Labelled: the leaders and trailers on each axis, plus the four highest "
                   "volume shooters.")
    fig.savefig(FIG / "07_player_quadrant.png")
    plt.close(fig)


def fig_leaders(ps: pd.DataFrame, season: int) -> None:
    d = ps[ps["season"] == season]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharex=False)
    panels = [
        ("sq", "Shot quality", "Expected points per attempt", S.SERIES[1], axes[0]),
        ("sm", "Shot making", "Points per attempt above expectation", S.SERIES[0], axes[1]),
    ]
    for col, title, xlab, colour, ax in panels:
        top = d.nlargest(12, col).iloc[::-1]
        ax.barh(top["player"], top[col], color=colour, height=0.62)
        for i, (_, r) in enumerate(top.iterrows()):
            ax.text(r[col], i, f"  {r[col]:+.3f}" if col == "sm" else f"  {r[col]:.3f}",
                    va="center", fontsize=8.6, color=S.INK_2)
        ax.set_xlim(0, top[col].max() * 1.22)
        ax.grid(axis="y", visible=False)
        S.strip(ax, keep=("bottom",))
        ax.tick_params(axis="y", labelsize=8.8)
        ax.set_xlabel(xlab)
        ax.set_title(title, loc="left")
    S.caption(fig, f"{season - 1}-{str(season)[2:]} regular season, {MIN_FGA}+ attempts. Two "
                   "different skills, two nearly disjoint lists.")
    fig.tight_layout()
    fig.savefig(FIG / "08_leaderboards.png")
    plt.close(fig)


def main() -> None:
    shots = load()
    ps = player_seasons(shots)
    ps.round(5).to_csv(OUT / "player_seasons.csv", index=False)

    season = LAST_SEASON
    fig_quadrant(ps, season)
    fig_leaders(ps, season)

    d = ps[ps["season"] == season]
    print(f"{len(ps):,} player-seasons at {MIN_FGA}+ attempts, "
          f"{ps.season.min()}-{ps.season.max()}\n")
    print(f"--- {season - 1}-{str(season)[2:]} shot quality ---")
    print(d.nlargest(10, "sq")[["player", "team", "fga", "sq", "sm"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n--- {season - 1}-{str(season)[2:]} shot making ---")
    print(d.nlargest(10, "sm")[["player", "team", "fga", "sq", "sm"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n--- {season - 1}-{str(season)[2:]} points above expectation ---")
    print(d.nlargest(10, "points_added")[
        ["player", "team", "fga", "sq", "sm", "points_added"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    overlap = set(d.nlargest(20, "sq")["player"]) & set(d.nlargest(20, "sm")["player"])
    print(f"\nplayers in both top-20 lists: {len(overlap)} {sorted(overlap)}")
    print(f"corr(SQ, SM) across player-seasons: {ps['sq'].corr(ps['sm']):+.3f}")


if __name__ == "__main__":
    main()
