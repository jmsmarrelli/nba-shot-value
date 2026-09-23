"""Stage 3 - split twenty-two seasons of league offense into shots and makes.

    python src/run_03_league.py

The identity the whole project rests on, for any set of attempts:

    PPS  =  xPPS  +  (PPS - xPPS)
            ^^^^     ^^^^^^^^^^^^
          the shots   the shooting

The era comparison uses a five-zone league-average expectation and no model at
all. That is not laziness. The source data changed twice underneath this
analysis - the action-type vocabulary was rewritten between 2007 and 2009, and
shot coordinates were recorded differently before 2010-11 - and a five-bucket
zone label is the only description of a shot whose meaning is the same in 2004
and 2025. Figure 06 is the evidence for both. The gradient-boosted model still
does the fine-grained work: the value surface, and every within-season player and
team comparison in later stages.

Writes output/league_season.csv, output/zone_share.csv, output/zone_value.csv
and six figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COORD_ERA_START, FIG, OUT, PROC  # noqa: E402
import court  # noqa: E402
import style as S  # noqa: E402

ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
         "Corner 3", "Above the Break 3"]
ZONE_LABEL = {"Restricted Area": "At the rim", "In The Paint (Non-RA)": "Paint, off the rim",
              "Mid-Range": "Mid-range", "Corner 3": "Corner three",
              "Above the Break 3": "Above-the-break three"}
# Zone labels survive the 2010-11 coordinate change, so the zone comparison keeps
# the full span. The court maps do not, so they start where the coordinates
# become consistent.
EARLY, LATE = (2004, 2006), (2023, 2025)
EARLY_COORD = (COORD_ERA_START, COORD_ERA_START + 2)
DRIFT_FAMILIES = ["Jump shot", "Layup", "Driving layup", "Pull-up", "Floater", "Step-back"]


def load(spec: str = "full") -> pd.DataFrame:
    """Shots plus the chosen expectation. spec is 'full' or 'geo'."""
    shots = pd.read_csv(PROC / "shots.csv.gz", low_memory=False)
    value = np.where(shots["is_three"], 3.0, 2.0)
    shots["xpps"] = np.load(PROC / f"xpps_{spec}.npy") * value
    shots["sm"] = shots["points"] - shots["xpps"]
    zone = shots["zone"].replace({"Left Corner 3": "Corner 3", "Right Corner 3": "Corner 3"})
    shots["zone5"] = pd.Categorical(zone, categories=ZONES, ordered=True)
    return shots


def _spread(values: dict, min_gap: float) -> dict:
    """Push end-of-line labels apart just enough that they stop colliding."""
    placed, prev = {}, -np.inf
    for k in sorted(values, key=values.get):
        y = max(values[k], prev + min_gap)
        placed[k], prev = y, y
    shift = np.mean([placed[k] - values[k] for k in placed]) / 2
    return {k: v - shift for k, v in placed.items()}


def fig_decomposition(season: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5),
                             gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    ax.plot(season.index, season["pps"], color=S.SERIES[0], zorder=3)
    ax.plot(season.index, season["xpps"], color=S.SERIES[1], zorder=3)
    ax.fill_between(season.index, season["xpps"], season["pps"],
                    color=S.SERIES[0], alpha=0.10, lw=0, zorder=1)
    S.strip(ax)
    ax.set_ylabel("Points per field goal attempt")
    ax.set_xticks(range(2004, 2026, 3))
    ax.text(0.985, 0.24, "What teams actually scored", transform=ax.transAxes,
            ha="right", color=S.SERIES[0], fontsize=9.5, fontweight="semibold")
    ax.text(0.985, 0.16, "What those shots were worth on zone alone,\n"
                         "at a fixed conversion rate",
            transform=ax.transAxes, ha="right", va="top", color=S.SERIES[1],
            fontsize=9.5, fontweight="semibold", linespacing=1.45)
    ax.set_title("League scoring efficiency and the shots behind it", loc="left")

    first, lastrow = season.iloc[0], season.iloc[-1]
    d_total = lastrow["pps"] - first["pps"]
    d_shots = lastrow["xpps"] - first["xpps"]

    ax2 = axes[1]
    labels = ["Where they shot from", "How well they shot", "Net change"]
    vals = [d_shots, d_total - d_shots, d_total]
    colors = [S.SERIES[1], S.SERIES[0], S.INK]
    bars = ax2.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.5)
    span = max(vals) - min(min(vals), 0)
    for bar, v in zip(bars, vals[::-1]):
        off = 0.035 * span
        ax2.text(v + (off if v >= 0 else -off), bar.get_y() + bar.get_height() / 2,
                 f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                 fontsize=9.5, color=S.INK_2, fontweight="semibold")
    ax2.axvline(0, color=S.INK_3, lw=0.9, zorder=4)
    ax2.set_xlim(min(min(vals), 0) - 0.28 * span, max(vals) + 0.26 * span)
    ax2.grid(axis="y", visible=False)
    S.strip(ax2, keep=("bottom",))
    ax2.tick_params(axis="y", labelsize=9.2)
    ax2.set_xlabel("Change in points per shot, 2003-04 to 2024-25")
    ax2.set_title("Where the gain came from", loc="left")

    S.caption(fig, "4.43m field goal attempts. Expected value is the league's own conversion "
                   "rate in that zone, pooled across all twenty-two seasons, so the orange line "
                   "moves only when the mix of zones moves. Heaves and backcourt attempts "
                   "excluded.")
    fig.tight_layout()
    fig.savefig(FIG / "01_league_decomposition.png")
    plt.close(fig)


def fig_zone_shares(shares: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ends = {z: 100 * shares[z].iloc[-1] for z in ZONES}
    nudge = _spread(ends, min_gap=1.5)
    for i, z in enumerate(ZONES):
        ax.plot(shares.index, 100 * shares[z], color=S.SERIES[i], zorder=3)
        ax.annotate(ZONE_LABEL[z], (shares.index[-1], nudge[z]), xytext=(7, 0),
                    textcoords="offset points", va="center", color=S.SERIES[i],
                    fontsize=9, fontweight="semibold")
    S.strip(ax)
    ax.set_xlim(2003.6, 2031.5)
    ax.set_xticks(range(2004, 2026, 3))
    ax.set_ylabel("Share of field goal attempts (%)")
    ax.set_title("The shot chart emptied from the middle out", loc="left")
    S.caption(fig, "Corner threes combine the left and right corners. Zones as labeled in "
                   "the source data.")
    fig.tight_layout()
    fig.savefig(FIG / "02_zone_shares.png")
    plt.close(fig)


def fig_zone_value(zval: pd.DataFrame) -> None:
    """Early era against late era, one row per zone. This is the finding."""
    d = zval.loc[ZONES[::-1]]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, (_, row) in enumerate(d.iterrows()):
        ax.plot([row["early"], row["late"]], [i, i], color=S.GRID, lw=2.6, zorder=1)
    ax.scatter(d["early"], y, s=84, color=S.NEUTRAL, edgecolors=S.SURFACE, lw=2, zorder=3)
    ax.scatter(d["late"], y, s=84, color=S.SERIES[0], edgecolors=S.SURFACE, lw=2, zorder=3)
    for i, (_, row) in enumerate(d.iterrows()):
        far = max(row["early"], row["late"])
        ax.text(far + 0.017, i, f"{row['late'] - row['early']:+.3f}", va="center",
                fontsize=9, color=S.INK_2, fontweight="semibold")
    ax.set_yticks(y, [ZONE_LABEL[z] for z in ZONES[::-1]])
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.grid(axis="y", visible=False)
    S.strip(ax, keep=("bottom",))
    ax.tick_params(axis="y", labelsize=9.2)
    ax.set_xlabel("Points per attempt from the zone")
    ax.set_title("Nobody got better at shooting threes", loc="left", pad=30)
    ax.text(0.0, 1.035, f"Grey: {EARLY[0] - 1}-{str(EARLY[1])[2:]}.   "
                        f"Blue: {LATE[0] - 1}-{str(LATE[1])[2:]}.   Labels are the change.",
            transform=ax.transAxes, fontsize=9, color=S.INK_2)
    S.caption(fig, "Points per attempt is the make rate times two or three, so it compares "
                   "directly across the arc.")
    fig.tight_layout()
    fig.savefig(FIG / "03_zone_value.png")
    plt.close(fig)


def _grid(sub: pd.DataFrame, values: str | None, cell: float = 1.0, ymax: float = 34.0):
    xb = np.arange(-25, 25 + cell, cell)
    yb = np.arange(0, ymax + cell, cell)
    xi = np.clip(np.digitize(sub["x"], xb) - 1, 0, len(xb) - 2)
    yi = np.clip(np.digitize(sub["y"], yb) - 1, 0, len(yb) - 2)
    n = np.zeros((len(yb) - 1, len(xb) - 1))
    np.add.at(n, (yi, xi), 1)
    if values is None:
        return n, xb, yb
    tot = np.zeros_like(n)
    np.add.at(tot, (yi, xi), sub[values].to_numpy())
    with np.errstate(invalid="ignore"):
        return tot / np.where(n == 0, np.nan, n), n, xb, yb


def fig_value_surface(shots: pd.DataFrame) -> None:
    sub = shots[(shots["y"] <= 34) & (shots["abs_x"] <= 25)
                & (shots["season"] >= COORD_ERA_START)]
    val, n, xb, yb = _grid(sub, "xpps")
    val = np.where(n >= 400, val, np.nan)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", S.BLUES)
    fig, ax = plt.subplots(figsize=(8.9, 5.9))
    mesh = ax.pcolormesh(xb, yb, val, cmap=cmap, vmin=0.75, vmax=1.35,
                         shading="flat", zorder=1)
    court.draw(ax, color=S.INK_3, lw=1.0, ymax=34)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("Expected points per attempt", color=S.INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=S.INK_3, length=0)
    ax.set_title("The expected value of a shot, by where it is taken", loc="left")
    fig.tight_layout()
    S.caption_under(fig, ax, f"One-foot cells with at least 400 attempts, "
                             f"{COORD_ERA_START - 1}-{str(COORD_ERA_START)[2:]} to 2024-25. "
                             "The pale band from fifteen feet to the arc is the long two.")
    fig.savefig(FIG / "04_value_surface.png")
    plt.close(fig)


def fig_migration(shots: pd.DataFrame) -> None:
    early = shots[shots["season"].between(*EARLY_COORD)]
    late = shots[shots["season"].between(*LATE)]
    ne, xb, yb = _grid(early[early["y"] <= 34], None)
    nl, _, _ = _grid(late[late["y"] <= 34], None)
    diff = 1000 * (nl / nl.sum() - ne / ne.sum())
    diff = np.where((nl + ne) >= 60, diff, np.nan)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("div", S.DIVERGING[::-1])
    lim = np.nanpercentile(np.abs(diff), 99)
    fig, ax = plt.subplots(figsize=(8.9, 5.9))
    mesh = ax.pcolormesh(xb, yb, diff, cmap=cmap, vmin=-lim, vmax=lim,
                         shading="flat", zorder=1)
    court.draw(ax, color=S.INK_3, lw=1.0, ymax=34)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("Change in attempts per 1,000 shots", color=S.INK_2, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=S.INK_3, length=0)
    ax.set_title("Twenty years of shot migration", loc="left")
    fig.tight_layout()
    S.caption_under(fig, ax, f"{LATE[0] - 1}-{str(LATE[1])[2:]} minus "
                             f"{EARLY_COORD[0] - 1}-{str(EARLY_COORD[1])[2:]}, per 1,000 "
                             "attempts. Blue is where teams moved to, red is what they left "
                             "behind. This map starts in 2010-11 because the coordinates "
                             "before it are not comparable - see figure 06.")
    fig.savefig(FIG / "05_shot_migration.png")
    plt.close(fig)


def fig_data_quality(shots: pd.DataFrame) -> None:
    fam = (shots.groupby(["season", "family"], observed=True).size()
           .unstack("family").fillna(0))
    fam = fam.div(fam.sum(axis=1), axis=0) * 100
    at_rim = shots.groupby("season")["dist"].apply(lambda d: 100 * (d < 0.6).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ends = {f: fam[f].iloc[-1] for f in DRIFT_FAMILIES}
    nudge = _spread(ends, min_gap=2.6)
    for i, f in enumerate(DRIFT_FAMILIES):
        series = fam[f].replace(0, np.nan)
        ax.plot(series.index, series, color=S.SERIES[i], zorder=3)
        ax.annotate(f, (fam.index[-1], nudge[f]), xytext=(7, 0),
                    textcoords="offset points", va="center", color=S.SERIES[i],
                    fontsize=9, fontweight="semibold")
    S.strip(ax)
    ax.set_xlim(2003.6, 2030.4)
    ax.set_xticks(range(2004, 2026, 3))
    ax.set_ylabel("Share of field goal attempts (%)")
    ax.set_title("The action labels were rewritten", loc="left")

    ax = axes[1]
    ax.plot(at_rim.index, at_rim, color=S.SERIES[0], zorder=3)
    ax.scatter(at_rim.index, at_rim, s=20, color=S.SERIES[0],
               edgecolors=S.SURFACE, lw=1.4, zorder=4)
    ax.axvline(COORD_ERA_START - 0.5, color=S.INK_3, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.annotate(f"{COORD_ERA_START - 1}-{str(COORD_ERA_START)[2:]}",
                (COORD_ERA_START - 0.5, at_rim.max() * 0.92), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=S.INK_2)
    S.strip(ax)
    ax.set_ylim(0, None)
    ax.set_xticks(range(2004, 2026, 4))
    ax.set_ylabel("Share of attempts logged at the rim (%)")
    ax.set_title("So were the coordinates", loc="left")

    S.caption(fig, "Left: three of these labels do not exist in the 2003-04 file. Pull-ups, "
                   "floaters and step-backs appear when the league's scorers start recording "
                   "them, not when players start taking them, and the plain layup they were "
                   "carved out of collapses to match. Right: before 2010-11 a quarter of all "
                   "attempts are logged at exactly zero feet, because most shots at the rim were "
                   "given the rim's own coordinate rather than their own. Both are changes in "
                   "the scorer's table, and a model handed either one will report them as "
                   "changes in basketball.")
    fig.tight_layout()
    fig.savefig(FIG / "06_data_quality.png")
    plt.close(fig)


def main() -> None:
    geo = load("geo")

    # The era yardstick: the league's own conversion rate in each zone, pooled
    # over every season. No coordinates, no model, nothing that drifted.
    zone_rate = geo.groupby("zone5", observed=True)["points"].mean()
    geo["xpps_zone"] = geo["zone5"].map(zone_rate).astype(float)

    season = geo.groupby("season").agg(
        fga=("points", "size"), pps=("points", "mean"),
        xpps=("xpps_zone", "mean"), xpps_model=("xpps", "mean"),
        sm=("sm", "mean"), three_rate=("is_three", "mean"),
    )
    season["efg"] = season["pps"] / 2
    season.round(5).to_csv(OUT / "league_season.csv")

    shares = (geo.groupby(["season", "zone5"], observed=True).size()
              .unstack("zone5").pipe(lambda d: d.div(d.sum(axis=1), axis=0)))
    shares.round(5).to_csv(OUT / "zone_share.csv")

    early = geo[geo["season"].between(*EARLY)].groupby("zone5", observed=True)["points"].mean()
    late = geo[geo["season"].between(*LATE)].groupby("zone5", observed=True)["points"].mean()
    zval = pd.DataFrame({"early": early, "late": late})
    zval["change"] = zval["late"] - zval["early"]
    zval.round(4).to_csv(OUT / "zone_value.csv")

    fig_decomposition(season)
    fig_zone_shares(shares)
    fig_zone_value(zval)
    fig_value_surface(geo)
    fig_migration(geo)
    fig_data_quality(geo)

    first, last = season.iloc[0], season.iloc[-1]
    d_total = last["pps"] - first["pps"]
    d_shots = last["xpps"] - first["xpps"]
    print(season[["fga", "pps", "efg", "xpps", "xpps_model", "three_rate"]].round(4).to_string())
    print(f"\nPPS {first['pps']:.4f} -> {last['pps']:.4f}   change {d_total:+.4f}")
    print(f"  where they shot from {d_shots:+.4f}  "
          f"({100 * d_shots / d_total:.0f}% of the change)")
    print(f"  how well they shot   {d_total - d_shots:+.4f}  "
          f"({100 * (d_total - d_shots) / d_total:.0f}%)")
    print(f"\neFG% {first['efg']:.4f} -> {last['efg']:.4f}")
    print(f"three-point rate {first['three_rate']:.3f} -> {last['three_rate']:.3f}")
    print("\npoints per attempt by zone")
    print(zval.round(4).to_string())

    modern = season.loc[COORD_ERA_START:]
    m_total = modern["pps"].iloc[-1] - modern["pps"].iloc[0]
    m_zone = modern["xpps"].iloc[-1] - modern["xpps"].iloc[0]
    m_model = modern["xpps_model"].iloc[-1] - modern["xpps_model"].iloc[0]
    print(f"\nsame split on {COORD_ERA_START - 1}-{str(COORD_ERA_START)[2:]} onward, "
          f"where the coordinates are consistent")
    print(f"  total {m_total:+.4f}   zone yardstick {m_zone:+.4f} "
          f"({100 * m_zone / m_total:.0f}%)   geometry model {m_model:+.4f} "
          f"({100 * m_model / m_total:.0f}%)")

    two = geo[~geo["is_three"]].groupby("season")["made"].mean()
    three = geo[geo["is_three"]].groupby("season")["made"].mean()
    print(f"\ntwo-point FG%   {two.iloc[0]:.4f} -> {two.iloc[-1]:.4f}")
    print(f"three-point FG% {three.iloc[0]:.4f} -> {three.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
