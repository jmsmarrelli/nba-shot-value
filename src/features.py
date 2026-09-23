"""Turn raw shot rows into the feature table the expected-value model sees.

Two decisions here are worth flagging because they move results:

1. Action families. The raw files carry 48 distinct ACTION_TYPE strings, most of
   them rare and several of them near-synonyms ("Floating Jump shot" vs
   "Floating Jump Shot", which differ only in capitalisation across seasons).
   They collapse to eleven families by keyword. This is the only place shot
   *context* enters the model, and it is doing real work: a driving layup and a
   tip-in happen from the same square foot of floor and convert very differently.

2. Heaves. A 45-footer with one second left is not a shot selection decision. We
   drop them rather than let them drag the model's tail.

3. Coordinate scale. Three season files - 2019-20, 2020-21 and 2021-22 - are
   published on a different coordinate scale from the other nineteen, with x
   compressed by a factor of ten and y compressed and shifted. See
   `normalise_coordinates` below. This is checked and repaired on every season
   rather than hard-coded to those three, because a scale change that is silently
   wrong is the kind of thing that survives all the way into a chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import HEAVE_DIST, HEAVE_SECS, RIM_Y

# Order matters: the first pattern that matches wins.
_FAMILY_RULES = [
    ("Dunk", ("dunk",)),
    ("Tip / putback", ("tip", "putback", "put back")),
    ("Alley-oop", ("alley oop", "alley-oop")),
    ("Hook", ("hook",)),
    ("Floater", ("floating", "float")),
    ("Driving layup", ("driving layup", "driving finger roll", "driving reverse",
                       "running finger roll", "finger roll")),
    ("Cut / roll layup", ("cutting", "running layup", "running reverse", "reverse layup")),
    ("Layup", ("layup",)),
    ("Step-back", ("step back", "step-back", "stepback")),
    ("Pull-up", ("pullup", "pull-up", "pull up")),
    ("Turnaround / fade", ("fadeaway", "fade away", "turnaround")),
    ("Bank", ("bank",)),
]


def normalise_coordinates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, bool]:
    """Put every season's coordinates on the same feet-from-the-rim scale.

    The court is fifty feet wide, so a season whose x never exceeds a few feet is
    not a season in which nobody shot from the wing - it is a season stored in
    different units. Three of the twenty-two files are: x is divided by ten and y
    is divided by ten and then shifted up by the rim's offset, which is exactly
    the transform you get by applying the league's raw-units conversion twice.

    Inverting it is arithmetic, and it is checkable. After the repair, the median
    above-the-break three sits at the same place on the floor in every season, and
    the recomputed distance agrees with the file's own SHOT_DISTANCE column to
    within half a foot in all twenty-two.
    """
    x, y = df["LOC_X"].astype(float), df["LOC_Y"].astype(float)
    compressed = x.abs().max() < 10.0
    if compressed:
        x = x * 10.0
        y = (y - RIM_Y) * 10.0
    return x, y, compressed


def action_family(action: pd.Series) -> pd.Series:
    """Collapse 48 raw action strings into eleven families plus a catch-all."""
    low = action.str.lower()
    out = pd.Series("Jump shot", index=action.index, dtype=object)
    assigned = pd.Series(False, index=action.index)
    for name, keys in _FAMILY_RULES:
        hit = pd.Series(False, index=action.index)
        for k in keys:
            hit |= low.str.contains(k, regex=False, na=False)
        out = out.mask(hit & ~assigned, name)
        assigned |= hit
    return out.astype("category")


def team_abbreviations(df: pd.DataFrame) -> pd.Series:
    """Recover each franchise's three-letter code without hard-coding a table.

    Every row names the two abbreviations in the game. A team's own code is the
    one that shows up in every game it played, so intersecting those pairs over a
    team's whole season leaves exactly one survivor. This survives relocations
    and rebrands (Bobcats to Hornets, Sonics to Thunder) with no maintenance.
    """
    pairs = df[["TEAM_NAME", "HOME_TEAM", "AWAY_TEAM"]].drop_duplicates()
    out = {}
    for team, grp in pairs.groupby("TEAM_NAME", observed=True):
        common = set(grp.iloc[0][["HOME_TEAM", "AWAY_TEAM"]])
        for _, row in grp.iloc[1:].iterrows():
            common &= {row["HOME_TEAM"], row["AWAY_TEAM"]}
            if len(common) == 1:
                break
        if len(common) != 1:
            raise ValueError(f"could not resolve an abbreviation for {team}: {common}")
        out[team] = common.pop()
    return pd.Series(out, name="team_abbrev")


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Raw season file -> model-ready rows."""
    x_s, y_s, rescaled = normalise_coordinates(df)
    x, y = x_s.to_numpy(), y_s.to_numpy()
    dy = y - RIM_Y

    out = pd.DataFrame(
        {
            "season": df["SEASON_1"].astype("int16"),
            "game_id": df["GAME_ID"].astype("int64"),
            "game_date": df["GAME_DATE"],
            "player_id": df["PLAYER_ID"].astype("int64"),
            "player": df["PLAYER_NAME"].astype("category"),
            "team": df["TEAM_NAME"].astype("category"),
            "x": x,
            "y": y,
            # Distance recomputed from coordinates. The file's SHOT_DISTANCE is
            # rounded to the foot, which puts visible banding in the model's
            # distance splits.
            "dist": np.hypot(x, dy),
            "abs_x": np.abs(x),
            # 0 deg is straight along the baseline, 90 deg is straight on.
            "angle": np.degrees(np.arctan2(dy, np.abs(x) + 1e-9)),
            "is_three": (df["SHOT_TYPE"] == "3PT Field Goal").to_numpy(),
            "family": action_family(df["ACTION_TYPE"]),
            "zone": df["BASIC_ZONE"].astype("category"),
            "period": df["QUARTER"].clip(upper=5).astype("int8"),
            "secs_left": (df["MINS_LEFT"] * 60 + df["SECS_LEFT"]).astype("int16"),
            "made": df["SHOT_MADE"].to_numpy(),
        }
    )

    abbrev = team_abbreviations(df)
    out["team_abbrev"] = out["team"].map(abbrev).astype("category")
    out["is_home"] = (out["team_abbrev"].astype(str) == df["HOME_TEAM"].to_numpy())
    out["points"] = np.where(out["made"], np.where(out["is_three"], 3, 2), 0).astype("int8")

    heave = (out["secs_left"] <= HEAVE_SECS) & (out["dist"] >= HEAVE_DIST)
    backcourt = out["zone"].astype(str).eq("Backcourt")
    out["excluded"] = heave | backcourt
    out.attrs["rescaled"] = rescaled

    # The file's own distance column, rounded to the foot, is an independent
    # check on the repair. If these ever disagree the coordinates are wrong.
    gap = (out["dist"] - df["SHOT_DISTANCE"].to_numpy()).abs()
    out.attrs["distance_gap"] = float(gap[~out["excluded"]].mean())
    if out.attrs["distance_gap"] > 1.0:
        raise ValueError(
            f"recomputed distance disagrees with SHOT_DISTANCE by "
            f"{out.attrs['distance_gap']:.2f} feet on average - coordinates are "
            f"not on the expected scale")
    return out
