"""Paths, constants and the small set of knobs the pipeline exposes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
OUT = ROOT / "output"

for d in (RAW, PROC, FIG, OUT):
    d.mkdir(parents=True, exist_ok=True)

# Season label is the year the season ends: 2025 == 2024-25.
FIRST_SEASON = 2004
LAST_SEASON = 2025
SEASONS = list(range(FIRST_SEASON, LAST_SEASON + 1))

# Before 2010-11 the league logged roughly four out of five restricted-area
# attempts at the rim coordinate itself rather than where the shot was taken:
# 26% of all attempts in those seasons sit at exactly zero feet, against 5%
# afterwards. Fine-grained coordinates are therefore not comparable across that
# boundary, so anything that leans on them is restricted to this season onward.
# Zone labels are coarse enough to be unaffected and keep the full span.
COORD_ERA_START = 2011

RIM_Y = 5.25          # rim centre, in the shot files' coordinate system
HEAVE_DIST = 30.0     # a shot from here with the clock about to expire is a heave
HEAVE_SECS = 2.0

# Minimum attempts for a player-season to appear in the leaderboards. 300 is
# roughly a rotation player who played most of the year; below that the shot
# making estimate is mostly noise, which is the point the reliability work makes.
MIN_FGA = 300
MIN_FGA_TEAM = 1000

SEED = 20240117
N_FOLDS = 5
