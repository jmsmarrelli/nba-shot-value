# Data

The pipeline reads one CSV per season at `data/raw/NBA_<season>_Shots.csv`, where
`<season>` is the year the season ended: `NBA_2025_Shots.csv` is 2024-25.

## Getting the files

The shot-level files come from
[DomSamangy/NBA_Shots_04_24](https://github.com/DomSamangy/NBA_Shots_04_24),
which packages the NBA's public shot chart endpoint into one zipped CSV per
season, 2003-04 through 2024-25.

```bash
git clone --depth 1 https://github.com/DomSamangy/NBA_Shots_04_24.git /tmp/nba_shots
for z in /tmp/nba_shots/NBA_*_Shots.csv.zip; do unzip -o -q "$z" -d data/raw/; done
```

About 850 MB unzipped. `data/raw/` and `data/processed/` are both gitignored;
everything the analysis actually cites is written to `output/` and committed.

## Columns used

| Column | Used for |
|---|---|
| `SEASON_1` | season, labeled by ending year |
| `GAME_ID` | fold grouping, opponent lookup, odd/even splits |
| `PLAYER_ID`, `PLAYER_NAME` | player-season aggregation |
| `TEAM_NAME`, `HOME_TEAM`, `AWAY_TEAM` | team identity and venue |
| `LOC_X`, `LOC_Y` | distance, angle, court position |
| `SHOT_TYPE` | two or three |
| `ACTION_TYPE` | action family (within-season model only) |
| `BASIC_ZONE` | zone summaries and the backcourt filter |
| `QUARTER`, `MINS_LEFT`, `SECS_LEFT` | clock features and the heave filter |
| `SHOT_MADE` | the target |

Coordinates are in feet with the rim at (0, 5.25). `SHOT_DISTANCE` is present in
the source but rounded to the foot, so distance is recomputed from the
coordinates instead — the rounding puts visible banding in the model's splits.

## A note on team abbreviations

`TEAM_NAME` is a full franchise name and `HOME_TEAM` / `AWAY_TEAM` are
three-letter codes, with no direct mapping in the file. `features.team_abbreviations`
recovers it by intersection: a team's own code is the one appearing in every game
it played. This handles relocations and rebrands across the twenty-two seasons
without a lookup table to maintain.
