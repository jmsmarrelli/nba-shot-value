"""Stage 1 - read the raw season files and write one model-ready table.

    python src/run_01_build.py

Reads data/raw/NBA_<season>_Shots.csv (see data/README.md for the fetch step)
and writes data/processed/shots.csv.gz plus a short audit of what was dropped.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT, PROC, RAW, SEASONS  # noqa: E402
import features  # noqa: E402

COLS = [
    "season", "game_id", "game_date", "player_id", "player", "team", "team_abbrev",
    "is_home", "x", "y", "dist", "abs_x", "angle", "is_three", "family", "zone",
    "period", "secs_left", "made", "points",
]


def main() -> None:
    frames, audit = [], []
    for season in SEASONS:
        path = RAW / f"NBA_{season}_Shots.csv"
        if not path.exists():
            print(f"  missing {path.name} - skipping", file=sys.stderr)
            continue
        t0 = time.time()
        raw = pd.read_csv(path, low_memory=False)
        tidy = features.build(raw)
        dropped = int(tidy["excluded"].sum())
        kept = tidy.loc[~tidy["excluded"], COLS]
        frames.append(kept)
        audit.append(
            {"season": season, "raw_rows": len(raw), "dropped": dropped,
             "kept": len(kept), "drop_pct": round(100 * dropped / len(raw), 3),
             "coords_rescaled": bool(tidy.attrs["rescaled"]),
             "distance_check_ft": round(tidy.attrs["distance_gap"], 3)}
        )
        flag = "  [coordinates rescaled]" if tidy.attrs["rescaled"] else ""
        print(f"  {season}  {len(raw):>7,} shots  dropped {dropped:>4}  "
              f"distance check {tidy.attrs['distance_gap']:.2f} ft"
              f"  ({time.time() - t0:.1f}s){flag}")

    if not frames:
        raise SystemExit("No raw season files found. See data/README.md.")

    shots = pd.concat(frames, ignore_index=True)
    out = PROC / "shots.csv.gz"
    shots.to_csv(out, index=False, compression="gzip")

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT / "build_audit.csv", index=False)

    print(f"\n{len(shots):,} shots, {shots.season.min()}-{shots.season.max()}")
    print(f"dropped {audit_df.dropped.sum():,} heaves and backcourt attempts "
          f"({100 * audit_df.dropped.sum() / audit_df.raw_rows.sum():.2f}%)")
    fixed = audit_df.loc[audit_df.coords_rescaled, "season"].tolist()
    print(f"coordinates rescaled in {len(fixed)} season(s): {fixed}")
    print(f"worst distance cross-check: {audit_df.distance_check_ft.max():.2f} ft")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
