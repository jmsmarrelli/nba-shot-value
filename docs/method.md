# Method notes

Longer notes on the decisions that move the results. The README has the findings;
this has the reasoning, and the places where I think the analysis is weakest.

## The decomposition

For any set of attempts indexed by *i*, with realized points `p_i` and modeled
expectation `x_i`:

```
mean(p) = mean(x) + mean(p − x)
```

That is an identity, not a model, and it holds exactly for any `x`. All the
content is in what `x` is allowed to depend on. Put location in `x` and the first
term becomes "the value of the shots taken". Put shooter identity in `x` and it
becomes something else entirely, and the second term goes to zero.

So the decomposition is only as meaningful as the answer to: *what is the
expectation conditioned on, and does that condition mean the same thing at both
ends of the comparison?*

For this data, over this span, the answer turned out to be "almost nothing does".

## Three expectations, and what each is for

**ZONE** — the league's own conversion rate in each of five zones (restricted
area, paint outside it, mid-range, corner three, above-the-break three), pooled
over all twenty-two seasons. No model, no coordinates, five numbers. This is the
era yardstick, and it is the headline in the README.

**GEO** — a gradient-boosted model on distance from the rim, absolute horizontal
position, distance up the floor, angle from the baseline, and whether the shot is
worth three. Used for the value surface, and for era comparisons restricted to
2010-11 onward.

**FULL** — GEO plus action family (eleven groups collapsed from 48 raw strings),
period, seconds left in the period, and home or away. The best model on every
scoring rule, used for player-season and team-season work, where every comparison
happens inside one season.

The reason for three rather than one is the next two sections.

## The action labels cannot cross eras

Share of all field goal attempts, by action family:

| Family | 2003-04 | 2011-12 | 2019-20 | 2024-25 |
|---|---|---|---|---|
| Jump shot | 62.3% | 57.0% | 35.1% | 31.8% |
| Layup | 16.2% | 13.2% | 4.0% | 1.6% |
| Driving layup | 5.5% | 6.7% | 14.0% | 13.7% |
| Pull-up | — | 2.5% | 11.7% | 13.0% |
| Floater | — | 1.2% | 7.7% | 9.5% |
| Step-back | — | 0.8% | 4.3% | 5.7% |

Three of those families are literally absent from the 2003-04 file. They are not
new shots — players took pull-ups and floaters in 2004 — they are new *labels*,
and every attempt that now gets one used to be filed under "jump shot" or "layup".

The consequence for a pooled model is direct. Conversion by family, 2003-06
against 2022-25:

| Family | 2003-06 | 2022-25 |
|---|---|---|
| Driving layup | 74.7% | 54.3% |
| Layup | 44.9% | 51.6% |
| Turnaround / fade | 56.8% | 42.3% |

Driving layups did not get twenty points harder. The category widened. A model fit
across both eras splits the difference and hands back a residual that is
scorekeeping rather than basketball. Running the era decomposition on FULL gives
shot selection 113% of the change and shot making −2%; running it on the zone
yardstick gives 38% and 62%.

## Neither can the coordinates

This one I found later and it is larger.

| Season | Share of all attempts logged at 0.0 feet | FG% of those attempts |
|---|---|---|
| 2003-04 | 25.4% | 59.3% |
| 2007-08 | 26.4% | 60.4% |
| 2009-10 | 25.9% | 61.1% |
| **2010-11** | **3.9%** | **74.2%** |
| 2014-15 | 7.5% | 82.1% |
| 2019-20 | 2.4% | 83.4% |
| 2024-25 | 4.1% | 71.4% |

Before 2010-11, about four out of five restricted-area attempts carry the rim's
own coordinate rather than the shooter's — a quarter of the entire file sits at
exactly zero feet. Afterwards, only genuine dunks and tip-ins land there, which is
why the conversion rate of a "zero-foot shot" jumps from 61% in 2009-10 to 74% in
2010-11.

Any model that reads fine coordinates therefore sees a different game before and
after 2010-11. Fit one on three middle seasons and apply it across the boundary
and it values a 2004 shot chart at 1.11 expected points and a 2013 one at 0.99,
which is nonsense in the direction of "the league got worse at shooting", and it
is entirely the recording change.

So `COORD_ERA_START = 2011` in `config.py`, and everything that touches
coordinates respects it: the value surface, the migration map, the reliability
curves, and the team-season work. The zone yardstick does not, because a shot
logged at the rim and the same shot logged eighteen inches away are both
"Restricted Area".

There is a third, smaller version of the same problem. Three season files —
2019-20, 2020-21 and 2021-22 — are published on a different coordinate scale
entirely, with x divided by ten and y divided by ten and shifted by the rim
offset. `features.normalise_coordinates` detects and inverts it, and
`output/build_audit.csv` carries the cross-check: recomputed distance against the
files' own rounded distance column, which agrees to within half a foot in all
twenty-two seasons after the repair and is off by twelve feet in those three
before it.

## Choosing the fold structure

Folds are grouped by `game_id`. Two alternatives were worse:

* **Random row folds** leak. Shots from the same game share a defense, a venue, a
  night's shooting luck, and often a shooter. A model that has seen 4,000 shots
  from a game predicts the 4,001st partly by memory.
* **Season folds** would fix the leakage but make the expectation era-specific,
  which defeats the purpose.

Grouping by game is the tightest split that keeps a single yardstick.

## Heaves

Attempts with two seconds or less left in a period from 30 feet or further, and
anything the source labels "Backcourt", are dropped: 18,735 shots, 0.42% of the
file. They are not shot selection decisions and they put a long, thin, low-value
tail on the distance feature that the model would otherwise spend splits on.

## Centering

Shot making is centered on the league mean for its season before any player or team
comparison. Without it, a 2011 shooter and a 2025 shooter are graded against each
other rather than against their peers, and every cross-era leaderboard becomes a
leaderboard of eras.

Shot quality is deliberately left uncentred at the league level, because its drift
across eras is a finding rather than a nuisance — but it *is* centered within
season anywhere a correlation is computed across pooled seasons, for the same
reason. Skipping that step inflates the team-defence carryover correlation from
0.72 to 0.95, which is a tidy demonstration of how easy it is to publish the
calendar as a result.

## Reliability

The split-half procedure:

1. Shuffle every shot.
2. Within each player-season, take the first *m* attempts as half A and the next
   *m* as half B. Disjoint by construction. Player-seasons with fewer than 2*m*
   attempts drop out at that *m*.
3. Compute the metric on each half, center within season, correlate across
   player-seasons.
4. Repeat over twelve shuffles and average.

Reported as the raw split-half correlation, with the Spearman–Brown projection to
full length alongside it in `output/reliability_curve.csv`.

Step 2 has a selection effect worth naming: as *m* grows, the surviving sample
shifts toward high-volume players, who differ from the population in ways beyond
volume. The curve at *m* = 600 describes 268 player-seasons of primary options,
not the league. The half-signal point at 303 attempts sits where the sample is
still 1,970 player-seasons deep, which is the range I would actually quote.

## Where I think this is weakest

**Openness is the whole missing variable.** Everything filed under "shot making"
here is conversion above expectation given location. Some of that is shooting
talent and some of it is being wide open, and the public files cannot separate
them. This is why I stop short of claiming the two-point conversion gain is better
finishing — spacing is at least as plausible and I cannot test it. It is also why
the defensive residual persists year to year: contests are real defense happening
inside a term I have labeled as if it were luck.

**The zone yardstick is coarse.** Five buckets is the price of a measure that
means the same thing in 2004 and 2025. Inside "above the break", a 24-footer and a
30-footer are the same shot, and the league has been moving from the first to the
second, which the yardstick reads as no change in shot quality. That biases the
headline *against* the location term, so the true location share is probably a
little above 38% — and the 2010-11-onward geometry check, which can see the
difference, puts it lower rather than higher. I have not resolved that tension and
I would want to before publishing this anywhere it mattered.

**Free throws are absent**, so this is shot value and not offensive value. A
player who turns pressure into trips to the line is systematically undersold here,
and a few of the names near the bottom of the shot-making list are there partly
for that reason.

**Two seasons are structurally odd.** 2011-12 was a 66-game lockout season, and
2019-20 and 2020-21 were interrupted and partly played without crowds. They are
kept, because the analysis is per-attempt rather than per-season, but the
season-to-season correlations that bridge them carry an extra source of noise.
