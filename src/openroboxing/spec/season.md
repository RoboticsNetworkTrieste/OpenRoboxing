# season.md — the league

Version **0.1** · created 2026-08-08 · task `M5-T1`

Season 0's format is **already decided** in `docs/OpenRoboxing_project_definition_v0.8.md` §8. This
page is that decision made precise enough to implement, plus the handful of details §8 does not
cover. Anything invented here is marked and logged in `docs/ASSUMPTIONS.md`.

---

## Season 0, from the project definition

| Quantity | Value | Source |
|---|---|---|
| Length | **10 weeks**, one round per week | project definition §8 |
| Divisions | one, one table | §8 |
| Entry | unlimited, open | §8 |
| Pairing | **Swiss** | §8 |
| Cadence | register Monday, fight by Wednesday | §8 |
| Matches to appear on the table | **8** | §8 |
| Rating | **Glicko-2**, with confidence interval | §8 |
| Playoff | top 4 at week 10, at the Trieste Open | §8 |

## Ratings — Glicko-2

Glickman's published algorithm, unmodified. Parameters are his defaults, not ours:

| | Value |
|---|---|
| Initial rating | 1500 |
| Initial rating deviation (RD) | 350 |
| Initial volatility | 0.06 |
| System constant τ | **0.5** |

τ constrains how much volatility can move per period; Glickman's paper says "smaller values of τ
prevent the volatility measures from changing by large amounts" and recommends 0.3–1.2. 0.5 is the
middle of that range and the common default.

`league/rating.py` is checked against **Glickman's own worked example** — a 1500/200/0.06 player
against three opponents produces 1464.06 / 151.52 / 0.05999. That test is what makes the
implementation trustworthy; it is not a number we chose.

**A rating period is one week.** Glicko-2 is defined over periods containing several matches, and
Glickman notes it works best with ~10–15 matches per period. Season 0 gives one match per week, which
is fewer than ideal — the consequence is that RD shrinks more slowly than it would in a busier
league, i.e. the table is *conservative* about confidence. That is the right direction to be wrong in.

### The 8-match threshold

A fighter is **ranked** once they have played 8 matches; before that they are **provisional** and
appear in a separate list, not on the table.

`WORKPLAN` M5-T1 says this must "behave as specified", and the specification is one line in §8. Made
precise:

- The threshold counts **matches played**, not matches won, and not walkovers.
- A provisional fighter's rating still updates every period. They are hidden, not excluded — so on
  the match that takes them to 8 they arrive with a rating that already reflects their season.
- A ranked fighter never becomes provisional again.
- The table is ordered by **conservative rating** = `rating − 2 × RD`, not by rating. Two fighters on
  1600 are not equal if one has played 8 matches and the other 40, and ordering by the lower bound of
  the interval says so. This is the standard practice and is what "with confidence interval" in §8
  is for.

## Pairing — Swiss

Each week, every registered fighter is paired with an opponent on a similar score.

1. Sort by **score**, then by conservative rating, then by handle (so pairing is deterministic).
2. Walk the list pairing each unpaired fighter with the nearest unpaired fighter below them **whom
   they have not already played**.
3. If no such opponent exists, allow a rematch rather than leaving someone unpaired — a fighter with
   no match gets nothing from the week, which is worse than a repeat.
4. An odd fighter out receives a **bye**.

**Score** is 1 per win, 0.5 per draw, 0 per loss — the Swiss standard.

### Byes ○

Not covered by §8. A bye is worth **1 score point** and **does not count towards the 8-match
threshold**, and **does not update the rating**. A fighter cannot be rated on a match that never
happened. The lowest-scoring fighter who has not yet had a bye receives it, so byes spread out.

## The table

| Column | Meaning |
|---|---|
| rank | by conservative rating, descending |
| handle | the fighter |
| played / won / drawn / lost | matches, excluding byes |
| score | Swiss score, including byes |
| rating | Glicko-2 rating |
| rd | rating deviation |
| interval | `rating ± 2 × RD`, the 95% interval |

Provisional fighters are listed below the table, ordered the same way, marked as provisional.

## The playoff

Top 4 by conservative rating at the end of week 10, semi-finals 1v4 and 2v3, then the final.
Only **ranked** fighters are eligible: an unbeaten fighter with 3 matches has not earned a title shot.

## Determinism

A season is a pure function of its fixtures and results. `simulate_season.py` takes a seed, and the
same seed produces the same season — which is what makes "ratings converge" a testable claim rather
than an impression.

## Open — to be decided, not invented

- **Forfeits and no-shows.** §8 says "register Monday, fight by Wednesday" but not what happens when
  someone does not. Currently an unplayed fixture is simply absent; it neither scores nor rates.
- **Whether the playoff affects ratings.** Currently it does not; the belt is separate from the table.
- **Multiple divisions**, if entry outgrows one table. Out of scope for Season 0 by §8.

## Changelog

- **0.1** (2026-08-08) — first version. Season 0's format from the project definition, made precise:
  Glicko-2 defaults, the 8-match threshold, conservative-rating ordering, bye handling.
