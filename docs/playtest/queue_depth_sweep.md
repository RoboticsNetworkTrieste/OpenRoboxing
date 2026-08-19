# How deep should the queue be? — a solo sweep

Run 2026-08-08 · `python -m openroboxing.tools.tune --knob queue_depth --values 1 2 3 5 8 --repeats 4`

`MAX_OUTSTANDING_COMMITS = 5` was chosen by the project owner when a committed move lasted 0.8–2.1 s,
which made a full queue 4–10 s of pre-planned action against a 60 s round (`docs/ASSUMPTIONS.md`
§A19). `spec/intent.md` 1.1 then made a commit run until it arrives, so a move is now its **walk plus
its pose**. The premise the five was chosen under is gone.

This is the step before `M4-T4`, as before: two baseline agents, one 60 s round per run, four runs per
value, so the owner arrives at the playtest with a shortlist rather than a shrug.

## The numbers

| queue depth | hits/min | commits/min | commit | of which walking | **unfinished at the bell** | engaged |
|---|---|---|---|---|---|---|
| **1** | **109.0** | 39.0 | 2.8 s | 1.9 s | **0.5** | 0.95 |
| **2** | 84.5 | 63.4 | 2.4 s | 1.5 s | **0.5** | 0.88 |
| **3** | 66.2 | 77.0 | 1.7 s | 0.8 s | **1.8** | 0.87 |
| **5** *(current)* | 74.6 | 74.9 | 1.9 s | 0.9 s | **6.0** | 0.91 |
| **8** | 72.2 | 79.0 | 2.2 s | 1.3 s | **12.2** | 0.89 |

*Engaged* is the scorer's own closeness weighting; 1.0 would be in range for the whole round.
Circling — the passivity failure mode `M4-T4` names — sat at 2–6 % throughout and separates nothing,
which is itself worth knowing: passivity is no longer the failure mode to design against.

## What it says

**A deeper queue lands fewer punches.** Depth 1 landed 109 hits/minute; everything from 3 upward
landed 66–75. The mechanism is visible in the other columns: a deeper queue does not make a fighter
busier in any useful sense — commits/minute roughly doubles from 39 to 77 — it makes each commit
*staler*. A move issued five deep is aimed at where the opponent stood several seconds ago, and it
walks there anyway, because there is no cancellation.

**At five, the queue outlives the round.** Six commits per round are still unfinished when the bell
goes at depth 5, and twelve at depth 8, against half a commit at depths 1 and 2. Those are moves a
player paid for and never saw. In a 60 s round with a 3-round match that is a substantial fraction of
what a player did being thrown away by the clock rather than by the opponent.

**Nothing here is knocked out by walking.** 2–4 % of commits hit the approach timeout and threw where
they stood, at every depth — so the ring is walkable and the timeout is not doing heavy lifting.

## What it does not say

**Whether it is fun**, as ever. Every number is a proxy, and this one has a specific blind spot: the
reference agent waits `RECOVERY_TICKS` (0.5 s) between commits and reacts to a state message, so it
fills a queue far more sedately than a person mashing a key. A human at depth 8 would plausibly stack
eight moves in two seconds and then watch for twenty. **The 8 row understates the failure mode.**

Four repeats is also not many for a metric this noisy: hits/minute varied by more than 2× between
repeats at the same setting, which is why only the large effects above are stated.

## The shortlist for `M4-T4`

1. **Try 2.** It keeps the "commit ahead and live with it" tension — you can still be one move
   committed while planning the next — with almost nothing lost to the bell, and it landed 84
   hits/minute against 5's 75.
2. **Do not raise it.** 8 is worse on every column that matters and much worse on the one that does.
3. **If 5 stays, shorten what it costs**, not the number: a queue is deep in *seconds*, not commits,
   and the seconds come from walking. A smaller ring (`--knob ring_size`) reduces the same
   commitment without touching the rule.

None of this has been changed in code. `MAX_OUTSTANDING_COMMITS` is the project owner's decision
(`docs/ASSUMPTIONS.md` §A19, §A23) and a sweep of two agents does not get to overrule a feel call.

## Reproduce

```bash
python -m openroboxing.tools.tune --knob queue_depth --values 1 2 3 5 8 --repeats 4 --out queue.json
```
