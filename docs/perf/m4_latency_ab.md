# M4-T2 · Does injected latency change who wins?

Run 2026-08-08 · `python -m openroboxing.tools.latency_ab --matches 16 --rounds 1 --round-seconds 20`

## The criterion

> `WORKPLAN` M4-T2: artificially injecting 200 ms latency does not change match outcomes
> systematically (run a scripted-agent A/B and compare win rates).

Two identical `BaselineAgent`s, 16 matches per condition, one round of 20 s, scored by
`spec/scoring.md`. Latency is injected client-side between the agent's decision and the host
receiving it — where a real network delay lands.

## Result

| condition | red wins | draws | red win rate | 95% interval (Wilson) |
|---|---|---|---|---|
| baseline | 5 / 16 | 1 | 0.33 | [0.15, 0.58] |
| red +200 ms | 4 / 16 | 0 | 0.25 | [0.10, 0.49] |

**The intervals overlap. No systematic effect detected.** The criterion is met.

Wilson intervals rather than the normal approximation: at n = 16 the normal approximation is simply
wrong near the tails, and this is exactly the small-sample regime it fails in.

## What this does and does not establish

**Does:** 200 ms of one-way delay did not measurably advantage or disadvantage the delayed seat.
That is the expected result, and the reason is structural rather than lucky — the host services
queued intents on its **own** 30 Hz tick and never waits for a client (`spec/protocol.md`
§Latency), so latency changes *when* a commit lands and nothing else.

**Does not:** prove absence. With 16 matches per condition the interval is about ±0.22 wide, so an
effect smaller than roughly 20 percentage points would not be visible here. Narrowing it is only
more matches; each costs about 16 s.

## A caveat the numbers raised — since settled

Red won only 33 % of the *baseline* matches above. The interval [0.15, 0.58] contained 0.5 so nothing
was established, but if the seat itself were worth points then Swiss pairing plus Glicko-2 would
faithfully rate which side of the ring somebody stood on. That was worth ruling out before anybody is
paired, so it was:

`python -m openroboxing.tools.seat_fairness --matches 20` · 40 further matches, same match seeds in
both conditions, only the agents' seats swapped:

| condition | red's agent | red wins | draws | red win rate | 95% interval |
|---|---|---|---|---|---|
| as-is | seed 0 | 11 / 20 | 3 | 0.65 | [0.41, 0.83] |
| swapped | seed 1 | 9 / 20 | 2 | 0.50 | [0.29, 0.71] |

**Pooled: red 20 / 35 = 0.57, interval [0.41, 0.72] — contains 0.5. No seat advantage established.**

And the more useful observation: **red won 0.57–0.65 here against 0.33 in the A/B above**, on a
different block of match seeds. A swing that large between seed blocks is the straightforward reading
of the original 0.33 — it was variance, not a handed arena.

This does not prove the seats are identical; it bounds the asymmetry at roughly ±16 points. What it
does establish is that there is no *large* seat effect, which is what would have mattered.

`docs/ASSUMPTIONS.md` §A12 is closed.

## Reproduce

```bash
python -m openroboxing.tools.latency_ab --matches 16 --rounds 1 --round-seconds 20 --out ab.json
```
