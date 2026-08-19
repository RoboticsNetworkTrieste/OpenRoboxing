# First tuning pass — and a gap it exposed

> **Resolved 2026-08-08, then resolved again.** The gap below was first fixed with a held-key
> movement channel (`spec/intent.md` 0.2), which measured **circling 26 %, engagement 0.69, mean
> separation 1.24 m** against 67–80 % / 0.19–0.32 / 3.2–3.8 m below.
>
> Hours later the project owner remodelled the game: a move is now *place a shadow and commit*, so
> **walking is a commit** and the movement channel was retired (`spec/intent.md` 1.0,
> `docs/ASSUMPTIONS.md` §A19). The numbers above are therefore historical too. **Nothing here has
> been re-measured against the queued model** — that needs humans, which is `M4-T4`.
>
> The page is kept because the *finding* is what mattered: nobody reviewed their way to it, a tuning
> sweep did.

Run 2026-08-08 · `python -m openroboxing.tools.tune --knob commit_horizon --values 15 60`

Solo tuning ahead of `M4-T4`, as decided by the project owner. Two baseline agents, one 60 s round
per value.

## The numbers

| commit horizon | hits/min | commits/min | circling | engaged | mean separation |
|---|---|---|---|---|---|
| **15 ticks** (0.3 s) | 70 | 70 | **80 %** | 0.19 | 3.80 m |
| **60 ticks** (1.2 s) | 104 | 54 | **67 %** | 0.32 | 3.23 m |

*Circling* is the fraction of the round spent more than 1.8 m apart — the passivity failure mode
`WORKPLAN` M4-T4 names. *Engaged* is the scorer's own closeness weighting; 1.0 would be in range for
the whole round.

The counter-intuitive result is real and worth a second look at the playtest: **a longer windup
produced more hits and less circling**, not fewer. A short horizon lets a fighter commit constantly,
and constant committing is not the same as constant fighting.

## ⚠ The finding that matters more than the sweep

**Both settings spend most of the round out of range — 67 % and 80 % — in a 4.90 m ring with a mean
separation over 3 m.** The fighters are at opposite ends of the ring for most of the fight.

Chasing that down: **a player has no way to walk towards the opponent.**

- `GeneratorIntent.movement_angle` exists and **is never set by anything**. It defaults to 0, meaning
  "the generator's own forward", which after the heading alignment is some fixed world direction that
  has nothing to do with where the opponent is.
- `FightWorld.facing_angle` makes a fighter *turn to face* the other one, so they look at each other
  while walking somewhere else entirely.
- The only positional control in the game is a commit's optional `placement`, and no pose in the
  library sets one.

So distance management — which in boxing is most of the sport — is currently not a thing a player
can do. That is a **design gap, not a tuning problem**, and no value of any knob in `tools/tune.py`
will fix it.

### Why the agents do not disguise it

`BaselineAgent` cannot close either — it cannot even see the separation, because
`spec/protocol.md` does not send it. So these numbers are the *floor*: they show what happens when
nobody is managing distance, which is currently what the rules allow. A human would be no better off.

### What the fix probably looks like

Not for me to choose, but the shape is narrow:

1. **A movement channel in the intent** — `spec/intent.md` already has staging channels; add a
   direction the player steers continuously, feed it to `movement_angle`. Smallest change, and it
   matches the "always steering, never paused" design the intent spec already describes.
2. **Send separation in the state** so a client (and an agent) can see range at all. One number,
   and it is not secret — both fighters can see how far apart they are by looking.
3. **Or make it automatic**: fighters always close to range unless committing. Removes distance
   management as a skill rather than adding it, which is probably the wrong direction for a game
   about timing.

(1) plus (2) is the version I would expect to want, and it is a small change to a spec that already
anticipated it. Logged as `docs/ASSUMPTIONS.md` §A18.

## Reproduce

```bash
python -m openroboxing.tools.tune --list
python -m openroboxing.tools.tune --knob commit_horizon --values 15 30 60 --repeats 3
python -m openroboxing.tools.tune --knob ring_size --values 3.0 4.0 4.9 --repeats 3
```

Every number here is a proxy. They find settings worth putting in front of people; they do not
decide whether it is fun.
