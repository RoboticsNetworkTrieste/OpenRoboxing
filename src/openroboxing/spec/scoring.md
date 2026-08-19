# scoring.md — how a match is judged

Version **0.1** · created 2026-08-08 · task `M5-T2`

Scoring reads a `MatchRecord` and nothing else. It never re-simulates, never opens a checkpoint, and
never looks at a policy — so a rule change re-scores the whole archive in seconds, and two people
running the same version on the same record must get the same table.

> **`WORKPLAN` M5-T2 says aggression and ring control "need concrete definitions first — that is
> workstream 03's job".** Those definitions did not exist when this was written. The ones below are
> **mine, not yours**, and they are logged in `docs/ASSUMPTIONS.md` §A1. A placeholder was the other
> option and it was worse: the acceptance criterion is that a human agrees with the score on eight of
> ten replays, and nobody can agree or disagree with a constant.

---

## The four dimensions

`WORKPLAN` names them: landed impulses, knockdowns, ring control, aggression. Each produces a number
per fighter per round, and the round is awarded from their weighted sum.

Every dimension is a **rate or a fraction**, never a raw count. A round cut short by a knockout is
shorter than a round that reached the bell, and a fighter must not score less for having ended one
early.

### 1 · Damage — landed impulses

```
damage = sum over this fighter's hits of  impulse_ns x region_weight
```

| Region | Weight | Why |
|---|---|---|
| head | **1.0** | reference |
| body | **0.7** | boxing scores both; the body is worth less per blow and is the safer target |
| arm | **0.0** | a punch stopped by the guard did not land — it is the guard working |
| leg | **0.0** | out of distribution: the policy was trained penalising contact outside feet, hands, elbows |
| other | **0.0** | unattributed |

Impulse rather than peak force. Peak force is a single substep's spike and is dominated by contact
stiffness — a glancing touch at a bad timestep can out-peak a real punch. Impulse is force integrated
over the exchange, which is the quantity that would actually move a head.

**Arms score zero, deliberately.** This is the one weight most likely to be argued with. A blocked
punch is *the defender succeeding*; paying the attacker for it would reward throwing at a guard.

### 2 · Knockdowns

```
knockdowns  = count of the opponent's knockdown episodes
knockouts   = count of those that reached the get-up window
```

Scored on the round, not summed into damage: going down is a categorical event, not more damage.

### 3 · Ring control

```
control = mean over ticks of  centre_advantage(t) x engagement(t)

centre_advantage(t) = 1 if this fighter is nearer the ring centre than the opponent, else 0
engagement(t)       = clamp(1 - (separation(t) - contact_range) / engagement_falloff, 0, 1)
```

with `contact_range = 0.80 m` and `engagement_falloff = 1.20 m`.

Ring control in boxing means *cutting the ring off* — owning the middle while the opponent works off
the back foot. Two parts, and both are needed:

- **Centre advantage** is who holds the middle. It is a comparison, not a distance, so the number
  means the same thing in any ring size (`ArenaConfig.ring_size` is a parameter, and `M4-T4` will
  change it).
- **Engagement** is the correction that stops it being free. Standing in the middle while the
  opponent is three metres away is not control, it is being alone. Weighting by closeness means
  control is only earned while the opponent is actually being pressured.

`contact_range` is **measured, not chosen**: the G1's hand reaches 0.38 m forward of its own pelvis
(`studio/pose_ik.py`), so two fighters can exchange at a pelvis separation up to ~0.76 m; 0.80 m is
that rounded up. `engagement_falloff` reaches zero at 2.0 m separation, which is
`ArenaConfig.start_separation x 2` less a little — i.e. control decays to nothing by the distance the
fighters start at.

### 4 · Aggression

```
aggression = (committed moves thrown in range) / minute / target_rate
```

with `target_rate = 12` commits per minute, clamped to `[0, 1.5]`.

A commit counts if the fighters were within `contact_range x aggression_reach` (**1.6**, so 1.28 m)
at the tick the punch was *thrown* — `strike_at`, not `issued_at` and, since `spec/intent.md` 1.1,
not `commit_at` either. `commit_at` is now the start of a walk that may run for seconds, so scoring
on it would credit a fighter for ground it was still covering. A commit that never threw before the
bell scores nothing.

`target_rate = 12/min` was **derived, not chosen**: a move runs 6–16 tokens ≈ 0.8–2.1 s
(`spec/rates.md`), and at the time exactly **one commit could be active**, so a fighter who is always
committing manages 28–75 per minute. Twelve is roughly a *fifth* of that maximum — busy without being
frantic — and it is the rate at which the `>1.0` region begins, so exceeding it is possible and means
something.

> ⚠ **The premise moved, twice.** `spec/intent.md` 1.0 allowed five outstanding commits and made a
> *step* a commit too. 1.1 then made a commit run until it arrives, so a move is no longer 0.8–2.1 s
> but its walk plus its pose — **measured at 3–5 s** each in a 4.90 m ring. A fighter that is always
> committing now manages roughly **12–20 per minute**, so the target this was a *fifth* of has become
> something close to the *maximum*: the same number now reads as "commit constantly" rather than
> "commit often", and a cagey round scores near zero.
>
> It has **not** been changed blind, twice over: re-deriving it wants a tuning sweep with humans
> playing the queued model (`M4-T4`), and the direction to move it in depends on whether five moves
> deep is itself too much — the same open question. Until then aggression is comparable within a
> season and not across the change, which `league/manifest.py`'s pinning already records.
> See `docs/ASSUMPTIONS.md` §A22 and §A23.

The clamp at 1.5 stops a spam strategy from outscoring everything else: past 18 commits per minute
there is no further reward.

**This dimension exists to punish the passivity failure mode** that `WORKPLAN` M4-T4 names — two
fighters circling and never engaging. A round where nobody commits in range scores zero here for
both, so it cannot be won on aggression by default.

---

## The round score

Boxing's **10-point must**: the round winner gets 10, and the loser gets less.

| Situation | Winner | Loser |
|---|---|---|
| Round won on points | 10 | **9** |
| One knockdown against the loser | 10 | **8** |
| Two or more knockdowns | 10 | **7** |
| Round ended in a knockout | 10 | **7** |
| Nothing separates them | 10 | 10 |

These are boxing's own numbers, taken because they are boxing's and not derived (`docs/ASSUMPTIONS.md`
§A2). A knockout is worth the same as a two-knockdown round, which keeps it decisive within a round
without contradicting the rule that **a knockout does not end the match**
(`spec/match_record.md`).

Who wins the round on points is the weighted sum:

| Dimension | Weight |
|---|---|
| damage | **0.50** |
| ring control | **0.25** |
| aggression | **0.25** |

Damage is half because landing punches is what boxing is. Control and aggression split the rest
evenly: they are the two dimensions that describe *how* a fighter took the round, and neither should
outrank hitting.

Damage is normalised **within the round** — each fighter's share of the two — so a cagey round and a
brawl are both scored out of the same total. A round where neither lands anything scores 0.5/0.5 on
damage and is decided by the other two.

**A round is drawn (10–10) only when the two scores are within `DRAW_MARGIN = 0.02`.** Without a
margin, floating-point noise decides rounds that are visibly even.

## The match

The winner is whoever has more round points. Equal points is a **draw**; there is no countback, on
purpose — a countback rule invents a tiebreak hierarchy nobody has agreed to, and a draw is an honest
answer.

The match record is not modified. A score is a *derivation* from a record, written beside it, so
re-scoring an archive under new rules never rewrites history.

---

## What this deliberately does not do

- **No judges, no rounds-won-by-majority.** One scorer, deterministic.
- **No defence dimension.** Blocking already pays: it moves an opponent's hits into the arm region,
  which is worth zero. A separate defence score would pay twice.
- **No stamina, no damage model.** Nothing accumulates across rounds (`docs/ASSUMPTIONS.md` §A3).
- **No style points.** Every input is measured off the record.

## Open — to be decided, not invented

- **Whether aggression should be symmetric.** Right now both fighters can score well on it. An
  argument exists that pressure is zero-sum, like ring control.
- **Whether a knockout should carry beyond its round.** Currently it does not.
- **Weights.** 0.50/0.25/0.25 has no evidence behind it yet. `M5-T2`'s acceptance criterion — ten
  replays, a human agreeing with eight — is the experiment that should move these.

## Changelog

- **0.1** (2026-08-08) — first version. Four dimensions, 10-point must, damage weighted 0.50.
  Aggression and ring control defined here for the first time.
