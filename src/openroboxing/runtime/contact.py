"""Contact sensing and hit attribution (M3-T3).

Physics decides the hit. This module reads what MuJoCo saw and turns it into events a match can
score: who hit whom, where, how hard, and when.

A hit is an episode, not a contact
----------------------------------
A punch produces contact on every substep it is touching — dozens of rows in ``data.contact`` for one
punch. Emitting an event per row would make a single jab outscore a real combination. So contacts are
accumulated into **episodes**: contact between the same pair of bodies, unbroken (allowing
:data:`EPISODE_GAP_TICKS` of separation for bounce), is one :class:`HitEvent` carrying the peak force
and the total impulse.

What counts as a hit
--------------------
Only a **glove** landing on the **opponent** counts. Not a shin, not a shoulder barge, not the ropes.
This is a rule, not a limitation: the policy was trained penalising contact outside feet, hands and
elbows, so anything else is out of distribution and would be scoring noise rather than boxing.

Conventions
-----------
- **Impulse** is in N·s, the force integrated over the episode's duration. **Peak force** is in N.
  Both come from ``mj_contactForce``, whose first component is the normal force.
- **Position** is the contact point in world coordinates, taken at the moment of peak force.
- Attribution is by **body name prefix** (``red_`` / ``blue_``), never by position: two fighters in a
  clinch are not separable geometrically. It has to be the *body* name rather than the geom's,
  because the robot model leaves its collision geoms unnamed — reading geom names attributes almost
  every real hit to nobody, and does it silently.
- Whether a contact is a **glove** is read from the geom name, because the arena names those itself
  and they share a body with the rest of the hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openroboxing.runtime.arena import FIGHTERS

#: How long contact may lapse before the episode is considered over, in ticks. A glove that skips off
#: a guard and lands again is one exchange; two clean punches are separated by far more than this.
EPISODE_GAP_TICKS = 3

#: Where on the body a hit landed. Boxing scores head and body differently, so the distinction is
#: made here rather than left to the scorer. Matched against the *stripped* body name, longest first.
BODY_REGIONS: tuple[tuple[str, str], ...] = (
    ("head", "head"),
    ("wrist", "arm"),
    ("elbow", "arm"),
    ("shoulder", "arm"),
    ("torso", "body"),
    ("pelvis", "body"),
    ("waist", "body"),
    ("hip", "leg"),
    ("knee", "leg"),
    ("ankle", "leg"),
)


class ContactError(RuntimeError):
    """Contact could not be read or attributed. Never recovered from silently."""


@dataclass(frozen=True)
class HitEvent:
    """One landed punch."""

    attacker: str
    defender: str
    attacker_body: str
    defender_body: str
    region: str
    start_tick: int
    end_tick: int
    peak_force_n: float
    impulse_ns: float
    position: tuple[float, float, float]

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick + 1


@dataclass
class _Episode:
    """A hit in progress. Not part of the public surface."""

    attacker: str
    defender: str
    attacker_body: str
    defender_body: str
    start_tick: int
    last_tick: int
    peak_force_n: float
    impulse_ns: float
    position: tuple[float, float, float]


def fighter_of(geom_name: str) -> str | None:
    """Which fighter a geom belongs to, or ``None`` for ring furniture."""
    for fighter in FIGHTERS:
        if geom_name.startswith(f"{fighter}_"):
            return fighter
    return None


def strip_fighter(geom_name: str) -> str:
    """A geom's name without its fighter prefix."""
    fighter = fighter_of(geom_name)
    return geom_name[len(fighter) + 1 :] if fighter else geom_name


def region_of(body_name: str) -> str:
    """Which scoring region a body belongs to. Unknown bodies are ``"other"``, never guessed."""
    stripped = strip_fighter(body_name)
    for token, region in BODY_REGIONS:
        if token in stripped:
            return region
    return "other"


def is_glove(geom_name: str) -> bool:
    return "glove" in geom_name


class ContactTracker:
    """Accumulates contacts into hit events, one tick at a time.

    Fed from the match loop after each tick's physics. Holds no reference to the model, so a match
    can be replayed through it from a recorded trace as easily as from a live simulation.
    """

    def __init__(self, gap_ticks: int = EPISODE_GAP_TICKS) -> None:
        if gap_ticks < 0:
            raise ContactError(f"gap_ticks must not be negative, got {gap_ticks}")
        self.gap_ticks = gap_ticks
        self._open: dict[tuple[str, str], _Episode] = {}
        self.events: list[HitEvent] = []

    def observe(self, model, data, tick: int) -> None:
        """Read this tick's contacts. Call once per control tick, after stepping physics."""
        import mujoco

        force = np.zeros(6)
        for index in range(data.ncon):
            contact = data.contact[index]
            first = self._describe(mujoco, model, contact.geom1)
            second = self._describe(mujoco, model, contact.geom2)

            landed = self._as_punch(first, second)
            if landed is None:
                continue
            attacker_body, defender_body = landed

            mujoco.mj_contactForce(model, data, index, force)
            normal = abs(float(force[0]))
            if normal <= 0.0:
                continue  # touching but not pushing; not a punch

            self._accumulate(
                attacker_body, defender_body, normal, tuple(contact.pos), tick, model.opt.timestep
            )

        self._close_stale(tick)

    @staticmethod
    def _describe(mujoco, model, geom: int) -> tuple[str, bool]:
        """``(body name, is a glove)`` for a geom.

        The body carries the fighter prefix; the geom name is usually empty, so it is only good for
        spotting a glove — which the arena does name.
        """
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        return body, is_glove(geom_name)

    def _as_punch(self, first, second):
        """``(attacker body, defender body)`` if this contact is a glove on an opponent."""
        (first_body, first_glove), (second_body, second_glove) = first, second
        owners = (fighter_of(first_body), fighter_of(second_body))
        if None in owners or owners[0] == owners[1]:
            return None  # ring furniture, or a fighter touching itself
        if first_glove and not second_glove:
            return first_body, second_body
        if second_glove and not first_glove:
            return second_body, first_body
        return None  # glove-on-glove is a parry, and body-on-body is a clinch

    def _accumulate(self, attacker_body, defender_body, normal, position, tick, timestep) -> None:
        attacker = fighter_of(attacker_body)
        defender = fighter_of(defender_body)
        key = (attacker_body, defender_body)

        episode = self._open.get(key)
        if episode is None:
            episode = _Episode(
                attacker=attacker,
                defender=defender,
                attacker_body=strip_fighter(attacker_body),
                defender_body=strip_fighter(defender_body),
                start_tick=tick,
                last_tick=tick,
                peak_force_n=0.0,
                impulse_ns=0.0,
                position=position,
            )
            self._open[key] = episode

        episode.last_tick = tick
        episode.impulse_ns += normal * timestep
        if normal > episode.peak_force_n:
            # The position of the *hardest* moment is the one worth reporting.
            episode.peak_force_n = normal
            episode.position = position

    def _close_stale(self, tick: int) -> None:
        for key, episode in list(self._open.items()):
            if tick - episode.last_tick > self.gap_ticks:
                self.events.append(self._finish(episode))
                del self._open[key]

    def flush(self) -> list[HitEvent]:
        """Close every open episode. Call at the end of a round, or nothing in flight is scored."""
        for key, episode in list(self._open.items()):
            self.events.append(self._finish(episode))
            del self._open[key]
        return self.events

    @staticmethod
    def _finish(episode: _Episode) -> HitEvent:
        return HitEvent(
            attacker=episode.attacker,
            defender=episode.defender,
            attacker_body=episode.attacker_body,
            defender_body=episode.defender_body,
            region=region_of(episode.defender_body),
            start_tick=episode.start_tick,
            end_tick=episode.last_tick,
            peak_force_n=episode.peak_force_n,
            impulse_ns=episode.impulse_ns,
            position=tuple(float(v) for v in episode.position),
        )


# -- traces ------------------------------------------------------------------------------------------
#: Torso height and uprightness when a fighter is standing in its default stance, measured on the
#: arena. Thresholds below are fractions of these rather than absolute metres, so they survive a
#: change of robot.
STANDING_TORSO_HEIGHT_M = 0.847
STANDING_TORSO_UPRIGHT = 1.0

#: A fighter is down when its torso drops below this fraction of standing height. Measured: a
#: collapsed G1's torso sits at 0.058 m, 7% of standing, so half leaves a wide margin either side of
#: the ambiguous middle — a deep crouch is not a knockdown, and a fighter on the canvas is.
DOWN_HEIGHT_FRACTION = 0.5

#: ...or when its torso tilts more than this far from vertical, as a cosine. 0.5 is 60 degrees. A
#: boxer slipping leans hard (the pose library's slip reaches ~20 degrees) but does not approach
#: this; a fallen G1 measures 0.05.
DOWN_UPRIGHT_COSINE = 0.5


@dataclass
class FightTrace:
    """Per-tick state a match needs but contacts do not carry.

    Records what was asked for and nothing derived: where each fighter is, how far apart they are,
    and each torso's **height and orientation** — height because a fighter on the canvas is the
    definition of a knockdown, orientation because a fighter can be face-down without its torso being
    low. :meth:`is_down` reads both; the *decision* about counts and knockouts is the match loop's.

    Ring control and aggression are defined on where the fighters are, so the trace is recorded
    whether or not anything lands.
    """

    tick: list[int] = field(default_factory=list)
    separation_m: list[float] = field(default_factory=list)
    positions: dict[str, list[np.ndarray]] = field(default_factory=dict)
    centre_distance_m: dict[str, list[float]] = field(default_factory=dict)
    #: Torso height above the canvas, metres.
    torso_height_m: dict[str, list[float]] = field(default_factory=dict)
    #: Cosine of the torso's tilt from vertical: 1.0 upright, 0.0 horizontal.
    torso_upright: dict[str, list[float]] = field(default_factory=dict)
    #: Torso orientation as a quaternion, ``wxyz``, kept in full so a replay can reconstruct it.
    torso_quat: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def observe(self, model, data, tick: int) -> None:
        import mujoco

        roots, torsos = {}, {}
        for fighter in FIGHTERS:
            pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{fighter}_pelvis")
            torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{fighter}_torso_link")
            if pelvis < 0 or torso < 0:
                raise ContactError(f"{fighter}: pelvis or torso_link is not in the model")
            roots[fighter] = data.xpos[pelvis].copy()
            torsos[fighter] = (data.xpos[torso].copy(), data.xmat[torso].reshape(3, 3).copy())

        self.tick.append(tick)
        pair = list(roots.values())
        self.separation_m.append(float(np.linalg.norm(pair[0][:2] - pair[1][:2])))

        for fighter, position in roots.items():
            self.positions.setdefault(fighter, []).append(position)
            self.centre_distance_m.setdefault(fighter, []).append(
                float(np.linalg.norm(position[:2]))
            )

        for fighter, (position, rotation) in torsos.items():
            self.torso_height_m.setdefault(fighter, []).append(float(position[2]))
            # The torso's own up-axis against the world's. A cosine, so it needs no unwrapping.
            self.torso_upright.setdefault(fighter, []).append(float(rotation[2, 2]))
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rotation.flatten())
            self.torso_quat.setdefault(fighter, []).append(quat)

    # -- reading it back ---------------------------------------------------------------------------
    def is_down(self, fighter: str, index: int) -> bool:
        """Whether a fighter was down at trace index ``index``.

        Down on *either* count: too low, or tilted too far. Both are needed — a fighter can be folded
        over with its torso still high, or flat on its back with the torso barely below stance.
        """
        if fighter not in self.torso_height_m:
            raise ContactError(f"no trace for {fighter!r}; recorded {sorted(self.torso_height_m)}")
        height = self.torso_height_m[fighter][index]
        upright = self.torso_upright[fighter][index]
        return (
            height < DOWN_HEIGHT_FRACTION * STANDING_TORSO_HEIGHT_M
            or upright < DOWN_UPRIGHT_COSINE
        )

    def down_ticks(self, fighter: str) -> list[int]:
        """Every tick at which a fighter was down. The match loop turns runs of these into counts."""
        return [
            self.tick[i] for i in range(len(self.tick)) if self.is_down(fighter, i)
        ]

    def summary(self) -> dict[str, float]:
        if not self.tick:
            return {"ticks": 0.0}
        out: dict[str, float] = {
            "ticks": float(len(self.tick)),
            "min_separation_m": float(np.min(self.separation_m)),
            "mean_separation_m": float(np.mean(self.separation_m)),
        }
        for fighter, distances in self.centre_distance_m.items():
            out[f"{fighter}_mean_centre_distance_m"] = float(np.mean(distances))
        for fighter, heights in self.torso_height_m.items():
            out[f"{fighter}_min_torso_height_m"] = float(np.min(heights))
            out[f"{fighter}_min_upright"] = float(np.min(self.torso_upright[fighter]))
            out[f"{fighter}_down_ticks"] = float(len(self.down_ticks(fighter)))
        return out
