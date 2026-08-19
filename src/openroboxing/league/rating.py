"""Glicko-2 ratings (M5-T1).

Mark Glickman's published algorithm, implemented unmodified from *"Example of the Glicko-2 system"*
(glicko.net/glicko/glicko2.pdf). Nothing here is our design, which is the point: a rating system is
only worth having if it is the one people can look up.

The implementation is checked against **Glickman's own worked example** in `tests/test_rating.py` —
a 1500/200/0.06 player against three opponents must produce 1464.06 / 151.52 / 0.05999. That
vector is what makes this trustworthy.

Two scales, and the confusion between them is the classic bug
--------------------------------------------------------------
Glicko-2 works internally on a scale where rating is ``mu`` and deviation is ``phi``, related to the
familiar 1500-ish numbers by a factor of ``GLICKO2_SCALE`` (173.7178). Everything public here is in
the **familiar** scale; the conversion happens at the edges of :meth:`Glicko2.rate` and nowhere else.

Conventions
-----------
- A **rating period** is one week (`spec/season.md`). Results are batched into a period and applied
  together; that is how Glicko-2 is defined, and rating one match at a time is a different (worse)
  system.
- A fighter who plays nobody in a period still has their RD grow, because uncertainty increases with
  time. :meth:`Glicko2.rate` with no results does exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

#: Glickman's defaults, unmodified.
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOLATILITY = 0.06

#: The system constant. Constrains volatility change per period; Glickman recommends 0.3-1.2.
DEFAULT_TAU = 0.5

#: Converts between the public rating scale and Glicko-2's internal one.
GLICKO2_SCALE = 173.7178

#: Convergence tolerance for the volatility iteration, from the paper.
CONVERGENCE_TOLERANCE = 1e-6

#: The paper's algorithm is guaranteed to converge; this only stops a pathological input looping.
MAX_ITERATIONS = 100


class RatingError(RuntimeError):
    """A rating could not be computed. Never recovered from silently."""


@dataclass(frozen=True)
class Rating:
    """One fighter's rating, in the public scale."""

    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD
    volatility: float = DEFAULT_VOLATILITY

    @property
    def interval(self) -> tuple[float, float]:
        """The 95% confidence interval, ``rating +- 2 x RD``."""
        return (self.rating - 2 * self.rd, self.rating + 2 * self.rd)

    @property
    def conservative(self) -> float:
        """The interval's lower bound — what the table is ordered by (`spec/season.md`).

        Two fighters on 1600 are not equal if one has played 8 matches and the other 40. Ordering by
        the bottom of the interval says so.
        """
        return self.rating - 2 * self.rd

    def __str__(self) -> str:
        return f"{self.rating:.0f}+-{2 * self.rd:.0f}"


@dataclass(frozen=True)
class Result:
    """One match, from one fighter's point of view. ``score`` is 1 win, 0.5 draw, 0 loss."""

    opponent: Rating
    score: float

    def __post_init__(self) -> None:
        if self.score not in (0.0, 0.5, 1.0):
            raise RatingError(f"score must be 0, 0.5 or 1, got {self.score}")


class Glicko2:
    """The rating system. Stateless — it maps ``(rating, results) -> rating``."""

    def __init__(self, tau: float = DEFAULT_TAU) -> None:
        if not 0.0 < tau <= 2.0:
            raise RatingError(f"tau {tau} outside a sane range; Glickman recommends 0.3-1.2")
        self.tau = tau

    # -- scale conversion ---------------------------------------------------------------------------
    @staticmethod
    def _to_glicko2(rating: Rating) -> tuple[float, float]:
        return (rating.rating - DEFAULT_RATING) / GLICKO2_SCALE, rating.rd / GLICKO2_SCALE

    @staticmethod
    def _from_glicko2(mu: float, phi: float, volatility: float) -> Rating:
        return Rating(
            rating=mu * GLICKO2_SCALE + DEFAULT_RATING,
            rd=phi * GLICKO2_SCALE,
            volatility=volatility,
        )

    # -- the algorithm -------------------------------------------------------------------------------
    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)

    @staticmethod
    def _e(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-Glicko2._g(phi_j) * (mu - mu_j)))

    def _new_volatility(self, phi: float, sigma: float, variance: float, delta: float) -> float:
        """Step 5 of the paper: the Illinois-algorithm root find for the new volatility."""
        a = math.log(sigma**2)
        tau_sq = self.tau**2

        def f(x: float) -> float:
            exp_x = math.exp(x)
            numerator = exp_x * (delta**2 - phi**2 - variance - exp_x)
            denominator = 2.0 * (phi**2 + variance + exp_x) ** 2
            return numerator / denominator - (x - a) / tau_sq

        big_a = a
        if delta**2 > phi**2 + variance:
            big_b = math.log(delta**2 - phi**2 - variance)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
                if k > MAX_ITERATIONS:
                    raise RatingError("volatility bracket did not converge")
            big_b = a - k * self.tau

        f_a, f_b = f(big_a), f(big_b)
        iterations = 0
        while abs(big_b - big_a) > CONVERGENCE_TOLERANCE:
            iterations += 1
            if iterations > MAX_ITERATIONS:
                raise RatingError("volatility iteration did not converge")
            big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
            f_c = f(big_c)
            if f_c * f_b <= 0:
                big_a, f_a = big_b, f_b
            else:
                f_a /= 2.0
            big_b, f_b = big_c, f_c

        return math.exp(big_a / 2.0)

    def rate(self, rating: Rating, results: Sequence[Result]) -> Rating:
        """A fighter's new rating after one rating period.

        With no results the rating is unchanged and the RD **grows**: not playing makes the system
        less sure about you, which is step 6 of the paper and the reason a dormant fighter drifts
        back towards the middle of the table rather than sitting on an old number.
        """
        mu, phi = self._to_glicko2(rating)
        sigma = rating.volatility

        if not results:
            return self._from_glicko2(mu, math.sqrt(phi**2 + sigma**2), sigma)

        variance_inverse = 0.0
        delta_sum = 0.0
        for result in results:
            mu_j, phi_j = self._to_glicko2(result.opponent)
            g = self._g(phi_j)
            e = self._e(mu, mu_j, phi_j)
            variance_inverse += g**2 * e * (1.0 - e)
            delta_sum += g * (result.score - e)

        if variance_inverse <= 0.0:
            raise RatingError("estimated variance is not positive; opponents may be malformed")
        variance = 1.0 / variance_inverse
        delta = variance * delta_sum

        sigma_new = self._new_volatility(phi, sigma, variance, delta)
        phi_star = math.sqrt(phi**2 + sigma_new**2)
        phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + variance_inverse)
        mu_new = mu + phi_new**2 * delta_sum

        return self._from_glicko2(mu_new, phi_new, sigma_new)


def win_probability(a: Rating, b: Rating) -> float:
    """Chance that ``a`` beats ``b``, accounting for both fighters' uncertainty.

    Used to simulate a season (`tools/simulate_season.py`) and to sanity-check a table. Not used to
    rate anybody — that is :meth:`Glicko2.rate`'s job.
    """
    _, phi_a = Glicko2._to_glicko2(a)
    mu_b, phi_b = Glicko2._to_glicko2(b)
    mu_a, _ = Glicko2._to_glicko2(a)
    combined = math.sqrt(phi_a**2 + phi_b**2)
    return 1.0 / (1.0 + math.exp(-Glicko2._g(combined) * (mu_a - mu_b)))
