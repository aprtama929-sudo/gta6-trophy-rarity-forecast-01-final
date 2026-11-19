"""Leonida Week-One Forecaster: GTA VI trophy earn-rate quantiles for make.fun.

Reads one case file (makefun/gta6-trophies-case/v1) from the input directory
and, for every trophy in it, writes five quantiles (5/25/50/75/95) of the
trophy's Exophase earn rate at the case target.  The score is a pinball loss
on the logit scale, so all modelling happens on logit(rate) and only the
final quantiles are mapped back to percent.

Method, per trophy
------------------
Five simple median models are fitted on the trophy's observed history and
every one of them is anchored at the last observed rate, so a model only
ever contributes a *move* away from that anchor:

  0  persistence        -- no move at all;
  1  log-time trend     -- least-squares line of logit(rate) against
                           ln(1 + hours since 2026-11-19T00:00Z);
  2  damped log-time    -- the same slope, its move shrunk by 0.6;
  3  hours trend        -- least-squares line of logit(rate) on raw hours;
  4  saturation         -- exponential approach to a ceiling in rate space:
                           the ceiling is picked from a 25-point grid by
                           least squares on ln(ceiling - rate) vs time, and
                           the remaining gap decays from the last rate.

An internal backtest refits the same five models on four prefixes of the
history (span cuts at 40/55/70/85 percent), scores each prefix fit against
the final observed logit, and normalises each error per square-root day.
Blend weights are proportional to 1 / (mean error + 0.15) ** 2, any single
model's move is capped at +-3 logits, and the blended move is shrunk by
0.85 toward persistence.

The spread combines three pieces: the trophy's own backtest scatter per
square-root day (shrunk 60/40 toward the pooled cross-trophy scatter, then
80/20 toward the plain 0.5 random-walk scale), combined in quadrature with
half the weighted disagreement between the five models' moves, scaled by
the square root of the horizon in days and clipped to [0.30, 2.5] logits.
A history spanning under 12 hours gets a 1.0-logit floor, because a refresh
of the tracked-player pool can move every rate at once and a short flat
history says nothing about that.

Fallbacks: a trophy with no observation at all borrows the cross-trophy
median last rate with a wide 1.5-logit band; any failure inside a case
falls back to the plain persistence forecast for that case.

Standard library only, fully deterministic: no randomness, no clock reads,
no network.  Run as ``python3 -P -s main.py`` with MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR set outside the grader (defaults /input and /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Under the grader's -P flag this file's directory is not on sys.path; put
# it back first so a helper module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

EXPECTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_FOR_LEVEL = [-1.645, -0.674, 0.0, 0.674, 1.645]

# All trend clocks are measured in hours from the start of release day.
RELEASE_DAY = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

RATE_LO = 0.01    # the scorer clamps percents into [0.01, 99.99] pre-logit
RATE_HI = 99.99
MOVE_LIMIT = 3.0  # no single model may move the median more than this
BACKTEST_CUTS = (0.40, 0.55, 0.70, 0.85)
SHORT_SPAN_HOURS = 12.0
SHORT_SPAN_FLOOR = 1.0
EMPTY_TROPHY_SPREAD = 1.5
SPREAD_LO = 0.30
SPREAD_HI = 2.5


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def parse_instant(text):
    """Epoch seconds for an ISO-8601 UTC timestamp from a case file."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def hours_since_release(epoch_seconds):
    return (epoch_seconds - RELEASE_DAY) / 3600.0


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, float(rate)))


def logit_of(rate_percent):
    r = clamp_rate(rate_percent)
    return math.log(r / (100.0 - r))


def rate_of(logit_value):
    # Branch on the sign so exp() never overflows for extreme logits.
    if logit_value >= 0.0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(value):
    """Round half up to 4 decimals; whole numbers come back as int."""
    units = math.floor(value * 10000.0 + 0.5)
    out = units / 10000.0
    return int(out) if float(out).is_integer() else out


def log_time(hours):
    """ln(1 + hours) with a guard for reads just before release midnight."""
    return math.log(max(1.0 + hours, 1e-9))


def least_squares(xs, ys):
    """Return (intercept, slope, sse) of the least-squares line y ~ x."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        sxx += dx * dx
        sxy += dx * (y - mean_y)
    slope = sxy / sxx if sxx > 0.0 else 0.0
    intercept = mean_y - slope * mean_x
    sse = 0.0
    for x, y in zip(xs, ys):
        resid = y - (intercept + slope * x)
        sse += resid * resid
    return intercept, slope, sse


# --------------------------------------------------------------------------
# One prefix fit: parameters for all five models, fitted exactly once
# --------------------------------------------------------------------------

class PrefixFit:
    """The five models fitted on one prefix of one trophy's history.

    ``points`` is a non-empty list of (hours, rate) oldest first.  Only the
    slopes and the saturation pair are stored; every prediction re-anchors
    at the prefix's last point, so scoring a horizon is O(1) per model.
    """

    __slots__ = ("anchor_hours", "anchor_rate", "anchor_logit",
                 "slope_log_time", "slope_hours", "ceiling", "sat_slope")

    def __init__(self, points):
        self.anchor_hours, self.anchor_rate = points[-1]
        self.anchor_logit = logit_of(self.anchor_rate)
        hours = [p[0] for p in points]
        logits = [logit_of(p[1]) for p in points]
        _, self.slope_log_time, _ = least_squares(
            [log_time(h) for h in hours], logits)
        _, self.slope_hours, _ = least_squares(hours, logits)
        rates = [p[1] for p in points]
        self.ceiling, self.sat_slope = self._fit_saturation(hours, rates)

    @staticmethod
    def _fit_saturation(hours, rates):
        """Pick (ceiling, decay slope) or (None, 0.0) when it cannot apply.

        Ceilings form a 25-point geometric grid of gaps above the highest
        rate seen; for each candidate the log-gap ln(C - rate) is fitted
        against time and the candidate with the least squared error wins.
        The grid search scores every candidate from running sums in one
        pass over the points, which keeps a 358-read case fast: the fit is
        rebuilt for each backtest prefix, so a three-pass least squares
        per candidate would dominate the whole run.
        """
        top = max(rates)
        if top >= 100.0:
            return None, 0.0
        n = len(rates)
        mean_h = sum(hours) / n
        sxx = sum((h - mean_h) ** 2 for h in hours)
        room = 100.0 - top
        gap_lo = max(0.02, 0.005 * room)
        best = None
        for k in range(25):
            gap = gap_lo * (room / gap_lo) ** (k / 24.0) if room > gap_lo \
                else room
            ceiling = min(100.0, top + gap)
            if ceiling <= top:
                continue
            sum_y = 0.0
            sum_xy = 0.0
            sum_yy = 0.0
            for h, r in zip(hours, rates):
                y = math.log(ceiling - r)
                sum_y += y
                sum_xy += h * y
                sum_yy += y * y
            mean_y = sum_y / n
            sxy = sum_xy - n * mean_h * mean_y
            syy = sum_yy - n * mean_y * mean_y
            slope = sxy / sxx if sxx > 0.0 else 0.0
            sse = syy - slope * sxy
            if best is None or sse < best[0]:
                best = (sse, ceiling, slope)
        if best is None:
            return None, 0.0
        return best[1], best[2]

    def predict_logit(self, model, target_hours):
        """Model ``model``'s median logit at ``target_hours``."""
        if model == 0:
            return self.anchor_logit
        if model == 1:
            step = log_time(target_hours) - log_time(self.anchor_hours)
            return self.anchor_logit + self.slope_log_time * step
        if model == 2:
            step = log_time(target_hours) - log_time(self.anchor_hours)
            return self.anchor_logit + 0.6 * self.slope_log_time * step
        if model == 3:
            return self.anchor_logit \
                + self.slope_hours * (target_hours - self.anchor_hours)
        # Model 4: saturation.  The gap to the ceiling decays (or, for a
        # falling trophy, grows) exponentially from the last observed rate.
        if self.ceiling is None:
            return self.anchor_logit
        exponent = self.sat_slope * (target_hours - self.anchor_hours)
        exponent = min(50.0, max(-50.0, exponent))
        gap_then = (self.ceiling - self.anchor_rate) * math.exp(exponent)
        return logit_of(self.ceiling - gap_then)


MODEL_COUNT = 5


# --------------------------------------------------------------------------
# Per-trophy blending
# --------------------------------------------------------------------------

class TrophyPlan:
    """Everything one trophy's forecast needs, computed in a first pass."""

    __slots__ = ("median_logit", "disagreement", "backtest_errors",
                 "span_hours")

    def __init__(self, median_logit, disagreement, backtest_errors,
                 span_hours):
        self.median_logit = median_logit
        self.disagreement = disagreement
        self.backtest_errors = backtest_errors
        self.span_hours = span_hours


def plan_trophy(points, target_hours):
    """Blend the five models for one trophy with a non-empty history."""
    full_fit = PrefixFit(points)
    first_hours = points[0][0]
    last_hours, last_rate = points[-1]
    last_logit = logit_of(last_rate)
    span = last_hours - first_hours

    # Internal backtest: refit on each span cut, predict the final logit.
    per_model_errors = [[] for _ in range(MODEL_COUNT)]
    every_error = []
    for cut in BACKTEST_CUTS:
        cut_hours = first_hours + cut * span
        prefix = [p for p in points if p[0] <= cut_hours + 1e-9]
        if not prefix:
            continue
        anchor_hours = prefix[-1][0]
        elapsed_days = (last_hours - anchor_hours) / 24.0
        if elapsed_days < 0.01:
            continue
        cut_fit = PrefixFit(prefix)
        scale = math.sqrt(elapsed_days)
        for model in range(MODEL_COUNT):
            guess = cut_fit.predict_logit(model, last_hours)
            err = abs(guess - last_logit) / scale
            per_model_errors[model].append(err)
            every_error.append(err)

    # Weights from mean backtest error; a model the backtest never reached
    # inherits the average error of the ones it did reach.
    mean_errors = []
    for errs in per_model_errors:
        mean_errors.append(sum(errs) / len(errs) if errs else None)
    reached = [e for e in mean_errors if e is not None]
    fill = sum(reached) / len(reached) if reached else 0.0
    weights = []
    for e in mean_errors:
        base = fill if e is None else e
        weights.append(1.0 / (base + 0.15) ** 2)
    weight_total = sum(weights)

    moves = []
    for model in range(MODEL_COUNT):
        raw = full_fit.predict_logit(model, target_hours) - last_logit
        moves.append(min(MOVE_LIMIT, max(-MOVE_LIMIT, raw)))
    blended_raw = sum(w * m for w, m in zip(weights, moves)) / weight_total
    disagreement = math.sqrt(
        sum(w * (m - blended_raw) ** 2 for w, m in zip(weights, moves))
        / weight_total)
    median_logit = last_logit + 0.85 * blended_raw
    return TrophyPlan(median_logit, disagreement, every_error, span)


def spread_for(plan, pooled_sigma, horizon_days):
    """Logit spread for one trophy, per the shrink / quadrature recipe."""
    if plan.backtest_errors:
        own = math.sqrt(
            sum(e * e for e in plan.backtest_errors)
            / len(plan.backtest_errors))
        sigma = 0.6 * own + 0.4 * pooled_sigma
    else:
        sigma = pooled_sigma
    sigma = 0.8 * sigma + 0.2 * 0.5
    base = sigma * math.sqrt(horizon_days)
    spread = math.sqrt(base * base + (0.5 * plan.disagreement) ** 2)
    if plan.span_hours < SHORT_SPAN_HOURS:
        spread = max(spread, SHORT_SPAN_FLOOR)
    return min(SPREAD_HI, max(SPREAD_LO, spread))


def quantile_row(median_logit, spread):
    qs = []
    running = 0.0
    for z in Z_FOR_LEVEL:
        value = round4(rate_of(median_logit + z * spread))
        # Rounding must never break the non-decreasing order.
        value = max(value, running)
        running = value
        qs.append(value)
    return qs


# --------------------------------------------------------------------------
# Case handling
# --------------------------------------------------------------------------

def series_by_trophy(case):
    """Split a case history into per-trophy (hours, rate) point lists."""
    trophies = case["trophies"]
    columns = [[] for _ in trophies]
    for read in case["history"]:
        at_hours = hours_since_release(parse_instant(read["at"]))
        rates = read["rates"]
        for idx in range(len(trophies)):
            if idx < len(rates) and rates[idx] is not None:
                columns[idx].append((at_hours, float(rates[idx])))
    return columns


def last_known_rate(points):
    return points[-1][1] if points else None


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def persistence_row(case, trophy_index, fallback_rate):
    """The house-baseline row: last rate, 0.5-per-root-day random walk."""
    rate = fallback_rate
    for read in reversed(case["history"]):
        rates = read["rates"]
        if trophy_index < len(rates) and rates[trophy_index] is not None:
            rate = float(rates[trophy_index])
            break
    if rate is None:
        rate = 1.0
    horizon_days = case["horizon_minutes"] / 1440.0
    walk = 0.5 * math.sqrt(horizon_days)
    return quantile_row(logit_of(rate), walk)


def forecast_case(case):
    trophies = case["trophies"]
    target_hours = hours_since_release(parse_instant(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    columns = series_by_trophy(case)

    last_rates = [last_known_rate(col) for col in columns]
    known = [r for r in last_rates if r is not None]
    borrow_rate = median_of(known) if known else 1.0

    plans = []
    pooled_pool = []
    for col in columns:
        if col:
            plan = plan_trophy(col, target_hours)
        else:
            plan = None
        plans.append(plan)
        if plan is not None:
            pooled_pool.extend(plan.backtest_errors)
    if pooled_pool:
        pooled_sigma = math.sqrt(
            sum(e * e for e in pooled_pool) / len(pooled_pool))
    else:
        pooled_sigma = 0.5

    rows = []
    for idx, trophy in enumerate(trophies):
        plan = plans[idx]
        if plan is None:
            row = quantile_row(logit_of(borrow_rate), EMPTY_TROPHY_SPREAD)
        else:
            spread = spread_for(plan, pooled_sigma, horizon_days)
            row = quantile_row(plan.median_logit, spread)
        rows.append({"id": trophy["id"], "q": row})
    return {"trophies": rows}


def forecast_case_guarded(case):
    """forecast_case with a whole-case persistence fallback on failure."""
    try:
        return forecast_case(case)
    except Exception:
        rows = []
        for idx, trophy in enumerate(case["trophies"]):
            rows.append({
                "id": trophy["id"],
                "q": persistence_row(case, idx, None),
            })
        return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels", EXPECTED_LEVELS) != EXPECTED_LEVELS:
            raise ValueError(f"{path.name}: unexpected levels")
        result = forecast_case_guarded(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            text, encoding="utf-8")


if __name__ == "__main__":
    main()
