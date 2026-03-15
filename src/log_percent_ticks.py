# Log-percent tick generation for matplotlib axes.
#
# Core algorithm developed with ChatGPT: picks from a library of "nice"
# percentage candidates and optimizes for even spacing in log space.
#
# Standalone usage:
#     plot_min, plot_max, tick_locs, tick_labels = choose_log_percent_ticks(xmin, xmax)
#
# Drop-in replacement for axes.get_axis_bounds_and_ticks_ln_pct:
#     bounds, ticks_vals, pct_labels = get_axis_bounds_and_ticks_ln_pct(data, padding=0.05)
#     ax.set_ylim(bounds)
#     ax.set_yticks(ticks_vals)
#     ax.set_yticklabels([f'{p:g}%' for p in pct_labels])

import math

import numpy as np


def percent_to_log(p):
    """Convert percent change to natural log ratio."""
    return math.log(1 + p / 100.0)


def log_to_percent(x):
    """Convert natural log ratio to percent change."""
    return 100 * (math.exp(x) - 1)


def format_percent(p):
    """Format percent labels cleanly."""
    if abs(p) < 1e-12:
        return "0%"
    sign = "+" if p > 0 else ""
    if abs(p) >= 1:
        return f"{sign}{round(p):.0f}%"
    else:
        return f"{sign}{p:.2f}%"


def build_percent_candidates():
    """Build a library of 'nice' percent ticks.

    Negative side:
        -90, -98, -99, -99.5
        plus survival powers of 10 ( -99.9, -99.99, ... )

    Positive side:
        1-2-5 sequence (10%, 20%, 50%, 100%, 200%, 500%, ...)
    """
    perc = set()

    # basic negative ticks
    perc.update([-90, -98, -99, -99.5])

    # survival powers of 10: -99.9, -99.99, -99.999, ...
    for k in range(1, 8):
        s = 10 ** (-k)
        p = -100 * (1 - s)
        perc.add(p)

    # small negative values
    perc.update([-50, -30, -20, -10, -5, -2, -1])

    # zero
    perc.add(0)

    # positive 1-2-5 sequence
    for scale in [1, 10, 100, 1000, 10000]:
        for base in [1, 2, 5]:
            perc.add(base * scale)

    # keep values > -100
    perc = [p for p in perc if p > -100]

    return sorted(perc)


def _variance(vals):
    """Population variance of a list of numbers."""
    if not vals:
        return 0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def _thin_ticks(ticks, plot_min, plot_max, target_n):
    """Reduce tick count by snapping to an evenly-spaced grid."""
    # Always keep 0% if present
    chosen = {}
    for p, x in ticks:
        if abs(x) < 1e-12:
            chosen[x] = p
            break

    grid = [
        plot_min + i * (plot_max - plot_min) / (target_n - 1)
        for i in range(target_n)
    ]

    for g in grid:
        p, x = min(ticks, key=lambda t: abs(t[1] - g))
        chosen[x] = p

    return sorted(((p, x) for x, p in chosen.items()), key=lambda t: t[1])


def choose_log_percent_ticks(xmin, xmax, min_ticks=5, max_ticks=8):
    """Choose nicely-spaced tick marks for a log-percent axis.

    Given axis bounds in log-ratio space, selects tick positions from a library
    of "nice" percentage values, optimizing for even spacing.

    Args:
        xmin: minimum value in log-ratio space
        xmax: maximum value in log-ratio space
        min_ticks: minimum number of ticks to consider
        max_ticks: maximum number of ticks to consider

    Returns:
        plot_min: adjusted minimum bound in log space
        plot_max: adjusted maximum bound in log space
        tick_locations: list of tick positions in log space
        tick_labels: list of formatted percentage strings
    """
    if xmax < xmin:
        xmin, xmax = xmax, xmin

    # ensure zero included if one-sided
    if xmin > 0:
        xmin = 0
    if xmax < 0:
        xmax = 0

    perc_candidates = build_percent_candidates()

    candidates = [(p, percent_to_log(p)) for p in perc_candidates]
    candidates.sort(key=lambda t: t[1])

    best = None
    best_score = None

    for N in range(min_ticks, max_ticks + 1):

        avg_spacing = (xmax - xmin) / (N - 1)
        threshold = avg_spacing / 3

        lower = max((p, x) for p, x in candidates if x <= xmin)
        upper = min((p, x) for p, x in candidates if x >= xmax)

        plot_min = lower[1] if (xmin - lower[1]) > threshold else xmin
        plot_max = upper[1] if (upper[1] - xmax) > threshold else xmax

        ticks = [(p, x) for p, x in candidates if plot_min <= x <= plot_max]

        if plot_min <= 0 <= plot_max and not any(abs(x) < 1e-12 for _, x in ticks):
            ticks.append((0, 0))

        ticks.sort(key=lambda t: t[1])

        if len(ticks) > max_ticks:
            ticks = _thin_ticks(ticks, plot_min, plot_max, N)

        if not (min_ticks <= len(ticks) <= max_ticks):
            continue

        xs = [x for _, x in ticks]
        gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1)]

        mean_gap = sum(gaps) / len(gaps)
        score = _variance(gaps) / (mean_gap**2 + 1e-12)

        score += 0.2 * abs(len(ticks) - 6.5)

        if best_score is None or score < best_score:
            best_score = score
            best = (plot_min, plot_max, ticks)

    plot_min, plot_max, ticks = best

    tick_locations = [x for _, x in ticks]
    tick_labels = [format_percent(p) for p, _ in ticks]

    return plot_min, plot_max, tick_locations, tick_labels


def get_axis_bounds_and_ticks_ln_pct(data, padding=0.0):
    """Calculate axis bounds and ticks for log-ratio data with percentage-change labels.

    Drop-in replacement for axes.get_axis_bounds_and_ticks_ln_pct.

    Args:
        data: iterable of values in log-ratio space, i.e. ln(value/baseline).
            Often just [min_val, max_val].
        padding: fractional padding added to each side of the data range

    Returns:
        bounds: [min, max] in log space, suitable for ax.set_ylim
        ticks_vals: numpy array of tick positions in log space
        pct_labels: list of numeric percentage values at each tick
            (e.g. [-25, 0, 25]). Use with
            ax.set_yticklabels([f'{p:g}%' for p in pct_labels]).
    """
    finite_data = [x for x in data if np.isfinite(x)]
    xmin = min(finite_data) if finite_data else -0.1
    xmax = max(finite_data) if finite_data else 0.1
    span = xmax - xmin
    xmin -= span * padding
    xmax += span * padding

    plot_min, plot_max, tick_locations, _ = choose_log_percent_ticks(xmin, xmax)

    # Add a small margin so outermost ticks aren't clipped by matplotlib
    margin = (plot_max - plot_min) * 0.02
    bounds = [plot_min - margin, plot_max + margin]
    ticks_vals = np.array(tick_locations)
    pct_labels = [log_to_percent(x) for x in tick_locations]

    return bounds, ticks_vals, pct_labels
