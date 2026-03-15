# Axis bounds and tick mark helpers for matplotlib plots.
#
# Two functions for different use cases:
#
#   get_axis_bounds_and_ticks(data, padding)
#       For linear-scale axes. Returns nice round tick marks.
#       Usage:
#           bounds, ticks = get_axis_bounds_and_ticks([min_y, max_y], padding=0.05)
#           ax.set_ylim(bounds)
#           ax.set_yticks(ticks)
#
#   get_axis_bounds_and_ticks_ln_pct(data, padding)
#       For axes where the data is in log-ratio space (i.e. ln(value/baseline))
#       but labels should display as percentage change. Returns tick positions
#       in log space and corresponding percentage labels.
#       Usage:
#           bounds, ticks, pcts = get_axis_bounds_and_ticks_ln_pct([min_y, max_y], padding=0.05)
#           ax.set_ylim(bounds)
#           ax.set_yticks(ticks)
#           ax.set_yticklabels([f'{p:g}%' for p in pcts])

import numpy as np
from math import log10, floor

from src.log_percent_ticks import get_axis_bounds_and_ticks_ln_pct  # noqa: F401


def get_axis_bounds_and_ticks(data, padding=0.0):
    """Calculate axis bounds and evenly-spaced 'nice' tick positions for linear data.

    Snaps the bounds to round numbers, anchors ticks at zero when the range
    crosses zero, and chooses a tick spacing that yields ~5-8 ticks.

    Args:
        data: iterable of numeric values (often just [min_val, max_val])
        padding: fractional padding added to each side of the data range
            (e.g. 0.05 adds 5% on each side). Padding is not applied to a
            bound that has been snapped to zero.

    Returns:
        bounds: [min, max] suitable for ax.set_ylim / ax.set_xlim
        ticks_vals: array of tick positions within (and slightly beyond) bounds
    """
    if not data:
        return (0, 1)  # default bounds if no data

    min_val = min(data)
    max_val = max(data)

    # Snap bound to zero when data is close to or entirely on one side of zero
    if abs(min_val) < 1e-6 * max_val or (min_val > 0 and max_val > 0 and min_val <= 0.5 * max_val):
        min_val = 0
    if abs(max_val) < 1e-6 * abs(min_val) or (min_val < 0 and max_val < 0 and max_val >= 0.5 * min_val):
        max_val = 0

    range_val = max_val - min_val

    padding_amount = range_val * padding
    if min_val != 0:
        min_val -= padding_amount
    if max_val != 0:
        max_val += padding_amount

    # Express range as mantissa × 10^power (mantissa in [1, 10))
    range_val = max_val - min_val
    range_power = floor(log10(range_val))
    range_mantissa = range_val / 10**range_power

    # Select a tick pattern (normalized to [0, range_scale]) based on mantissa.
    # The pattern's step size (ticks[1]) is scaled by 10^power to get the
    # actual tick spacing.
    if range_mantissa == 1.0:
        ticks = np.arange(0,1.2,0.2)
        range_scale = 1
    elif range_mantissa <= 1.5:
        ticks = np.arange(0,1.8,0.3)
        range_scale = 1.5
    elif range_mantissa <= 2:
        ticks = np.arange(0,2.2,0.4)
        range_scale = 2
    elif range_mantissa <= 3:
        ticks = np.arange(0,3.5,0.5)
        range_scale = 3 
    elif range_mantissa <= 4:
        ticks = np.arange(0,4.5,0.5)
        range_scale = 4
    elif range_mantissa <= 7:
        ticks = np.arange(0,8,1)
        range_scale = -floor(-range_mantissa)  # round up to nearest integer
    elif range_mantissa <= 10:
        ticks = np.arange(0,12,2)
        range_scale = -floor(-range_mantissa) 
    else:
        print("Unexpected range mantissa:", range_mantissa)
        raise ValueError("Unexpected range mantissa: {}".format(range_mantissa))
    
    min_tick = round(min_val / 10**range_power) * 10**range_power
    max_tick = round(max_val / 10**range_power) * 10**range_power

    min_val = min(min_val, min_tick)
    max_val = max(max_val, max_tick)

    if min_val <= 0 and max_val >= 0:
        # 0 is in range, so anchor ticks at zero
        tick_step = ticks[1] * 10**range_power  # spacing from the tick pattern
        pos_ticks = np.arange(0, max_val + tick_step, tick_step)
        neg_ticks = np.arange(-tick_step, min_val - tick_step, -tick_step)
        ticks_vals = np.sort(np.concatenate([neg_ticks, pos_ticks]))

    else:
        # 0 not in range, so add ticks at regular intervals of 10**range_power
        tick_step = ticks[1] * 10**range_power
        ticks_vals = min_tick + ticks * 10**range_power

    # Ensure ticks extend one step beyond data range on each end
    if len(ticks_vals) > 0 and ticks_vals[0] > min_val:
        ticks_vals = np.concatenate([[ticks_vals[0] - tick_step], ticks_vals])
    if len(ticks_vals) > 0 and ticks_vals[-1] < max_val:
        ticks_vals = np.concatenate([ticks_vals, [ticks_vals[-1] + tick_step]])

    # Update bounds to match outermost ticks
    if len(ticks_vals) > 0:
        min_val = min(min_val, ticks_vals[0])
        max_val = max(max_val, ticks_vals[-1])

    return [min_val, max_val], ticks_vals
