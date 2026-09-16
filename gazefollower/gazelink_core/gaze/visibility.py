"""Hide a point whose source frame is no longer current (ARCH-01 stage D).
"""

from __future__ import annotations

import math
from typing import Any

OVERLAY_STALE_S = 0.25

def visible_overlay_point(
    state: Any, now_s: float, stale_after_s: float = OVERLAY_STALE_S
) -> tuple[float, float] | None:
    """Hide a prediction when drawing continues but camera frames stop."""

    return visible_point(state.overlay_norm, state.overlay_updated_s, now_s, stale_after_s)


def visible_point(
    point: tuple[float, float] | None,
    updated_s: float | None,
    now_s: float,
    stale_after_s: float = OVERLAY_STALE_S,
) -> tuple[float, float] | None:
    """Return a display point only while its source frame is current."""

    if point is None or updated_s is None:
        return None
    age_s = float(now_s) - updated_s
    if not math.isfinite(age_s) or age_s < 0.0 or age_s > stale_after_s:
        return None
    return point
