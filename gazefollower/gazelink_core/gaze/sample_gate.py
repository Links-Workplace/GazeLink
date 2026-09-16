"""Which frames are gaze samples, and how a point goes through the filter.

One implementation of the rules the live view and the recorder each wrote
inline (ARCH-01 stage E). Their existing differences are kept as explicit
policies, not unified (TECHNICAL_SPEC 15):

=================  ===========  ==========================  ======================
policy             features as  gaze sample requires        point timestamp
=================  ===========  ==========================  ======================
``LIVE``           float64      gaze + both eyes open       every frame
``RECORD_OVERLAY`` float32      gaze + both eyes open +     only frames with a
                                a target phase (not idle)   point
=================  ===========  ==========================  ======================

Unifying them changes results and is a separate behaviour task.

"Both eyes open" is the absolute ``BLINK_THRESHOLD`` gate on the raw polygon
area, the one every measurement in this project was made against. It is NOT
the per-eye baseline gate the gesture layer uses: a frame that is no gaze
sample can still carry the closure a gesture needs (TECHNICAL_SPEC 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gazelink_core.domain import common as C
from gazelink_core.domain.observation import FrameObservation


@dataclass(frozen=True)
class SamplePolicy:
    name: str
    feature_dtype: Any
    requires_active_target: bool
    stamp_every_frame: bool


LIVE = SamplePolicy("live", np.float64, requires_active_target=False, stamp_every_frame=True)
RECORD_OVERLAY = SamplePolicy(
    "record-overlay", np.float32, requires_active_target=True, stamp_every_frame=False
)


def both_eyes_open(openness: tuple[float, float], threshold: float = C.BLINK_THRESHOLD) -> bool:
    left, right = openness
    return left > threshold and right > threshold


def is_gaze_sample(
    obs: FrameObservation, policy: SamplePolicy, *, target_active: bool = True
) -> bool:
    if policy.requires_active_target and not target_active:
        return False
    return obs.gaze_status and both_eyes_open(obs.openness_image)


def features_for(obs: FrameObservation, policy: SamplePolicy) -> np.ndarray[Any, Any] | None:
    if obs.features is None:
        return None
    return np.asarray(obs.features, dtype=policy.feature_dtype).reshape(-1)


def filter_point(
    point_filter: Any, raw: tuple[float, float] | None, now_s: float
) -> tuple[float, float] | None:
    """The filtered point for this frame, or None.

    No raw point resets the filter: a blink, occlusion, tracking loss or a bad
    model result must not pull the next valid point toward stale history. A
    filter that rejects its input (``ValueError``) is reset and yields nothing.
    """

    if raw is None:
        if point_filter is not None:
            point_filter.reset()
        return None
    if point_filter is None:
        return raw
    try:
        point: tuple[float, float] = point_filter.update(raw, now_s)
        return point
    except ValueError:
        point_filter.reset()
        return None
