"""One frame through the fitted model, in screen fractions (ARCH-01 stage D).

Moved from ``gf_record`` so the live session and the recorder run the SAME
prediction code without the live session importing the recorder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gazelink_core.calibration import schema as S
from gazelink_core.domain import common as C
from gazelink_core.gaze import head_features as H


def predict_one(
    model: Any, features: np.ndarray, head: np.ndarray | None, rig: C.RigGeometry
) -> tuple[float, float] | None:
    """One frame through one fitted model, in screen fractions.

    Module level rather than a method because the live view runs the same
    prediction with no protocol, no targets and no recording around it. Two
    copies of this would be two ways for the dot on screen to disagree with
    the numbers scored afterwards.
    """

    head_names = model.schema.head_names
    if head_names:
        if head is None:
            return None
        design = S.assemble(features.reshape(1, -1), head.reshape(1, -1), head_names, H.HEAD6_NAMES)
    else:
        design = features.reshape(1, -1)
    point = model.predict_norm(design, rig)[0]
    if not np.all(np.isfinite(point)):
        return None
    return float(point[0]), float(point[1])


def predict_overlay_point(
    model: Any,
    model_y: Any,
    features: np.ndarray | None,
    head: np.ndarray | None,
    rig: C.RigGeometry,
) -> tuple[float, float] | None:
    """Run this frame through the loaded overlay model, if any.

    With a second model supplied for the vertical axis, x comes from the
    primary model and y from that one. The two feature-scaling families are
    each accurate on a different axis -- measured offline, ``zscore`` gives x
    median 72.9px with the vertical compressed to slope 0.717, while ``none``
    reaches y slope 0.850 with x median 277.7px -- so a model per axis takes
    the better half of each. Both run on the same frame, so the pair cannot
    drift apart in time.

    Never raises into the camera thread: a bad frame for the overlay is a
    missing dot, not a crashed recording.
    """

    if model is None or features is None:
        return None
    try:
        point = predict_one(model, features, head, rig)
        if point is None:
            return None
        if model_y is None:
            return point
        vertical = predict_one(model_y, features, head, rig)
        if vertical is None:
            # Half a point is not a point: showing x with a stale or absent y
            # would put the dot somewhere neither model claims.
            return None
        return point[0], vertical[1]
    except Exception:  # noqa: BLE001 - a visual aid must never break recording
        return None
