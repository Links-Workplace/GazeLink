"""Is this model usable in the conditions the camera sees now? (ARCH-01 stage D).

A diagnostic before a session, not a control-time safety gate
(TECHNICAL_SPEC 4.2). The recorder and the live session each apply it
with their existing, different policies; see the callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from gazelink_core.calibration import schema as S
from gazelink_core.gaze import head_features as H

# A calibrated model only answers meaningfully for inputs near the conditions
# it was trained on. Outside them the SVR returns its constant bias, so the
# overlay point stops following the eye and freezes -- with the camera, the
# face detector and every other check still reporting success. That failure
# cost a full afternoon of recordings before it was identified, so the model
# is now checked against live frames before a protocol is spent on it.
# Measured: healthy sessions sit at 0.86-0.98 median activation, and every
# session whose point had frozen sat at 0.000. The threshold is deliberately
# far below the healthy band -- this refuses only the unmistakable case.
PREFLIGHT_MIN_ACTIVATION = 0.20
PREFLIGHT_MAX_FRAMES = 240

def design_row(
    model: Any,
    features: Any,
    head: Callable[[], np.ndarray | None],
) -> np.ndarray | None:
    """One model-ready row from a live frame, or None if this frame cannot make one.

    ``head`` is asked for only when the model has head columns (building the
    pose is not free). A model fitted with head columns needs those columns
    here too: feeding it the bare feature vector raises on the column count,
    and doing that during warm-up would abort the session for a configuration
    the recorder fully supports. Never raises -- a frame the check cannot use
    is one fewer sample, not a failed recording.
    """

    try:
        if features is None:
            return None
        features = np.asarray(features, dtype=np.float64).reshape(1, -1)
        head_names = model.schema.head_names
        if not head_names:
            return features.reshape(-1)
        head = head()
        if head is None:
            return None
        design = S.assemble(features, head.reshape(1, -1), head_names, H.HEAD6_NAMES)
        return np.asarray(design, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001 - a preflight sample must never break recording
        return None


def preflight_verdict(
    activation: np.ndarray | None,
    *,
    minimum: float = PREFLIGHT_MIN_ACTIVATION,
) -> tuple[bool, str]:
    """Is this model usable in the conditions the camera is seeing right now?

    Returns (ok, message). Answers ok for anything it cannot measure -- a
    ridge model, no frames yet -- because refusing on absent evidence would
    block recording for no reason. Only a measured collapse fails.
    """

    if activation is None:
        return True, "model has no kernel to check (not an SVR); skipping"
    finite = activation[np.isfinite(activation)]
    if finite.size == 0:
        return True, "no valid frames to check the model against; skipping"
    median = float(np.median(finite))
    frozen = float(np.mean(finite < 0.01))
    if median >= minimum:
        return True, f"model matches current conditions (activation {median:.3f})"
    return False, (
        f"this model does not fit the current conditions: median support "
        f"activation {median:.3f} (healthy is above {minimum:.2f}), and "
        f"{100.0 * frozen:.0f}% of frames carry no calibration information "
        "at all.\nThe overlay point would freeze on a fixed wrong position "
        "rather than follow your eye.\nRecalibrate (run a round with "
        "protocol A) and fit a model from it, or pass --skip-model-check to "
        "record anyway."
    )


