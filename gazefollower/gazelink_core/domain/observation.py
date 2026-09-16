"""What one camera frame told us, with nothing of the library attached.

``FrameObservation`` is the only thing that crosses from a tracking source into
the core (TECHNICAL_SPEC 4.3, 5). The library's ``face_info``/``gaze_info``
objects are read in exactly one place -- ``tracking.gazefollower_source.observe``
-- and never leave it.

Units and frames:
* ``observed_s``: monotonic seconds from the session's injected clock, taken
  when the frame reached the process. Control-time decisions use this.
* ``timestamp_ns``: the library's own capture stamp, 0 when it gave none
  (``timestamp_available`` says which). Stored in recordings; never compared
  with ``observed_s``.
* ``openness_image``: (left, right) eye polygon area in px^2 in the LIBRARY's
  frame -- index 0 is the eye on the left of the IMAGE, the person's right.
  Anything that means the person's eyes goes through
  ``interaction.gesture.eyes_as_the_person_has_them``. A missing value is 0.0,
  exactly as every existing recording stores it; ``openness_available`` says
  whether the library actually reported both.
* ``features``: the model embedding exactly as the library produced it (dtype
  preserved), present only when ``gaze_status`` is true. A read-only copy: no
  consumer can change what another consumer sees. Each consumer casts it under
  its own explicit policy (``gaze.sample_gate``).
* ``head6``: ``head_features.HEAD6_NAMES`` vector or None, computed under the
  source's head policy (see :class:`HeadPolicy`). None is "not trustworthy or
  not computed", never zeros.
* ``raw_gaze_cm``: the library's own uncalibrated estimate in its label
  centimetres, diagnostic only; None when absent or malformed.

Confidence: the library exposes no probability, so none is invented. Validity
is the explicit gate in ``gaze.sample_gate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class HeadPolicy(StrEnum):
    """When the (not free) head-pose builder runs for a frame.

    GAZE: only when the gaze is usable -- what the recorder has always done.
    GAZE_OR_FACE: also for a face with no usable gaze -- what the live view
    needs, because the pose while an eye is SHUT is the thing it reports.
    """

    GAZE = "gaze"
    GAZE_OR_FACE = "gaze-or-face"


class Reason(StrEnum):
    NO_FACE = "no-face"
    NO_GAZE = "no-gaze"
    NO_FEATURES = "no-features"
    OPENNESS_MISSING = "openness-missing"
    HEAD_UNAVAILABLE = "head-unavailable"
    RAW_GAZE_INVALID = "raw-gaze-invalid"
    NO_RESULT = "no-result"


def _frozen(array: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any] | None:
    if array is None:
        return None
    copy = np.array(array, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class FrameObservation:
    frame_seq: int
    observed_s: float
    timestamp_ns: int
    timestamp_available: bool
    face_present: bool
    gaze_status: bool
    features: np.ndarray[Any, Any] | None
    openness_image: tuple[float, float]
    openness_available: bool
    head6: np.ndarray[Any, Any] | None
    pnp_deg: tuple[float, float, float] | None
    raw_gaze_cm: tuple[float, float] | None
    # The library's tracking state: its enum NAME when it has one, and its
    # str() form (``"None"`` when absent). The recorder renders these two ways
    # for its primary and shadow rows; both are kept so neither changes.
    tracking_state_name: str | None
    tracking_state_text: str
    tracking_state_present: bool
    # True when the source had a result object at all (a dual-capture shadow
    # frame can be missing entirely).
    result_present: bool = True
    reasons: frozenset[Reason] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _frozen(self.features))
        object.__setattr__(self, "head6", _frozen(self.head6))

    @property
    def tracking_label(self) -> str:
        """The primary recording's rendering: name, else str(), else UNKNOWN."""

        if self.tracking_state_name:
            return self.tracking_state_name
        return self.tracking_state_text if self.tracking_state_present else "UNKNOWN"
