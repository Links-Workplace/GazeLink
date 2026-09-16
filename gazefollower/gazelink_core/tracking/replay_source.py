"""A recorded session played back as a TrackingSource (ARCH-01 stage E).

Publishes :class:`FrameObservation` built from a ``calibration.schema.Recording``
so the core pipeline, and a whole live session, can run with no camera. It
opens nothing and emits no input.

What a recording can and cannot give back:
* features (float32, as stored), library-frame openness, gaze status, head6
  when it was valid, the library's raw estimate, tracking state and the
  library's timestamp are reproduced exactly (``openness_available`` is always True: the recorder
  stored 0.0 for a missing value and cannot say which zeros were missing);
* face presence is not stored. It is taken as "gaze usable, or either eye was
  reported open";
* idle rows were stored without an embedding (NaN). They are published with
  NO features (reason ``no-features``), never as a NaN vector that would pass
  for a usable one; the head pose exists only where the recorder built it
  (usable gaze), so the shut-eye pose a live session reports is absent;
* frames are published in order, with ``observed_s`` from the recorded
  timestamps (relative to the first) plus ``start_s``. Replay proves the
  core's behaviour on those inputs; it does not reproduce camera timing, queue
  delays or camera-to-cursor latency.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from gazelink_core.domain.observation import FrameObservation, Reason

ObservationSink = Callable[[FrameObservation], None]


def observations(recording: Any, *, start_s: float = 0.0) -> Iterator[FrameObservation]:
    """Every row of ``recording`` as the observation the live source would publish."""

    stamps = np.asarray(recording.timestamp_ns, dtype=np.int64)
    t0 = int(stamps[0]) if len(stamps) else 0
    for i in range(recording.n_rows):
        status = bool(recording.gaze_status[i])
        openness = (float(recording.openness[i, 0]), float(recording.openness[i, 1]))
        face = status or openness[0] > 0.0 or openness[1] > 0.0
        head = recording.head[i] if bool(recording.head_valid[i]) else None
        raw = recording.raw_cm[i]
        raw_cm = (float(raw[0]), float(raw[1])) if status and np.all(np.isfinite(raw)) else None
        state = str(recording.tracking_state[i])
        features = recording.features[i] if status else None
        reasons = set()
        if features is not None and not np.any(np.isfinite(features)):
            features = None
            reasons.add(Reason.NO_FEATURES)
        if not status:
            reasons.add(Reason.NO_GAZE)
        if not face:
            reasons.add(Reason.NO_FACE)
        if head is None:
            reasons.add(Reason.HEAD_UNAVAILABLE)
        yield FrameObservation(
            frame_seq=int(recording.frame_seq[i]),
            observed_s=start_s + (int(stamps[i]) - t0) / 1e9,
            timestamp_ns=int(stamps[i]),
            timestamp_available=int(stamps[i]) != 0,
            face_present=face,
            gaze_status=status,
            features=features,
            openness_image=openness,
            openness_available=True,
            head6=head,
            pnp_deg=None,
            raw_gaze_cm=raw_cm,
            tracking_state_name=state,
            tracking_state_text=state,
            tracking_state_present=True,
            reasons=frozenset(reasons),
        )


class ReplaySource:
    """Same shape as ``GazeFollowerSource``; ``run`` publishes synchronously."""

    def __init__(self, recording: Any, *, start_s: float = 0.0) -> None:
        self.recording = recording
        self.start_s = start_s
        self._sinks: list[ObservationSink] = []
        self._stopping = threading.Event()
        self.started = False
        self.closed = False
        self.published = 0
        self.errors = 0
        self.last_error: str | None = None

    def open(self) -> ReplaySource:
        return self

    def subscribe(self, sink: ObservationSink) -> None:
        self._sinks.append(sink)

    def start(self) -> None:
        self.started = True

    def run(self, *, before_each: Callable[[FrameObservation], None] | None = None) -> int:
        """Publish every row in order. Returns how many were published."""

        for obs in observations(self.recording, start_s=self.start_s):
            if self._stopping.is_set():
                break
            if before_each is not None:
                before_each(obs)
            for sink in list(self._sinks):
                sink(obs)
            self.published += 1
        return self.published

    def stop(self, *, wait_s: float = 0.0) -> bool:
        self._stopping.set()
        return True

    def close(self) -> dict[str, Any]:
        self.stop()
        self.closed = True
        return {"replay": True, "published": self.published}
