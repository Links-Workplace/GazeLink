"""Frame pipeline numeric equivalence against pre-refactor digests (ARCH-01 stage C).

``tests/golden/pipeline_digests.json`` holds SHA-256 digests (and counts) of
what the pre-refactor frame code produced for fixed inputs:

* LIVE policy -- ``gf_live.LiveRunner``: gate decision, steadiness, raw and
  filtered prediction, face presence, openness ratios, gesture and wink events,
  per frame (float64 features).
* RECORD policy -- ``gf_record.ProtocolRunner`` over protocol A: overlay raw and
  filtered prediction and its timestamp, plus a digest of the recording the
  builder produced (float32 features, idle frames dropped from the overlay).

Only digests are stored: no prediction, point, feature or head value from a
recording is written anywhere by this test (CLAUDE.md 4.3). Equality is exact:
the refactor must perform the same operations in the same order on the same
dtypes. The real-data cases skip when the local recordings are absent unless
``GAZELINK_REQUIRE_REAL_DATA=1``.

What this does NOT prove: camera capture, the library's feature extraction,
queue timing or camera-to-cursor latency. Replayed features are float32 as
stored, so the live float64 path is exercised on float32 inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import gf_common as C  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_live as L  # noqa: E402
import gf_presets as PRE  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "pipeline_digests.json"
CONFIG = "svr_fzscore_lnone_C100_g0.0005"
RIG = C.RigGeometry(60.0, 63.6, 120.0, 33.75, 5120, 1440)
FILTER = GF.FilterSettings(5120, 1440, kind=GF.FilterKind.ONE_EURO, one_euro_min_cutoff_hz=0.4)
REAL_CASES = {
    "round46_T1_pooled": (
        HERE / "recordings" / "round46",
        "T1",
        HERE
        / "results"
        / "zones"
        / "models"
        / "svr_fzscore_lnone_C100_g0.0005__pooled_incl46_excl44",
    ),
    "round46_A_pooled": (
        HERE / "recordings" / "round46",
        "A",
        HERE
        / "results"
        / "zones"
        / "models"
        / "svr_fzscore_lnone_C100_g0.0005__pooled_incl46_excl44",
    ),
    "resolution_round3_T1_hi": (
        HERE / "recordings" / "resolution" / "round3",
        "T1",
        HERE / "results" / "hi" / "models" / "svr_fzscore_lnone_C100_g0.0005__hi_pool_r1-3",
    ),
}


class _Digest:
    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self.rows = 0

    def add(self, *values: Any) -> None:
        self._h.update(repr(tuple(_canon(v) for v in values)).encode("ascii"))
        self._h.update(b"\n")
        self.rows += 1

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def _canon(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return "nan" if math.isnan(value) else value.hex()
    if isinstance(value, np.generic):
        return _canon(value.item())
    if isinstance(value, np.ndarray):
        return hashlib.sha256(
            np.ascontiguousarray(value).tobytes() + str(value.dtype).encode()
        ).hexdigest()
    if isinstance(value, (tuple, list)):
        return tuple(_canon(v) for v in value)
    return repr(value)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _frames(rec: Any) -> list[tuple[float, Any, Any, np.ndarray | None]]:
    """(time_s, face_info, gaze_info, head) per recorded row, library-shaped."""

    t0 = int(rec.timestamp_ns[0])
    out = []
    for i in range(rec.n_rows):
        status = bool(rec.gaze_status[i])
        feats = rec.features[i]
        head = rec.head[i] if bool(rec.head_valid[i]) else None
        raw = rec.raw_cm[i]
        gaze = SimpleNamespace(
            status=status,
            features=feats if status else None,
            raw_gaze_coordinates=None if not np.all(np.isfinite(raw)) else raw,
            tracking_state=SimpleNamespace(name=str(rec.tracking_state[i])),
            timestamp=int(rec.timestamp_ns[i]),
        )
        face = SimpleNamespace(
            status=status or float(rec.openness[i, 0]) > 0.0,
            left_eye_openness=float(rec.openness[i, 0]),
            right_eye_openness=float(rec.openness[i, 1]),
            timestamp=int(rec.timestamp_ns[i]),
        )
        out.append(((int(rec.timestamp_ns[i]) - t0) / 1e9, face, gaze, head))
    return out


def live_digest(model: Any, frames: list, *, wink: Any = None, gate: Any = None) -> dict[str, Any]:
    clock = _Clock()
    heads = {"h": None}
    runner = L.LiveRunner(
        model,
        None,
        RIG,
        FILTER,
        clock=clock,
        head_builder=lambda _face: heads["h"],
        wink=wink,
        gate=gate,
    )
    digest = _Digest()
    valid = winks = gestures = 0
    for t, face, gaze, head in frames:
        clock.now = t
        heads["h"] = head
        runner.on_frame(face, gaze)
        s = runner.state
        g = runner.drain_gesture_events()
        w = runner.drain_wink_events()
        valid += int(s.tracking)
        winks += len(w)
        gestures += len(g)
        digest.add(
            s.tracking,
            s.eyes_steady,
            s.face_present,
            s.unfiltered,
            s.point,
            s.raw_model,
            s.openness,
            s.openness_ratio,
            s.head_pitch,
            s.updated_s,
            tuple((when, str(ev)) for when, ev in g),
            tuple(w),
        )
    if runner.errors:
        raise AssertionError(f"live runner raised: {runner.last_error}")
    return {
        "frames": digest.rows,
        "valid": valid,
        "winks": winks,
        "gestures": gestures,
        "sha256": digest.hexdigest(),
    }


def record_digest(model: Any, frames: list) -> dict[str, Any]:
    clock = _Clock()
    heads = {"h": None}
    spec = R.protocol_a()
    meta = {"rig": RIG.to_dict(), "targets": spec.exported_targets()}
    builder = S.RecordingBuilder("A", 0, meta)
    runner = R.ProtocolRunner(
        spec,
        builder,
        RIG,
        clock=clock,
        speed=1.0,
        head_builder=lambda _f: heads["h"],
        pnp=lambda _f: None,
        overlay_model=model,
        overlay_filter_settings=FILTER,
    )
    runner.start()
    digest = _Digest()
    shown = 0
    for t, face, gaze, head in frames:
        clock.now = t
        heads["h"] = head
        runner.on_frame(face, gaze)
        s = runner.state
        shown += int(s.overlay_norm is not None)
        digest.add(
            s.phase,
            s.progress,
            s.target_pos,
            s.overlay_raw_norm,
            s.overlay_norm,
            s.overlay_updated_s,
            s.raw_norm,
            s.head_valid,
        )
    rec = builder.freeze() if len(builder) else None
    rec_digest = _Digest()
    if rec is not None:
        for name in (
            "frame_seq",
            "timestamp_ns",
            "elapsed_ms",
            "target_id",
            "phase",
            "features",
            "head",
            "openness",
            "tracking_state",
            "gaze_status",
            "accepted",
        ):
            rec_digest.add(name, np.asarray(getattr(rec, name)))
    if runner.errors:
        raise AssertionError(f"protocol runner raised: {runner.last_error}")
    return {
        "frames": digest.rows,
        "overlay_shown": shown,
        "sha256": digest.hexdigest(),
        "recording_sha256": rec_digest.hexdigest(),
    }


def _synthetic() -> tuple[Any, list]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 258)).astype(np.float32)
    y = np.column_stack([rng.uniform(-24, 24, 80), rng.uniform(5, 30, 80)])
    config = next(c for c in PRE.sweep_with_presets(()) if c.name == CONFIG)
    model = FIT.FittedModel.fit(config, X, y, rig=RIG, train_meta={"fitted_by": "test"})
    frames = []
    for i in range(300):
        t = i / 30.0
        status = not (100 <= i < 110)  # a tracking gap
        # A deliberate right wink (the person's right = image-left column) at 150-160,
        # a natural both-eye blink at 200-204, open otherwise.
        left_img, right_img = 150.0, 130.0
        if 150 <= i < 160:
            left_img = 10.0
        if 200 <= i < 205:
            left_img, right_img = 5.0, 5.0
        feats = rng.normal(size=258).astype(np.float32)
        gaze = SimpleNamespace(
            status=status,
            features=feats if status else None,
            raw_gaze_coordinates=(float(i % 7), 12.0),
            tracking_state=None,
            timestamp=int(t * 1e9),
        )
        face = SimpleNamespace(
            status=status,
            left_eye_openness=left_img,
            right_eye_openness=right_img,
            timestamp=int(t * 1e9),
        )
        frames.append((t, face, gaze, None))
    return model, frames


def capture() -> dict[str, Any]:
    out: dict[str, Any] = {}
    model, frames = _synthetic()
    wink = PROF.Profile(name="x", model_dir=".", rig={}, filter={}).wink_config()
    out["synthetic"] = {
        "live": live_digest(model, frames, wink=wink),
        "record": record_digest(model, frames),
    }
    for name, (rec_dir, protocol, model_dir) in REAL_CASES.items():
        if not (rec_dir / f"{protocol}.npz").exists() or not (model_dir / "schema.json").exists():
            continue
        rec = S.Recording.load(rec_dir, protocol)
        real_model = FIT.FittedModel.load(model_dir)
        fr = _frames(rec)
        out[name] = {"live": live_digest(real_model, fr), "record": record_digest(real_model, fr)}
    return out


class PipelineEquivalenceTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("GAZELINK_UPDATE_PIPELINE_DIGESTS") == "1":
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(json.dumps(capture(), indent=1) + "\n", encoding="utf-8")
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        # A file just written is compared with itself: refuse to call that a pass.
        cls.regenerated = os.environ.get("GAZELINK_UPDATE_PIPELINE_DIGESTS") == "1"

    def setUp(self) -> None:
        if self.regenerated:
            self.skipTest("golden file regenerated in this run; comparison skipped")

    def test_synthetic_frames_reproduce_the_pre_refactor_digests(self) -> None:
        model, frames = _synthetic()
        wink = PROF.Profile(name="x", model_dir=".", rig={}, filter={}).wink_config()
        self.assertEqual(live_digest(model, frames, wink=wink), self.golden["synthetic"]["live"])
        self.assertEqual(record_digest(model, frames), self.golden["synthetic"]["record"])

    def test_the_synthetic_case_exercises_gate_filter_and_gestures(self) -> None:
        live = self.golden["synthetic"]["live"]
        self.assertGreater(live["valid"], 0)
        self.assertLess(live["valid"], live["frames"], "the tracking gap never closed the gate")

    def test_recordings_reproduce_the_pre_refactor_digests(self) -> None:
        ran = 0
        for name, (rec_dir, protocol, model_dir) in REAL_CASES.items():
            present = (rec_dir / f"{protocol}.npz").exists() and (
                model_dir / "schema.json"
            ).exists()
            with self.subTest(case=name):
                if not present or name not in self.golden:
                    if os.environ.get("GAZELINK_REQUIRE_REAL_DATA") == "1":
                        self.fail(f"required real data missing for {name}")
                    continue
                rec = S.Recording.load(rec_dir, protocol)
                model = FIT.FittedModel.load(model_dir)
                fr = _frames(rec)
                self.assertEqual(live_digest(model, fr), self.golden[name]["live"])
                self.assertEqual(record_digest(model, fr), self.golden[name]["record"])
                ran += 1
        if ran == 0:
            self.skipTest("no local recordings to compare")


if __name__ == "__main__":
    unittest.main()
