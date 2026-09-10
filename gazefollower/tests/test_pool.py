"""The accumulated training pool: no camera, no fitting, no OS input.

Recordings are synthesised on disk through the real ``gf_schema`` writer, so
these exercise the same loader the fitter uses rather than a stand-in for it.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_pool as POOL  # noqa: E402
import gf_schema as S  # noqa: E402

RIG = {
    "camera_x_cm": 60.0,
    "camera_y_cm": 63.6,
    "screen_w_cm": 120.0,
    "screen_h_cm": 33.75,
    "device_w_px": 5120,
    "device_h_px": 1440,
}
FEATURE_DIM = 4


def _write_recording(
    directory: Path,
    protocol: str,
    *,
    round_id: int,
    positions: list[tuple[float, float]],
    per_point: int = 5,
    session: str = "s1",
    feature_dim: int = FEATURE_DIM,
    screen_id: str = "PANEL-1",
    rig: dict[str, Any] | None = None,
    corrupt_row: int | None = None,
) -> Path:
    """Write one recording the way a session would, minus the camera.

    Calibration protocols mark their rows ACCEPTED and phase COLLECT; timed
    protocols mark phase COLLECTING and accept nothing, which is exactly the
    difference that makes one mask right for one kind and wrong for the other.
    """

    directory.mkdir(parents=True, exist_ok=True)
    n = len(positions) * per_point
    labels = np.repeat(np.array(positions, dtype=np.float64), per_point, axis=0)
    rng = np.random.default_rng(round_id * 100 + len(protocol))
    features = rng.normal(size=(n, feature_dim)).astype(np.float32)
    if corrupt_row is not None:
        features[corrupt_row, 0] = np.nan
    calibration = protocol in S.CALIBRATION_PROTOCOLS
    phase = S.PHASE_COLLECT if calibration else S.PHASE_COLLECTING
    rec = S.Recording(
        protocol=protocol,
        round_id=round_id,
        meta={
            "session": session,
            "head_names": [],
            "rig": dict(rig or RIG),
            "target_geometry": {"screen_id": screen_id},
        },
        frame_seq=np.arange(n, dtype=np.int64),
        timestamp_ns=np.arange(n, dtype=np.int64) * 33_000_000,
        elapsed_ms=np.zeros(n, dtype=np.float64),
        target_id=np.repeat(np.arange(len(positions)), per_point).astype(np.int32),
        block=np.zeros(n, dtype=np.int32),
        phase=np.array([phase] * n),
        target_xy=labels / 100.0,
        label_cm=labels,
        features=features,
        head=rng.normal(size=(n, 6)),
        head_valid=np.ones(n, dtype=bool),
        pnp_deg=np.zeros((n, 3), dtype=np.float64),
        raw_cm=np.zeros((n, 2), dtype=np.float64),
        openness=np.full((n, 2), 100.0, dtype=np.float64),
        tracking_state=np.array(["TRACKING"] * n),
        gaze_status=np.ones(n, dtype=bool),
        accepted=np.ones(n, dtype=bool) if calibration else np.zeros(n, dtype=bool),
    )
    rec.save(directory)
    return directory


class PoolBuildTests(unittest.TestCase):
    """Nine points measured three times is not the same as 27 points, and the
    difference is the whole reason this module exists."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _three_sessions(self) -> None:
        cal = [(10.0, 5.0), (60.0, 5.0), (110.0, 5.0)]
        check = [(20.0, 20.0), (80.0, 20.0)]
        for i, rid in enumerate((1, 2, 3)):
            _write_recording(
                self.root / f"round{rid}", "A", round_id=rid, positions=cal, session=f"s{i}"
            )
            _write_recording(
                self.root / f"round{rid}", "T1", round_id=rid, positions=check, session=f"s{i}"
            )

    def test_check_protocols_contribute_their_rows(self) -> None:
        """The trap this module was built around. ``training_rows`` returns an
        all-false mask for a check protocol, so a pool that used it everywhere
        silently trained on the calibration points ALONE while every count
        still looked plausible."""

        self._three_sessions()
        pool = POOL.build((), root=self.root)
        self.assertEqual(pool.rows, 3 * (3 * 5 + 2 * 5))
        self.assertEqual(
            pool.positions,
            5,
            "the check protocols contributed no positions, so the pool is three copies "
            "of one calibration -- the thing measured NOT to help",
        )

    def test_the_held_out_recording_never_trains(self) -> None:
        """Scoring a model on rows it trained on reports memory as accuracy."""

        self._three_sessions()
        held = (self.root / "round2", "T1")
        pool = POOL.build((), root=self.root, exclude=[held])
        self.assertNotIn(
            (Path(self.root / "round2"), "T1"),
            [(s.directory, s.protocol) for s in pool.sources],
            "the recording the result is scored on is in its own training set",
        )
        self.assertIn("round2", str([r.directory for r in pool.rejected]))

    def test_holding_out_one_protocol_keeps_the_others_in_that_round(self) -> None:
        """One directory holds several protocols. Excluding by round would
        throw away the calibration that the check is meant to validate."""

        self._three_sessions()
        pool = POOL.build((), root=self.root, exclude=[(self.root / "round2", "T1")])
        kept = {(s.directory.name, s.protocol) for s in pool.sources}
        self.assertIn(("round2", "A"), kept)
        self.assertNotIn(("round2", "T1"), kept)

    def test_a_relative_and_an_absolute_path_exclude_the_same_recording(self) -> None:
        """A held-out recording named by a different but equal path would
        train, and nothing in the output would say so."""

        self._three_sessions()
        awkward = self.root / "round2" / ".." / "round2"
        pool = POOL.build((), root=self.root, exclude=[(awkward, "T1")])
        self.assertNotIn(("round2", "T1"), {(s.directory.name, s.protocol) for s in pool.sources})

    def test_an_incompatible_screen_is_refused_and_the_reason_is_kept(self) -> None:
        """The same eye geometry means a different centimetre on another
        panel. Averaging it in moves the error with nothing recording why."""

        self._three_sessions()
        _write_recording(
            self.root / "round9",
            "A",
            round_id=9,
            positions=[(1.0, 1.0)],
            session="other",
            screen_id="A-DIFFERENT-PANEL",
        )
        pool = POOL.build((), root=self.root)
        self.assertNotIn("round9", {s.directory.name for s in pool.sources})
        reasons = [r.reason for r in pool.rejected if r.directory.name == "round9"]
        self.assertTrue(reasons and "screen" in reasons[0], reasons)

    def test_an_incompatible_feature_width_is_refused(self) -> None:
        self._three_sessions()
        _write_recording(
            self.root / "round9", "A", round_id=9, positions=[(1.0, 1.0)], feature_dim=7
        )
        pool = POOL.build((), root=self.root)
        self.assertNotIn("round9", {s.directory.name for s in pool.sources})

    def test_an_unreadable_recording_is_reported_not_raised(self) -> None:
        """One corrupt file must not make every past session unusable."""

        self._three_sessions()
        (self.root / "round9").mkdir()
        (self.root / "round9" / "A.meta.json").write_text("{not json", encoding="utf-8")
        pool = POOL.build((), root=self.root)
        self.assertEqual(pool.sessions, 3)
        self.assertTrue(any("unreadable" in r.reason for r in pool.rejected))

    def test_non_finite_rows_are_dropped_and_counted(self) -> None:
        """A NaN reaches the fitter as an exception at the end of a session."""

        self._three_sessions()
        _write_recording(
            self.root / "round4",
            "A",
            round_id=4,
            positions=[(30.0, 10.0)],
            session="s4",
            corrupt_row=0,
        )
        pool = POOL.build((), root=self.root)
        self.assertTrue(np.isfinite(pool.X).all(), "a non-finite row reached the training set")
        self.assertTrue(any("non-finite" in r.reason for r in pool.rejected))

    def test_an_empty_root_is_an_empty_pool_and_not_a_crash(self) -> None:
        pool = POOL.build((), root=self.root / "nothing-here")
        self.assertEqual(pool.rows, 0)
        self.assertEqual(pool.positions, 0)
        self.assertFalse(POOL.usable(pool)[0])

    def test_the_pool_is_the_same_twice(self) -> None:
        """Row order changes an SVR fit. A pool that shuffles between runs
        makes two measurements of the same data disagree."""

        self._three_sessions()
        first = POOL.build((), root=self.root)
        second = POOL.build((), root=self.root)
        np.testing.assert_array_equal(first.X, second.X)
        np.testing.assert_array_equal(first.Y_cm, second.Y_cm)

    def test_tune_is_never_pooled(self) -> None:
        """TUNE selects a configuration. Training on it would select on
        training error."""

        self._three_sessions()
        _write_recording(
            self.root / "round1", "TUNE", round_id=1, positions=[(50.0, 15.0)], session="s0"
        )
        pool = POOL.build((), root=self.root)
        self.assertNotIn("TUNE", {s.protocol for s in pool.sources})


class UsabilityGateTests(unittest.TestCase):
    """A pool too small to have been measured must say so, not return a
    number that looks like the measured one."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_one_calibration_alone_is_refused(self) -> None:
        """405 rows at 9 positions is exactly what measured 226 px."""

        _write_recording(
            self.root / "round1",
            "A",
            round_id=1,
            positions=[(float(i) * 10, 5.0) for i in range(9)],
            per_point=45,
        )
        pool = POOL.build((), root=self.root)
        self.assertEqual(pool.positions, 9)
        self.assertEqual(pool.rows, 405, "not the shape a single calibration actually has")
        ok, why = POOL.usable(pool)
        self.assertFalse(ok, "the exact shape that measured 226 px passed the gate")
        # The row gate is reached first at this size; the position gate is
        # covered by the sibling test, which supplies rows and withholds
        # positions. Asserting a particular half here would pin the ORDER of
        # two independent floors rather than the refusal.
        self.assertIn("405", why)

    def test_many_rows_at_few_points_is_refused_for_the_right_reason(self) -> None:
        """The measured failure mode: more measurements of the same points is
        what made a pooled fit WORSE (307 px against 156 px)."""

        _write_recording(
            self.root / "round1",
            "A",
            round_id=1,
            positions=[(10.0, 5.0), (60.0, 5.0), (110.0, 5.0)],
            per_point=2000,
        )
        pool = POOL.build((), root=self.root)
        self.assertGreater(pool.rows, POOL.MIN_POOL_ROWS)
        ok, why = POOL.usable(pool)
        self.assertFalse(ok, "a pool of three points passed the gate on row count alone")
        self.assertIn("same few points", why)

    def test_a_pool_that_matches_the_measured_shape_is_accepted(self) -> None:
        for rid in range(1, 4):
            _write_recording(
                self.root / f"round{rid}",
                "A",
                round_id=rid,
                positions=[(float(i) * 7, float(i % 3) * 10) for i in range(15)],
                per_point=45,
                session=f"s{rid}",
            )
        pool = POOL.build((), root=self.root)
        ok, why = POOL.usable(pool)
        self.assertTrue(ok, why)


class ProvenanceTests(unittest.TestCase):
    """A model trained on an accumulating pool is only reproducible if what
    went into it was recorded."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        _write_recording(
            self.root / "round1", "A", round_id=1, positions=[(10.0, 5.0), (60.0, 5.0)]
        )
        _write_recording(
            self.root / "round2", "T1", round_id=2, positions=[(20.0, 9.0)], session="s2"
        )

    def test_provenance_names_every_recording_that_trained(self) -> None:
        pool = POOL.build((), root=self.root)
        got = pool.provenance()
        self.assertEqual(got["rows"], pool.rows)
        self.assertEqual(len(got["recordings"]), 2)
        self.assertEqual({r["protocol"] for r in got["recordings"]}, {"A", "T1"})

    def test_provenance_survives_json(self) -> None:
        """It is stored in a profile, which is a JSON file."""

        pool = POOL.build((), root=self.root)
        round_tripped = json.loads(json.dumps(pool.provenance()))
        self.assertEqual(round_tripped["positions"], pool.positions)

    def test_rejections_are_reported_not_hidden(self) -> None:
        pool = POOL.build((), root=self.root, exclude=[(self.root / "round2", "T1")])
        self.assertTrue(pool.provenance()["rejected"])


if __name__ == "__main__":
    unittest.main()
