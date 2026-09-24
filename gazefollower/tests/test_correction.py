"""The post-model correction layer: does it correct, and does it hide anything?

Two kinds of test here, and the second matters more.

The first kind is arithmetic: a correction fitted on a known linear distortion
must recover it, applying it must undo it, and a round trip through disk must
come back unchanged.

The second kind is about what a WRAPPER can silently take away. A model that
is wrapped still has to answer everything the system asks a model, because the
callers guard those questions with broad excepts so that a failed check can
never abort a good recording -- and a check that cannot run is treated as a
check that passed. A wrapper missing one method therefore does not crash: it
turns a safety gate off and prints a reassuring line. That happened here with
``support_activation``, which is the frozen-model abort, on exactly the run
meant to validate the correction.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gazelink_core.calibration import correction as CORR  # noqa: E402


class FakeModel:
    """Everything the system asks a model, plus a note of what was asked."""

    def __init__(self, gain: float = 1.0):
        self.gain = gain
        self.schema = SimpleNamespace(base_dim=258, head_names=(), columns=range(258))
        self.config = SimpleNamespace(name="fake")
        self.asked: list[str] = []

    def predict_norm(self, X_raw: np.ndarray, rig: object) -> np.ndarray:
        self.asked.append("predict_norm")
        return np.asarray(X_raw, dtype=np.float64)[:, :2] * self.gain

    def predict_cm(self, X_raw: np.ndarray) -> np.ndarray:
        self.asked.append("predict_cm")
        return np.asarray(X_raw, dtype=np.float64)[:, :2]

    def support_activation(self, X_raw: np.ndarray) -> np.ndarray:
        """The frozen-model check. The one a wrapper must not swallow."""

        self.asked.append("support_activation")
        return np.full(len(X_raw), 0.9)


class AxisCorrectionTests(unittest.TestCase):
    def test_it_recovers_a_known_distortion(self) -> None:
        targets = np.column_stack([np.linspace(0.3, 0.7, 9), np.linspace(0.1, 0.9, 9)])
        predictions = np.column_stack(
            [targets[:, 0] * 1.25 - 0.12, targets[:, 1] * 0.96 + 0.03]
        )
        fitted = CORR.AxisCorrection.fit(targets, predictions)
        self.assertAlmostEqual(fitted.gain_x, 1.25, places=6)
        self.assertAlmostEqual(fitted.offset_x, -0.12, places=6)
        self.assertAlmostEqual(fitted.gain_y, 0.96, places=6)
        self.assertAlmostEqual(fitted.offset_y, 0.03, places=6)

    def test_applying_it_undoes_the_distortion(self) -> None:
        targets = np.column_stack([np.linspace(0.3, 0.7, 9), np.linspace(0.1, 0.9, 9)])
        predictions = np.column_stack(
            [targets[:, 0] * 1.25 - 0.12, targets[:, 1] * 0.96 + 0.03]
        )
        fitted = CORR.AxisCorrection.fit(targets, predictions)
        np.testing.assert_allclose(fitted.apply(predictions), targets, atol=1e-9)

    def test_an_identity_correction_changes_nothing(self) -> None:
        points = np.array([[0.31, 0.22], [0.5, 0.5], [0.69, 0.81]])
        identity = CORR.AxisCorrection(gain_x=1.0, offset_x=0.0, gain_y=1.0, offset_y=0.0)
        np.testing.assert_allclose(identity.apply(points), points)

    def test_missing_predictions_stay_missing(self) -> None:
        """A NaN is a tracking loss and must not become a confident number."""

        fitted = CORR.AxisCorrection(gain_x=1.2, offset_x=-0.1, gain_y=1.0, offset_y=0.0)
        out = fitted.apply(np.array([[np.nan, np.nan], [0.5, 0.5]]))
        self.assertTrue(np.all(np.isnan(out[0])))
        self.assertTrue(np.all(np.isfinite(out[1])))

    def test_a_zero_gain_is_refused(self) -> None:
        """Dividing by it would map the whole screen to one point."""

        with self.assertRaises(ValueError):
            CORR.AxisCorrection(gain_x=0.0, offset_x=0.0, gain_y=1.0, offset_y=0.0)
        with self.assertRaises(ValueError):
            CORR.AxisCorrection(gain_x=1.0, offset_x=0.0, gain_y=float("nan"), offset_y=0.0)

    def test_too_few_points_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CORR.AxisCorrection.fit(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_it_survives_a_round_trip_through_disk(self) -> None:
        fitted = CORR.AxisCorrection(
            gain_x=1.2354, offset_x=-0.115, gain_y=0.9659, offset_y=0.0366,
            source=("round3:T1",), note="why it exists",
        )
        with tempfile.TemporaryDirectory() as tmp:
            fitted.save(Path(tmp))
            again = CORR.AxisCorrection.load(Path(tmp))
            self.assertEqual(again, fitted)

    def test_no_correction_file_means_no_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(CORR.AxisCorrection.load(Path(tmp)))

    def test_an_unknown_version_is_refused_rather_than_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / CORR.CORRECTION_FILE).write_text(
                json.dumps({"gain_x": 1.0, "offset_x": 0.0, "gain_y": 1.0,
                            "offset_y": 0.0, "version": "something-else"}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                CORR.AxisCorrection.load(Path(tmp))

    def test_the_provenance_is_carried(self) -> None:
        """The gain is not a constant of the system; which data it came from decides."""

        fitted = CORR.AxisCorrection.fit(
            np.column_stack([np.linspace(0.3, 0.7, 5)] * 2),
            np.column_stack([np.linspace(0.3, 0.7, 5)] * 2),
            source=("round3:T1", "round4:T1"),
        )
        self.assertEqual(fitted.source, ("round3:T1", "round4:T1"))
        self.assertIn("round3:T1", fitted.describe())


class CorrectedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = FakeModel(gain=1.25)
        self.correction = CORR.AxisCorrection(
            gain_x=1.25, offset_x=0.0, gain_y=1.25, offset_y=0.0
        )
        self.wrapped = CORR.CorrectedModel(self.base, self.correction)

    def test_predict_norm_is_corrected(self) -> None:
        X = np.array([[0.4, 0.6] + [0.0] * 256])
        np.testing.assert_allclose(self.wrapped.predict_norm(X, rig=None), [[0.4, 0.6]])
        np.testing.assert_allclose(self.base.predict_norm(X, rig=None), [[0.5, 0.75]])

    def test_the_frozen_model_check_still_reaches_the_model(self) -> None:
        """REGRESSION. The wrapper defined no support_activation and did not
        delegate, so the recorder's call raised AttributeError, the caller's
        broad except swallowed it, and preflight_verdict(None) answers ok.
        The frozen-model abort was therefore off for every corrected run.
        """

        self.assertTrue(hasattr(self.wrapped, "support_activation"))
        activation = self.wrapped.support_activation(np.zeros((4, 258)))
        self.assertEqual(len(activation), 4)
        self.assertIn("support_activation", self.base.asked)

    def test_the_wrapper_removes_nothing_the_model_offers(self) -> None:
        """Any public attribute of the model must still be reachable."""

        missing = [
            name
            for name in dir(self.base)
            if not name.startswith("_") and not hasattr(self.wrapped, name)
        ]
        self.assertEqual(missing, [], f"the wrapper hides {missing}")

    def test_it_can_be_copied_without_recursing(self) -> None:
        """copy/pickle build the object WITHOUT __init__ and then look up
        __deepcopy__ / __reduce_ex__. A plain delegation re-enters __getattr__
        looking for self.base and recurses until the stack ends."""

        import copy  # noqa: PLC0415
        import pickle  # noqa: PLC0415

        copied = copy.deepcopy(self.wrapped)
        self.assertEqual(copied.correction, self.correction)
        self.assertIsNot(copied.base, self.base)

        restored = pickle.loads(pickle.dumps(self.wrapped))
        self.assertEqual(restored.correction, self.correction)
        # and it is still a working, still-corrected model afterwards
        X = np.array([[0.4, 0.6] + [0.0] * 256])
        np.testing.assert_allclose(restored.predict_norm(X, rig=None), [[0.4, 0.6]])
        self.assertTrue(hasattr(restored, "support_activation"))

    def test_the_schema_is_the_models_own(self) -> None:
        self.assertIs(self.wrapped.schema, self.base.schema)
        self.assertEqual(self.wrapped.schema.base_dim, 258)

    def test_the_base_stays_reachable_for_comparison(self) -> None:
        """The correction may only ever be judged against the model without it."""

        self.assertIs(self.wrapped.base, self.base)

    def test_predict_cm_is_not_secretly_corrected(self) -> None:
        """The correction is defined on screen fractions; a 'corrected cm'
        would invite a caller to convert it a second time."""

        X = np.array([[0.4, 0.6] + [0.0] * 256])
        np.testing.assert_allclose(self.wrapped.predict_cm(X), self.base.predict_cm(X))


class LoadWithCorrectionTests(unittest.TestCase):
    def test_a_bare_model_directory_gives_the_bare_model(self) -> None:
        base = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            loaded = CORR.load_with_correction(Path(tmp), lambda _p: base)
            self.assertIs(loaded, base)

    def test_a_correction_file_wraps_it(self) -> None:
        base = FakeModel()
        with tempfile.TemporaryDirectory() as tmp:
            CORR.AxisCorrection(
                gain_x=1.2, offset_x=0.0, gain_y=1.0, offset_y=0.0
            ).save(Path(tmp))
            loaded = CORR.load_with_correction(Path(tmp), lambda _p: base)
            self.assertIsInstance(loaded, CORR.CorrectedModel)
            self.assertIs(loaded.base, base)
            # and it is still a usable model, not a crippled one
            self.assertTrue(hasattr(loaded, "support_activation"))


class RecorderWiringTests(unittest.TestCase):
    """The recorder and the live view must both load through the correction.

    Checked structurally: exercising them needs a camera, and the failure is
    a silent one -- the uncorrected prediction shown under the corrected
    model's name -- which no smoke test would catch.
    """

    # Every entry point that loads a model. gf_click_practice and
    # gf_dwell_practice matter most: they load the ACTIVE PROFILE's model and
    # gf_click_practice can emit a REAL click, so an uncorrected point there is
    # a click in the wrong place while the profile claims otherwise.
    ENTRY_POINTS = (
        "gf_record.py",
        "gf_live.py",
        "gf_click_practice.py",
        "gf_dwell_practice.py",
        "gf_filter_benchmark.py",
        "gf_recal_compare.py",
        "gf_pool_experiment.py",
        "gf_resolution_view.py",
    )

    def test_every_entry_point_loads_with_correction(self) -> None:
        for name in self.ENTRY_POINTS:
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn(
                "load_with_correction",
                source,
                f"{name} loads a model without honouring correction.json",
            )

    def test_no_entry_point_still_calls_the_bare_loader(self) -> None:
        """A file may import FittedModel.load and pass it INTO
        load_with_correction; what it may not do is call it directly on a
        model directory, which is how the correction gets skipped."""

        offenders = []
        for name in self.ENTRY_POINTS:
            for number, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "FittedModel.load(" in line and "load_with_correction" not in line:
                    offenders.append(f"{name}:{number}")
        self.assertEqual(offenders, [], f"bare FittedModel.load at {offenders}")


if __name__ == "__main__":
    unittest.main()
