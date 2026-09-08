"""Dwell selection state machine and layouts: no camera, no model, no OS input.

These assert the safety properties, not that the functions return something:
a dwell must fire once per entry, must not fire on stale or missing data, and
must not be able to see which target the operator was asked to look at.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_dwell as D  # noqa: E402

HZ = 60.0
STEP = 1.0 / HZ
INSIDE_LEFT = (0.35, 0.5)
INSIDE_RIGHT = (0.65, 0.5)
DEAD_ZONE = (0.50, 0.5)


def _frames(ms: float) -> int:
    return int(round(ms / 1000.0 * HZ)) + 1


class _Driver:
    """Feeds an engine frame by frame and collects what it fired."""

    def __init__(self, engine: D.DwellEngine) -> None:
        self.engine = engine
        self.t = 0.0
        self.fired: list[D.Activation] = []

    def hold(
        self, point: tuple[float, float] | None, ms: float, *, fresh: bool = True
    ) -> list[D.Activation]:
        for _ in range(_frames(ms)):
            self.t += STEP
            got = self.engine.update(self.t, point, fresh=fresh)
            if got is not None:
                self.fired.append(got)
        return self.fired


class ButtonTests(unittest.TestCase):
    def test_an_inverted_rectangle_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            D.Button("bad", 0.6, 0.1, 0.4, 0.9)
        with self.assertRaises(ValueError):
            D.Button("empty", 0.4, 0.1, 0.4, 0.9)

    def test_containment_includes_the_edge(self) -> None:
        b = D.Button("b", 0.3, 0.1, 0.5, 0.9)
        self.assertTrue(b.contains((0.3, 0.1)))
        self.assertTrue(b.contains((0.5, 0.9)))
        self.assertFalse(b.contains((0.5001, 0.5)))

    def test_duplicate_keys_are_refused(self) -> None:
        """Two targets with one key would make an activation ambiguous."""

        with self.assertRaises(ValueError):
            D.DwellEngine([D.Button("x", 0, 0, 0.1, 0.1), D.Button("x", 0.2, 0, 0.3, 0.1)])


class DwellEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = D.DwellEngine(D.layout_a(), D.DwellConfig(dwell_ms=900.0))
        self.drive = _Driver(self.engine)

    def test_resting_long_enough_activates_the_target_being_looked_at(self) -> None:
        fired = self.drive.hold(INSIDE_LEFT, 1000.0)
        self.assertEqual([a.button for a in fired], ["LEFT"])

    def test_resting_too_briefly_does_not_activate(self) -> None:
        self.assertEqual(self.drive.hold(INSIDE_LEFT, 500.0), [])
        self.assertGreater(self.engine.progress, 0.0)
        self.assertLess(self.engine.progress, 1.0)

    def test_holding_far_longer_still_activates_only_once(self) -> None:
        """The stuck-action failure: one intention is one activation."""

        fired = self.drive.hold(INSIDE_LEFT, 6000.0)
        self.assertEqual(len(fired), 1)

    def test_leaving_and_returning_re_arms(self) -> None:
        self.drive.hold(INSIDE_LEFT, 1000.0)
        self.drive.hold(DEAD_ZONE, 300.0)
        self.drive.hold(INSIDE_LEFT, 1000.0)
        self.assertEqual([a.button for a in self.drive.fired], ["LEFT", "LEFT"])

    def test_progress_does_not_carry_from_one_target_to_another(self) -> None:
        """Half a dwell on one target plus half on another must select
        neither, or a sweep across the screen would activate something."""

        self.drive.hold(INSIDE_LEFT, 600.0)
        fired = self.drive.hold(INSIDE_RIGHT, 600.0)
        self.assertEqual(fired, [])

    def test_a_point_in_dead_space_activates_nothing(self) -> None:
        self.assertEqual(self.drive.hold(DEAD_ZONE, 3000.0), [])
        self.assertIsNone(self.engine.hovered)

    def test_losing_tracking_cancels_the_fill(self) -> None:
        self.drive.hold(INSIDE_LEFT, 700.0)
        self.drive.hold(None, 100.0)
        self.assertEqual(self.engine.progress, 0.0)
        fired = self.drive.hold(INSIDE_LEFT, 400.0)
        self.assertEqual(fired, [], "the fill resumed from where it was interrupted")

    def test_a_stale_point_cannot_activate(self) -> None:
        """A frozen prediction keeps 'resting' on a target while the camera is
        dead; acting on it would select without anyone looking."""

        self.assertEqual(self.drive.hold(INSIDE_LEFT, 3000.0, fresh=False), [])

    def test_a_blink_part_way_through_does_not_activate(self) -> None:
        self.drive.hold(INSIDE_LEFT, 600.0)
        self.drive.hold(None, 150.0)
        self.assertEqual(self.drive.hold(INSIDE_LEFT, 600.0), [])

    def test_losing_tracking_does_not_re_arm_a_latched_target(self) -> None:
        """Absence is not evidence of looking away. If loss cleared the latch,
        a blink while resting on a target would fire it a second time."""

        self.drive.hold(INSIDE_LEFT, 1000.0)
        self.drive.hold(None, 500.0)
        fired = self.drive.hold(INSIDE_LEFT, 2000.0)
        self.assertEqual(len(fired), 1, "a blink re-armed the target being rested on")

    def test_reset_clears_everything_in_flight(self) -> None:
        """Pause, resume and a profile switch must not leave progress that was
        accumulated against a different model."""

        self.drive.hold(INSIDE_LEFT, 700.0)
        self.engine.reset()
        self.assertEqual(self.engine.progress, 0.0)
        self.assertIsNone(self.engine.hovered)
        self.assertEqual(self.drive.hold(INSIDE_LEFT, 400.0), [])

    def test_the_activation_records_where_and_how_long(self) -> None:
        fired = self.drive.hold(INSIDE_LEFT, 1000.0)
        act = fired[0]
        self.assertEqual(act.position, INSIDE_LEFT)
        self.assertGreaterEqual(act.dwell_ms, 900.0)
        self.assertEqual(act.event_type, "DWELL_SELECT")
        self.assertEqual(act.source, "DWELL")


class BlindnessTests(unittest.TestCase):
    """A selection mechanism that can see the answer is not measuring
    selection. This is enforced structurally, not by convention."""

    def test_update_takes_no_argument_naming_the_requested_target(self) -> None:
        names = set(inspect.signature(D.DwellEngine.update).parameters)
        for forbidden in ("requested", "target", "expected", "answer", "correct"):
            self.assertNotIn(forbidden, names)

    def test_the_engine_holds_no_reference_to_a_requested_target(self) -> None:
        engine = D.DwellEngine(D.layout_a())
        for attribute in vars(engine):
            for forbidden in ("requested", "expected", "answer"):
                self.assertNotIn(forbidden, attribute)

    def test_identical_points_give_identical_results_whatever_was_asked(self) -> None:
        """Two runs of the same gaze data must agree exactly; nothing outside
        the point stream may influence what fires."""

        runs = []
        for _ in range(2):
            drive = _Driver(D.DwellEngine(D.layout_a()))
            drive.hold(INSIDE_LEFT, 400.0)
            drive.hold(INSIDE_RIGHT, 1000.0)
            runs.append([(a.button, round(a.dwell_ms, 3)) for a in drive.fired])
        self.assertEqual(runs[0], runs[1])
        self.assertEqual([b for b, _ in runs[0]], ["RIGHT"])


class LayoutTests(unittest.TestCase):
    def test_every_target_stays_inside_the_validated_band(self) -> None:
        """Outside the band the model reads 964 px against 234 px inside it,
        so a target there would measure the model's failure, not dwell."""

        for buttons in (D.layout_a(), D.layout_b()):
            for b in buttons:
                self.assertGreaterEqual(b.x0, D.BAND[0] - 1e-9)
                self.assertLessEqual(b.x1, D.BAND[1] + 1e-9)

    def test_targets_do_not_overlap(self) -> None:
        ordered = sorted(D.layout_b(), key=lambda b: b.x0)
        for left, right in zip(ordered, ordered[1:], strict=False):
            self.assertLess(left.x1, right.x0)

    def test_a_band_too_narrow_for_the_targets_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            D.column_layout(["a", "b", "c", "d", "e"], gap=0.15)

    def test_layout_a_clears_the_worst_measured_bias(self) -> None:
        """Worst measured: 0.056 of width horizontally, 0.387 vertically."""

        self.assertEqual(D.layout_warnings(D.layout_a(), bias_x=0.056, bias_y=0.387), [])

    def test_layout_b_is_reported_as_too_tight_before_it_is_run(self) -> None:
        """B is expected to under-perform; the point is that it says so in
        advance rather than being explained away afterwards."""

        warnings = D.layout_warnings(D.layout_b(), bias_x=0.056, bias_y=0.387)
        self.assertTrue(any("MISS RISK" in w for w in warnings))

    def test_neither_layout_puts_a_neighbour_within_reach_of_the_measured_bias(self) -> None:
        """Half-width plus gap, not gap alone. At 0.056 the neighbour is out
        of reach in both layouts: the realistic failure is a miss."""

        for buttons in (D.layout_a(), D.layout_b()):
            warnings = D.layout_warnings(buttons, bias_x=0.056, bias_y=0.387)
            self.assertFalse([w for w in warnings if "NEIGHBOUR RISK" in w])

    def test_a_large_enough_bias_does_put_the_neighbour_in_reach(self) -> None:
        """The guard has to be able to fire, or it guards nothing."""

        warnings = D.layout_warnings(D.layout_b(), bias_x=0.12, bias_y=0.1)
        self.assertTrue(any("NEIGHBOUR RISK" in w for w in warnings))

    def test_a_sideways_bias_misses_in_the_tight_layout_but_holds_in_the_safe_one(self) -> None:
        """The same biased operator: safe in A, non-activating in B. This is
        the layout decision made into a test."""

        bias = 0.06
        safe = _Driver(D.DwellEngine(D.layout_a()))
        left_a = next(b for b in D.layout_a() if b.key == "LEFT")
        safe.hold((left_a.centre[0] + bias, 0.5), 1200.0)
        self.assertEqual([a.button for a in safe.fired], ["LEFT"])

        tight = _Driver(D.DwellEngine(D.layout_b()))
        middle = next(b for b in D.layout_b() if b.key == "MIDDLE")
        tight.hold((middle.centre[0] + bias, 0.5), 1200.0)
        self.assertEqual(
            [a.button for a in tight.fired],
            [],
            "the biased point should land in dead space, selecting nothing",
        )

    def test_a_bias_past_the_dead_zone_really_does_select_the_neighbour(self) -> None:
        """Dead space buys tolerance, it does not remove the failure."""

        tight = _Driver(D.DwellEngine(D.layout_b()))
        middle = next(b for b in D.layout_b() if b.key == "MIDDLE")
        tight.hold((middle.centre[0] + 0.12, 0.5), 1200.0)
        self.assertEqual([a.button for a in tight.fired], ["RIGHT"])


if __name__ == "__main__":
    unittest.main()
