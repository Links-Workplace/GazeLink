"""Shutdown and partial-acquisition safety of the live session (ARCH-01 stage B).

Fakes only: no camera, no window, no OS input. Each test names the failure it
injects and states what happened BEFORE the fix, so the expected change of
trace is limited to the failure scenario itself (TECHNICAL_SPEC 16, stage B).
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_click as CK  # noqa: E402
import gf_cursor as CUR  # noqa: E402
import gf_keys as KEYS  # noqa: E402
import gf_live as L  # noqa: E402
import gf_menu as MENU  # noqa: E402
from gazelink_core.app.lifecycle import CleanupStack  # noqa: E402

from test_live_loop import _run  # noqa: E402


def _record_releases(
    test: unittest.TestCase, events: list[str], *, keys_raise: bool = False
) -> None:
    """Name every adapter release in ``events``, calling the real release too."""

    originals = {
        (CK.ClickAdapter, "release"): CK.ClickAdapter.release,
        (KEYS.KeyAdapter, "release"): KEYS.KeyAdapter.release,
        (CUR.CursorAdapter, "release"): CUR.CursorAdapter.release,
    }

    def wrap(name: str, real, fail: bool = False):  # noqa: ANN001, ANN202
        def release(self) -> None:  # noqa: ANN001
            events.append(name)
            real(self)
            if fail:
                raise OSError(f"{name} failed")

        return release

    CK.ClickAdapter.release = wrap("button release", originals[(CK.ClickAdapter, "release")])
    KEYS.KeyAdapter.release = wrap(
        "key release", originals[(KEYS.KeyAdapter, "release")], fail=keys_raise
    )
    CUR.CursorAdapter.release = wrap("pointer restore", originals[(CUR.CursorAdapter, "release")])

    def restore() -> None:
        for (cls, attr), fn in originals.items():
            setattr(cls, attr, fn)

    test.addCleanup(restore)


class CleanupOrderTests(unittest.TestCase):
    def test_input_is_released_before_the_display_the_library_and_the_report(self) -> None:
        """Before: the session report printed first, then button, keys, pointer."""

        events: list[str] = []
        _record_releases(self, events)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, events=events, max_seconds=0.2)
        self.assertEqual(
            events,
            [
                "button release",
                "key release",
                "pointer restore",
                "display close",
                "library shutdown",
            ],
        )
        text = out.getvalue()
        self.assertIn("\nsession:", text)
        # The library shutdown is the LAST cleanup step and prints a marker, so
        # the report printed after it is after every release as well.
        self.assertLess(text.index("LIBRARY-SHUT"), text.index("\nsession:"))

    def test_a_failing_report_still_releases_everything(self) -> None:
        """Before: an exception in the report skipped every release."""

        events: list[str] = []
        _record_releases(self, events)
        original = MENU.layout_warnings

        def broken() -> list[str]:
            raise RuntimeError("report broke")

        MENU.layout_warnings = broken
        self.addCleanup(setattr, MENU, "layout_warnings", original)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, events=events, max_seconds=0.2)
        self.assertEqual(
            events,
            [
                "button release",
                "key release",
                "pointer restore",
                "display close",
                "library shutdown",
            ],
        )
        self.assertIn("session report failed", out.getvalue())

    def test_one_failing_release_does_not_stop_the_others(self) -> None:
        """Before: the releases were unguarded statements in a row."""

        events: list[str] = []
        _record_releases(self, events, keys_raise=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, events=events, max_seconds=0.2)
        self.assertEqual(
            events,
            [
                "button release",
                "key release",
                "pointer restore",
                "display close",
                "library shutdown",
            ],
        )
        self.assertIn("cleanup step 'key release' failed", out.getvalue())


class EarlyExitTests(unittest.TestCase):
    def test_escape_during_warm_up_releases_and_shuts_down_without_crashing(self) -> None:
        """Before: the report read locals assigned only after warm-up, raised
        UnboundLocalError inside ``finally`` and skipped every release and the
        library shutdown."""

        events: list[str] = []
        _record_releases(self, events)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, events=events, escape_during_warmup=True)
        self.assertEqual(
            events,
            [
                "button release",
                "key release",
                "pointer restore",
                "display close",
                "library shutdown",
            ],
        )
        self.assertNotIn("\nsession:", out.getvalue(), "no session ran, so none is reported")


class LoopFailureTests(unittest.TestCase):
    def test_an_interrupt_inside_the_loop_releases_then_reports_then_propagates(self) -> None:
        events: list[str] = []
        _record_releases(self, events)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(KeyboardInterrupt):
            _run(self, events=events, loop_error=KeyboardInterrupt())
        self.assertEqual(
            events,
            [
                "button release",
                "key release",
                "pointer restore",
                "display close",
                "library shutdown",
            ],
        )
        text = out.getvalue()
        self.assertLess(text.index("LIBRARY-SHUT"), text.index("\nsession:"))

    def test_a_stuck_button_is_reported_even_when_the_report_fails(self) -> None:
        original_release = CK.ClickAdapter.release
        original_warnings = MENU.layout_warnings

        def stuck(self) -> None:  # noqa: ANN001
            self.stuck_button = "left"

        def broken() -> list[str]:
            raise RuntimeError("report broke")

        CK.ClickAdapter.release = stuck
        MENU.layout_warnings = broken
        self.addCleanup(setattr, CK.ClickAdapter, "release", original_release)
        self.addCleanup(setattr, MENU, "layout_warnings", original_warnings)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, max_seconds=0.2)
        self.assertIn("WARNING: input may still be held after shutdown", out.getvalue())


class SourceStopFailureTests(unittest.TestCase):
    def test_a_source_that_fails_to_stop_does_not_skip_any_release(self) -> None:
        """Stage E review B1. Before: ``source.stop()`` ran unguarded ahead of
        the cleanup stack, so a stop that raised skipped every release."""

        import fake_live_env  # noqa: PLC0415

        events: list[str] = []
        _record_releases(self, events)
        original = fake_live_env.FakeSource.stop

        calls = {"n": 0}

        def failing_stop(self) -> bool:  # noqa: ANN001
            # Fails when shutdown asks first; the library's own close stops it
            # again later, and that must still get through.
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("camera thread wedged")
            self.stopped = True
            return True

        fake_live_env.FakeSource.stop = failing_stop
        self.addCleanup(setattr, fake_live_env.FakeSource, "stop", original)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, events=events, max_seconds=0.2)
        self.assertEqual(
            events,
            [
                "button release",
                "key release",
                "pointer restore",
                "display close",
                "library shutdown",
            ],
        )
        self.assertIn("cleanup step 'stop' failed", out.getvalue())

    def test_a_stop_that_times_out_is_reported(self) -> None:
        import fake_live_env  # noqa: PLC0415

        original = fake_live_env.FakeSource.stop
        fake_live_env.FakeSource.stop = lambda self: False
        self.addCleanup(setattr, fake_live_env.FakeSource, "stop", original)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(self, max_seconds=0.2)
        self.assertIn("still being processed when shutdown began", out.getvalue())


class PartialAcquisitionTests(unittest.TestCase):
    def test_a_display_that_fails_to_open_still_shuts_the_library(self) -> None:
        """Before: the display was built outside the try, so the library leaked."""

        events: list[str] = []
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            _run(self, events=events, display_error=RuntimeError("no window"))
        self.assertEqual(events, ["library shutdown"])

    def test_a_refused_keyboard_configuration_still_closes_what_was_opened(self) -> None:
        """Before: SystemExit from the scan config skipped display and library cleanup."""

        events: list[str] = []
        _record_releases(self, events)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            _run(self, events=events, scan_ms=-1.0)
        # The pointer adapter exists and is restored; the button and keys were
        # never created, so nothing of theirs is registered.
        self.assertEqual(events, ["pointer restore", "display close", "library shutdown"])


class StoppingTests(unittest.TestCase):
    def test_a_frame_arriving_after_shutdown_never_reaches_the_runner(self) -> None:
        """Before: the subscriber stayed live while the session was torn down."""

        calls: list[object] = []
        _display, _sends, runner = _run(self, max_seconds=0.2)
        runner.on_observation = lambda *a: calls.append(a)
        self.assertEqual(len(runner.subscribers), 1)
        self.assertTrue(runner.world.source.stopped, "the source was not stopped on shutdown")
        runner.world.source.publish(SimpleNamespace(gaze_status=True))
        self.assertEqual(calls, [], "a frame after shutdown reached the runner")


class FaceFreshnessClockTests(unittest.TestCase):
    def test_freshness_is_judged_on_the_runner_clock(self) -> None:
        """Before: updated_s from the runner clock was compared with time.monotonic()."""

        state = SimpleNamespace(updated_s=100.0, face_present=True)
        fresh = SimpleNamespace(state=state, clock=lambda: 100.1)
        stale = SimpleNamespace(state=state, clock=lambda: 105.0)
        self.assertTrue(L.state_face_ok(fresh))
        self.assertFalse(L.state_face_ok(stale))
        original = L.time.monotonic
        L.time.monotonic = lambda: (_ for _ in ()).throw(AssertionError("wall clock read"))
        self.addCleanup(setattr, L.time, "monotonic", original)
        self.assertTrue(L.state_face_ok(fresh), "the monotonic clock was consulted")


class CleanupStackTests(unittest.TestCase):
    def test_runs_in_reverse_once_and_survives_failures(self) -> None:
        stack = CleanupStack()
        ran: list[str] = []
        stack.push("a", lambda: ran.append("a"))
        stack.push("b", lambda: (_ for _ in ()).throw(OSError("b")))
        stack.push("c", lambda: ran.append("c"))
        failures = stack.close()
        self.assertEqual(ran, ["c", "a"])
        self.assertEqual([f.step for f in failures], ["b"])
        stack.close()
        self.assertEqual(ran, ["c", "a"], "a second close ran the steps again")

    def test_interrupt_in_one_step_is_raised_only_after_the_rest_ran(self) -> None:
        stack = CleanupStack()
        ran: list[str] = []

        def interrupted() -> None:
            raise KeyboardInterrupt

        stack.push("first", lambda: ran.append("first"))
        stack.push("interrupted", interrupted)
        with self.assertRaises(KeyboardInterrupt):
            stack.close()
        self.assertEqual(ran, ["first"])

    def test_a_step_registered_after_close_runs_immediately(self) -> None:
        stack = CleanupStack()
        stack.close()
        ran: list[str] = []
        stack.push("late", lambda: ran.append("late"))
        self.assertEqual(ran, ["late"])


if __name__ == "__main__":
    unittest.main()
