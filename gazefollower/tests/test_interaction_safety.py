"""Permission, execution and stopping in the live session (ARCH-01 stage F).

Fakes only: every sender is a recorder, time is a FakeClock, no camera, no
window. Covers TECHNICAL_SPEC 15/16-F: forbidden actions, permission changing
between the decision and the send, releases while paused or stopping, no input
after stopping, an action in flight during shutdown, and events from before a
pause or recovery.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gazelink_core.app.options import LiveOptions  # noqa: E402
from gazelink_core.app.telemetry import SessionTelemetry  # noqa: E402
from gazelink_core.domain.clock import FakeClock  # noqa: E402
from gazelink_core.interaction import actions as ACT  # noqa: E402
from gazelink_core.interaction import control as CTL  # noqa: E402
from gazelink_core.interaction import gesture as GEST  # noqa: E402
from gazelink_core.interaction import keyboard as KB  # noqa: E402
from gazelink_core.interaction import menu as MENU  # noqa: E402
from gazelink_core.interaction import scroll as SCR  # noqa: E402
from gazelink_core.interaction.controller import VK_SPACE, InteractionController  # noqa: E402
from gazelink_core.interaction.executor import ActionExecutor  # noqa: E402
from gazelink_core.interaction.safety import SafetyController  # noqa: E402
from gazelink_core.platform import click as CK  # noqa: E402
from gazelink_core.platform import cursor as CUR  # noqa: E402
from gazelink_core.platform import keys as KEYS  # noqa: E402


class _Pipeline:
    """A snapshot with a face and a steady point, and hand-fed event queues."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.winks: list = []
        self.gestures: list = []
        self.cancels = 0
        self.winks_dropped_mid_frame = 0

    @property
    def state(self):  # noqa: ANN201
        return SimpleNamespace(
            point=(0.5, 0.5),
            raw_model=None,
            unfiltered=(0.5, 0.5),
            tracking=True,
            eyes_steady=True,
            face_present=True,
            openness_ratio=(1.0, 1.0),
            head_pitch=None,
            updated_s=self.clock.now(),
            frames=1,
            fps=30.0,
        )

    def drain_wink_events(self) -> list:
        out, self.winks = self.winks, []
        return out

    def drain_gesture_events(self) -> list:
        out, self.gestures = self.gestures, []
        return out

    def cancel_wink(self) -> int:
        self.cancels += 1
        dropped, self.winks = len(self.winks), []
        return dropped


class _Rig:
    """Executor and controller over fake senders, as a live session builds them."""

    def __init__(self, *, start_active: bool = True, sender=None) -> None:  # noqa: ANN001
        self.clock = FakeClock(1000.0)
        self.sends: list[int] = []
        self.moves: list[tuple[int, int]] = []
        self.keys_sent: list[tuple[int, int, int]] = []
        self.wheel: list[int] = []
        self.space = False
        self.telemetry = SessionTelemetry()
        self.pipeline = _Pipeline(self.clock)
        control = CTL.ToggleMachine(
            start=CTL.ToggleMachine.ACTIVE if start_active else CTL.Mode.PAUSED
        )
        self.safety = SafetyController(control=control, input_enabled=True, clock=self.clock.now)
        self.cursor = CUR.CursorAdapter(
            enabled=True,
            setter=lambda x, y: self.moves.append((x, y)),
            getter=lambda: (0, 0),
            clock=self.clock.now,
        )
        self.clicker = CK.ClickAdapter(
            enabled=True,
            sender=sender or self.sends.append,
            clock=self.clock.now,
            sleep=self.clock.sleep,
        )
        self.scroller = CK.ScrollAdapter(
            enabled=True, sender=self.wheel.append, clock=self.clock.now
        )
        self.keys = KEYS.KeyAdapter(
            enabled=True,
            sender=lambda *k: self.keys_sent.append(k),
            sleep=self.clock.sleep,
            foreground=lambda: 7,
            title=lambda h: "w",
        )
        self.executor = ActionExecutor(
            safety=self.safety,
            router=ACT.ActionRouter(),
            cursor=self.cursor,
            clicker=self.clicker,
            scroller=self.scroller,
            keys=self.keys,
            to_pixels=lambda p: (int(p[0] * 100), int(p[1] * 100)),
            telemetry=self.telemetry,
            clock=self.clock.now,
            hold_after_click_s=0.0,
        )
        self.controller = InteractionController(
            options=LiveOptions(move_cursor=True, confirmed=True, click_by="wink"),
            pipeline=self.pipeline,
            safety=self.safety,
            executor=self.executor,
            telemetry=self.telemetry,
            clock=self.clock.now,
            key_is_down=lambda vk: vk == VK_SPACE and self.space,
            to_pixels=lambda p: (int(p[0] * 100), int(p[1] * 100)),
            board=MENU.MenuModel(dwell_ms=900.0, click_type="single"),
            keyboard=KB.ScanningKeyboard(KB.ScanConfig()),
            scroll_cfg=SCR.ScrollConfig(),
            recovery_menu=GEST.RecoveryMenu([GEST.MenuOption("dismiss", "x")]),
        )
        self.safety.evaluate_face(self.pipeline)

    def click(self) -> bool:
        return self.executor.route(
            ACT.Action.LEFT_CLICK,
            ACT.Source.WINK,
            (0.5, 0.5),
            self.clock.now(),
            ui_mode=ACT.UiMode.CURSOR,
        )

    def frame(self) -> None:
        state = self.pipeline.state
        self.controller.tick_control(state)
        self.controller.tick_pointer(state)
        self.clock.advance(0.005)


class PermissionTests(unittest.TestCase):
    def test_active_session_clicks(self) -> None:
        rig = _Rig()
        self.assertTrue(rig.click())
        self.assertEqual(rig.sends, [CK.MOUSEEVENTF_LEFTDOWN, CK.MOUSEEVENTF_LEFTUP])

    def test_paused_session_refuses_every_click_and_sends_nothing(self) -> None:
        rig = _Rig(start_active=False)
        self.assertFalse(rig.click())
        self.assertEqual(rig.sends, [])
        self.assertEqual(rig.executor.router.refused[ACT.Refusal.PAUSED], 1)

    def test_permission_revoked_between_decision_and_send_is_honoured(self) -> None:
        """The controller screened the wink while ACTIVE; the mode changed
        before the executor sent it. The send reads the mode NOW."""

        rig = _Rig()
        screened = rig.executor.router.screen_gesture(
            issued_at_s=rig.clock.now(),
            now_s=rig.clock.now(),
            ui_mode=ACT.UiMode.CURSOR,
            paused=not rig.safety.selection_armed,
        )
        self.assertIs(screened, ACT.Refusal.NONE)
        rig.safety.control.update(GEST.Event.CONFIRM, tracking_ok=True)  # -> PAUSED
        self.assertFalse(rig.click())
        self.assertEqual(rig.sends, [])

    def test_adapters_receive_the_real_permission_not_a_constant(self) -> None:
        rig = _Rig()
        seen: list[bool] = []
        real = rig.clicker.click

        def spy(*, armed: bool, at=None) -> bool:  # noqa: ANN001
            seen.append(armed)
            return real(armed=armed, at=at)

        rig.clicker.click = spy
        rig.click()
        self.assertEqual(seen, [True])

    def test_scroll_permission_follows_face_and_mode(self) -> None:
        rig = _Rig()
        self.assertTrue(rig.executor.scroll(1))
        rig.safety.face_ok = False
        rig.clock.advance(1.0)
        self.assertFalse(rig.executor.scroll(1))
        self.assertEqual(rig.scroller.refused_not_armed, 1)
        self.assertEqual(rig.wheel, [120])

    def test_resume_is_allowed_while_paused_but_other_commands_are_not(self) -> None:
        rig = _Rig(start_active=False)
        now = rig.clock.now()
        self.assertFalse(
            rig.executor.route(
                ACT.Action.BACK, ACT.Source.DWELL, None, now, ui_mode=ACT.UiMode.CURSOR
            )
        )
        self.assertEqual(rig.keys_sent, [])
        self.assertTrue(
            rig.executor.route(
                ACT.Action.RESUME, ACT.Source.DWELL, None, now, ui_mode=ACT.UiMode.CURSOR
            )
        )
        self.assertTrue(rig.safety.selection_armed)


class StoppingTests(unittest.TestCase):
    def test_nothing_is_sent_or_moved_after_stopping(self) -> None:
        rig = _Rig()
        rig.safety.begin_stop()
        self.assertFalse(rig.click())
        rig.executor.follow((10, 10))
        self.assertFalse(rig.executor.scroll(1))
        self.assertEqual((rig.sends, rig.moves, rig.wheel), ([], [], []))

    def test_releases_still_work_while_paused_and_after_stopping(self) -> None:
        rig = _Rig(start_active=False)
        rig.clicker._held = "left"  # a button left down by a failure
        rig.safety.begin_stop()
        rig.clicker.release()
        self.assertEqual(rig.sends, [CK.MOUSEEVENTF_LEFTUP])

    def test_stop_waits_for_an_action_already_being_sent(self) -> None:
        inside, release = threading.Event(), threading.Event()
        order: list[str] = []
        sends: list[int] = []

        def slow_sender(flag: int) -> None:
            if flag == CK.MOUSEEVENTF_LEFTDOWN:
                inside.set()
                release.wait(5)
            sends.append(flag)

        rig = _Rig(sender=slow_sender)
        worker = threading.Thread(target=lambda: (rig.click(), order.append("click returned")))
        worker.start()
        self.assertTrue(inside.wait(5))
        stopper = threading.Thread(
            target=lambda: (rig.safety.begin_stop(), order.append("stopped"))
        )
        stopper.start()
        time.sleep(0.05)
        self.assertEqual(order, [], "stopping went ahead while a click was half sent")
        release.set()
        worker.join(5)
        stopper.join(5)
        self.assertEqual(order, ["click returned", "stopped"])
        self.assertEqual(sends, [CK.MOUSEEVENTF_LEFTDOWN, CK.MOUSEEVENTF_LEFTUP])
        self.assertFalse(rig.click(), "an action started after stopping")


class StaleEventTests(unittest.TestCase):
    def test_a_stale_wink_sends_nothing(self) -> None:
        rig = _Rig()
        rig.pipeline.winks.append((rig.clock.now() - 1.0, (0.5, 0.5), GEST.Eye.LEFT))
        rig.frame()
        self.assertEqual(rig.sends, [])
        self.assertEqual(rig.telemetry.tally["winks_stale"], 1)

    def test_a_wink_from_before_a_resume_does_not_click_after_it(self) -> None:
        """Bug fixed in ARCH-01 F (TECHNICAL_SPEC 15). Before: a wink the camera
        thread published while PAUSED, drained in the frame where SPACE resumed,
        passed the (now armed) check and clicked -- an event from before the
        recovery acting after it. After: a change of control mode drops every
        wink in flight, exactly as a change of UI mode already did."""

        rig = _Rig(start_active=False)
        rig.frame()
        rig.pipeline.winks.append((rig.clock.now(), (0.5, 0.5), GEST.Eye.LEFT))
        rig.space = True
        rig.frame()
        self.assertTrue(rig.safety.selection_armed, "the fixture never resumed")
        self.assertEqual(rig.sends, [], "a wink made while paused clicked after resuming")
        self.assertEqual(rig.telemetry.tally["winks_dropped_at_mode_edge"], 1)

    def test_a_wink_from_before_a_menu_resume_does_not_click_after_it(self) -> None:
        """RESUME chosen as a menu action goes through the executor, not the
        frame tick; the wink in flight is dropped because choosing a tile
        leaves the menu (a UI-mode change)."""

        rig = _Rig(start_active=False)
        rig.controller.enter_mode(ACT.UiMode.MENU, why="test")
        rig.pipeline.winks.append((rig.clock.now(), (0.5, 0.5), GEST.Eye.LEFT))
        choice = SimpleNamespace(
            item=SimpleNamespace(effect=None, mode=None, action=ACT.Action.RESUME)
        )
        rig.controller.take_choice(choice, None, rig.clock.now())
        self.assertTrue(rig.safety.selection_armed, "the menu RESUME did not resume")
        self.assertIs(rig.controller.ui_mode, ACT.UiMode.CURSOR)
        rig.frame()
        self.assertEqual(rig.sends, [], "a wink from before the resume clicked")

    def test_a_wink_after_the_resume_still_clicks(self) -> None:
        rig = _Rig(start_active=False)
        rig.space = True
        rig.frame()
        rig.space = False
        rig.frame()
        rig.pipeline.winks.append((rig.clock.now(), (0.5, 0.5), GEST.Eye.LEFT))
        rig.frame()
        self.assertEqual(rig.sends, [CK.MOUSEEVENTF_LEFTDOWN, CK.MOUSEEVENTF_LEFTUP])


if __name__ == "__main__":
    unittest.main()
