"""The one file that can press a key, and the two ways it can go wrong.

Nothing here emits a real keystroke: every send is recorded by a fake, and the
adapter calls exactly one sender, so a keystroke arriving at the recorder is
proof the real one was not the one that got it.

The two failures worth testing are not "did a letter arrive".  They are:

* a modifier left DOWN, which turns every later keystroke -- including the
  ones the person makes with their hands -- into a command, with no way to
  release it by looking;
* text arriving in the WRONG WINDOW, silently, because focus moved after the
  keyboard was opened.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gf_keys as K  # noqa: E402

WINDOW = 4242


def _adapter(*, enabled=True, front=None, sender=None):
    sends: list[tuple[int, int, int]] = []
    adapter = K.KeyAdapter(
        enabled=enabled,
        sender=sender or (lambda vk, scan, flags: sends.append((vk, scan, flags))),
        sleep=lambda _s: None,
        foreground=front or (lambda: WINDOW),
    )
    return adapter, sends


class TargetLockTests(unittest.TestCase):
    """The half that was nearly missed.

    "The overlay never steals focus" guarantees only that WE do not take it.
    It says nothing about the original window keeping it: a notification, an
    installer, or another application taking focus would have had the text
    continue into it, silently.
    """

    def test_nothing_is_typed_before_a_target_is_locked(self) -> None:
        adapter, sends = _adapter()
        self.assertFalse(adapter.type_text("שלום", armed=True))
        self.assertEqual(sends, [], "text was sent into whatever happened to be in front")
        self.assertEqual(adapter.refused_no_target, 1)

    def test_a_locked_target_that_has_not_changed_is_not_blocked(self) -> None:
        """The other half. A rule that refused everything would pass the test
        below on its own and would also make the keyboard useless."""

        adapter, sends = _adapter()
        adapter.lock_target()
        self.assertTrue(adapter.type_text("א", armed=True))
        self.assertTrue(sends, "a valid target refused a keystroke")
        self.assertEqual(adapter.chars_typed, 1)

    def test_a_changed_target_stops_the_typing(self) -> None:
        front = {"hwnd": WINDOW}
        adapter, sends = _adapter(front=lambda: front["hwnd"])
        adapter.lock_target()
        self.assertTrue(adapter.type_text("א", armed=True))
        before = len(sends)
        front["hwnd"] = 9999
        self.assertFalse(adapter.type_text("ב", armed=True))
        self.assertEqual(len(sends), before, "text went into a window nobody chose")
        self.assertEqual(adapter.refused_target_changed, 1)
        self.assertTrue(adapter.target_lost)

    def test_it_does_not_resume_on_its_own_when_the_window_comes_back(self) -> None:
        """The person cannot see that focus returned, so a keyboard that
        started typing again would be typing without their knowledge."""

        front = {"hwnd": WINDOW}
        adapter, _sends = _adapter(front=lambda: front["hwnd"])
        adapter.lock_target()
        front["hwnd"] = 9999
        adapter.type_text("א", armed=True)
        front["hwnd"] = WINDOW
        self.assertFalse(
            adapter.type_text("ב", armed=True), "typing resumed with nobody asking"
        )

    def test_locking_again_is_what_brings_it_back(self) -> None:
        front = {"hwnd": WINDOW}
        adapter, _sends = _adapter(front=lambda: front["hwnd"])
        adapter.lock_target()
        front["hwnd"] = 9999
        adapter.type_text("א", armed=True)
        adapter.lock_target()
        self.assertTrue(adapter.type_text("ב", armed=True), "the explicit way back failed")
        self.assertFalse(adapter.target_lost)

    def test_a_changed_target_blocks_commands_too(self) -> None:
        front = {"hwnd": WINDOW}
        adapter, sends = _adapter(front=lambda: front["hwnd"])
        adapter.lock_target()
        front["hwnd"] = 9999
        self.assertFalse(adapter.back(armed=True))
        self.assertEqual(sends, [])


class NothingIsLeftHeldTests(unittest.TestCase):
    def test_a_chord_releases_the_modifier(self) -> None:
        adapter, sends = _adapter()
        adapter.lock_target()
        self.assertTrue(adapter.back(armed=True))
        ups = [flags & K.KEYEVENTF_KEYUP for vk, _scan, flags in sends if vk == K.VK_MENU]
        self.assertIn(K.KEYEVENTF_KEYUP, ups, "Alt was pressed and never released")
        self.assertFalse(adapter.keys_stuck)

    def test_the_modifier_is_released_even_when_the_inner_key_fails(self) -> None:
        """The stuck-Alt case, and the worst thing this module can do."""

        state = {"calls": 0}

        def flaky(vk, scan, flags):
            state["calls"] += 1
            # The Alt down succeeds; the arrow down does not.
            if vk == K.VK_LEFT and not flags & K.KEYEVENTF_KEYUP:
                raise OSError("SendInput sent 0 of 1 events")
            state.setdefault("sent", []).append((vk, scan, flags))

        adapter, _ = _adapter(sender=flaky)
        adapter.lock_target()
        self.assertFalse(adapter.back(armed=True))
        released = [
            (vk, flags)
            for vk, _scan, flags in state["sent"]
            if vk == K.VK_MENU and flags & K.KEYEVENTF_KEYUP
        ]
        self.assertTrue(released, "Alt stayed down after the key under it failed")
        self.assertFalse(adapter.keys_stuck)

    def test_a_key_whose_release_fails_is_reported_and_retried(self) -> None:
        sent: list[tuple[int, int, int]] = []
        fail = {"on": True}

        def flaky(vk, scan, flags):
            if flags & K.KEYEVENTF_KEYUP and fail["on"]:
                raise OSError("no")
            sent.append((vk, scan, flags))

        adapter, _ = _adapter(sender=flaky)
        adapter.lock_target()
        self.assertFalse(adapter.escape(armed=True))
        self.assertTrue(adapter.keys_stuck, "a key was left down with nothing saying so")
        fail["on"] = False
        adapter.release()
        self.assertFalse(adapter.keys_stuck, "the retry did not clear it")
        self.assertTrue(
            any(flags & K.KEYEVENTF_KEYUP for _vk, _scan, flags in sent),
            "the retry never sent an up",
        )

    def test_release_works_whoever_has_focus(self) -> None:
        """Refusing to release because the window changed would leave the
        person's own keyboard unusable, which is worse than typing nothing."""

        front = {"hwnd": WINDOW}
        sent: list[tuple[int, int, int]] = []

        adapter = K.KeyAdapter(
            enabled=True,
            sender=lambda vk, scan, flags: sent.append((vk, scan, flags)),
            sleep=lambda _s: None,
            foreground=lambda: front["hwnd"],
        )
        adapter.lock_target()
        adapter._down(K.VK_MENU)
        front["hwnd"] = 9999
        adapter.release()
        self.assertTrue(
            any(vk == K.VK_MENU and flags & K.KEYEVENTF_KEYUP for vk, _s, flags in sent),
            "Alt was left held because focus had moved",
        )

    def test_the_context_manager_releases_on_the_way_out(self) -> None:
        sent: list[tuple[int, int, int]] = []
        with K.KeyAdapter(
            enabled=True,
            sender=lambda vk, scan, flags: sent.append((vk, scan, flags)),
            sleep=lambda _s: None,
            foreground=lambda: WINDOW,
        ) as adapter:
            adapter.lock_target()
            adapter._down(K.VK_MENU)
        self.assertTrue(any(flags & K.KEYEVENTF_KEYUP for _vk, _s, flags in sent))


class SimulationTests(unittest.TestCase):
    def test_a_disabled_adapter_sends_nothing_and_still_counts(self) -> None:
        sends: list[tuple[int, int, int]] = []
        adapter = K.KeyAdapter(
            enabled=False,
            sender=lambda vk, scan, flags: sends.append((vk, scan, flags)),
            sleep=lambda _s: None,
            foreground=lambda: WINDOW,
        )
        adapter.lock_target()
        self.assertTrue(adapter.type_text("שלום", armed=True))
        self.assertTrue(adapter.back(armed=True))
        self.assertEqual(sends, [], "a simulation reached Windows")
        self.assertEqual(adapter.chars_typed, 4)
        self.assertEqual(adapter.commands_sent, 1)

    def test_an_unarmed_request_sends_nothing(self) -> None:
        adapter, sends = _adapter()
        adapter.lock_target()
        self.assertFalse(adapter.type_text("א", armed=False))
        self.assertFalse(adapter.enter(armed=False))
        self.assertEqual(sends, [])


class UnicodeTests(unittest.TestCase):
    def test_hebrew_is_sent_as_unicode_rather_than_as_a_virtual_key(self) -> None:
        """A virtual key is read through whatever keyboard LAYOUT is active,
        so an aleph sent that way arrives as whatever letter sits on that key.
        Unicode carries the character itself."""

        adapter, sends = _adapter()
        adapter.lock_target()
        adapter.type_text("א", armed=True)
        self.assertTrue(sends)
        for vk, scan, flags in sends:
            self.assertEqual(vk, 0, "a virtual key was used for a letter")
            self.assertTrue(flags & K.KEYEVENTF_UNICODE)
            self.assertEqual(scan, ord("א"))

    def test_a_run_of_characters_is_refused_rather_than_truncated(self) -> None:
        adapter, _ = _adapter()
        adapter.lock_target()
        with self.assertRaises(ValueError):
            adapter.type_text("x" * 500, armed=True)


class SeparationTests(unittest.TestCase):
    def test_this_file_cannot_press_a_pointer_button(self) -> None:
        """The same audit gf_cursor has against click words, in the other
        direction. Two files that can each do the other's job are one file."""

        source = Path(__import__("gf_keys").__file__).read_text(
            encoding="utf-8"
        )
        for forbidden in ("MOUSEEVENTF", "SetCursorPos", "INPUT_MOUSE", "mouseData"):
            self.assertNotIn(forbidden, source, f"{forbidden} appeared in the key adapter")


if __name__ == "__main__":
    unittest.main()


class NativeLayoutTests(unittest.TestCase):
    """The size Windows is told about has to be the size Windows expects.

    ``SendInput`` takes ``sizeof(INPUT)`` and refuses anything else. The union
    must therefore be as large as the BIGGEST member of the real INPUT, not as
    large as the one member this file uses -- and with only the keyboard
    member it came out at 32 bytes against the 40 that 64-bit Windows wants.
    Every real keystroke would have been rejected while every test above
    passed, because they all replace the sender.
    """

    def test_the_input_structure_is_the_size_the_click_path_uses(self) -> None:
        """Compared against ``gf_click``'s rather than against a number.

        That structure has been sending real clicks and real wheel notches on
        this rig for weeks, so it is the one piece of evidence available here
        about what Windows actually accepts."""

        import ctypes  # noqa: PLC0415

        import gf_click as CK  # noqa: PLC0415

        self.assertEqual(ctypes.sizeof(K._Input), ctypes.sizeof(CK._Input))

    def test_the_keyboard_member_still_fits_inside_it(self) -> None:
        import ctypes  # noqa: PLC0415

        self.assertGreaterEqual(
            ctypes.sizeof(K._InputUnion), ctypes.sizeof(K._KeyInput)
        )


class NothingGoesOutOverAHeldKeyTests(unittest.TestCase):
    """A character typed under a held Alt is a command, not a character.

    So a request that arrives while something is still down retries the
    release first and is refused if it cannot be delivered.
    """

    def _stuck(self):
        sent: list[tuple[int, int, int]] = []
        fail = {"on": True}

        def flaky(vk, scan, flags):
            if flags & K.KEYEVENTF_KEYUP and fail["on"]:
                raise OSError("no")
            sent.append((vk, scan, flags))

        adapter, _ = _adapter(sender=flaky)
        adapter.lock_target()
        adapter.escape(armed=True)
        return adapter, sent, fail

    def test_a_request_is_refused_while_a_key_is_still_held(self) -> None:
        adapter, sent, _fail = self._stuck()
        self.assertTrue(adapter.keys_stuck, "the fixture did not get stuck")
        before = len(sent)
        self.assertFalse(adapter.type_text("א", armed=True))
        self.assertEqual(len(sent), before, "text was typed over a held key")
        self.assertEqual(adapter.refused_keys_held, 1)

    def test_it_works_again_once_the_release_gets_through(self) -> None:
        """The other half: refusing for ever would be its own failure."""

        adapter, _sent, fail = self._stuck()
        fail["on"] = False
        self.assertTrue(adapter.type_text("א", armed=True))
        self.assertFalse(adapter.keys_stuck)

    def test_a_chord_that_cannot_release_its_modifier_reports_failure(self) -> None:
        """It returned True with Alt still down, and the loop went on
        accepting requests underneath it."""

        sent: list[tuple[int, int, int]] = []

        def flaky(vk, scan, flags):
            if vk == K.VK_MENU and flags & K.KEYEVENTF_KEYUP:
                raise OSError("no")
            sent.append((vk, scan, flags))

        adapter, _ = _adapter(sender=flaky)
        adapter.lock_target()
        self.assertFalse(adapter.back(armed=True), "a stuck Alt was reported as success")
        self.assertTrue(adapter.keys_stuck)
        self.assertEqual(adapter.commands_sent, 0, "a failed command was counted")

    def test_a_chord_that_releases_cleanly_still_reports_success(self) -> None:
        adapter, _sent = _adapter()
        adapter.lock_target()
        self.assertTrue(adapter.back(armed=True))
        self.assertEqual(adapter.commands_sent, 1)
