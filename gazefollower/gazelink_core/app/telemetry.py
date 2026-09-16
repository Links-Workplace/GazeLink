"""Counters, per-frame diagnostics and the end-of-session report (ARCH-01 F).

``SessionTelemetry`` is the one owner of what a live session counts. It is
written by the interaction controller and the executor, read by the presenter,
and printed only after every release has run: a report is never part of
cleanup, and a failing report cannot undo a release (TECHNICAL_SPEC 4.5).

The counter names and report lines are exactly those the live view has always
printed; ``tests/golden/live_trace.json`` holds them to that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gazelink_core.interaction import gesture as GEST
from gazelink_core.interaction import menu as MENU


def _new_tally() -> dict[str, int]:
    return {
        "clicks": 0,
        "doubles": 0,
        # Every wink the detector produced, before any of the reasons one
        # might not become a click. Without it, "nothing happened" cannot be
        # told apart from "the wink was seen and then dropped", and the two
        # need opposite fixes.
        "winks": 0,
        "winks_with_no_aim": 0,
        # Scrolling. Counted separately from clicks throughout: "the wheel
        # turned" and "a click happened" are different outcomes and the report
        # exists to tell them apart.
        "notches": 0,
        "scroll_sessions": 0,
        "scroll_cancelled": 0,
        "winks_while_scrolling": 0,
        "winks_dropped_at_mode_edge": 0,
        "suppressed_while_paused": 0,
        "cancelled": 0,
        # The new capabilities, each counted where it can actually fail.
        "right_clicks": 0,
        "menu_opened": 0,
        "menu_choices": 0,
        # Long closes that arrived with the menu already open. Counted rather
        # than silently ignored: on this rig most of them are not gestures at
        # all, they are the operator looking UP at a tile (section 57), and
        # the number says how often that signal is being produced.
        "menu_close_ignored": 0,
        "winks_in_menu": 0,
        "keys_chosen": 0,
        "chars_typed": 0,
        "commands": 0,
        # The window being typed into changed underneath the person. Counted,
        # because it is the one keyboard failure they cannot see the cause of.
        "keyboard_target_lost": 0,
        # A wink that arrived describing a moment that had passed. Its own
        # number: "the wink was never seen" and "the wink was seen too late"
        # need opposite fixes.
        "winks_stale": 0,
    }


def ratio_summary(values: list[tuple[float, float]], rule: GEST.WinkConfig) -> str:
    """How close the eyes came to the wink rule over a whole session.

    A session that recorded no winks has to be able to say WHY: the eye never
    closed far enough, the other eye came with it, or the rule was never the
    problem. Without this the only report was "0 winks", which says none of
    those three.
    """

    if not values:
        return "no frames"
    probe = GEST.RightWinkDetector(rule)
    # Rows are (left, right) or (left, right, pitch): the pitch was added later
    # and every earlier caller still passes the pair.
    right = [row[1] for row in values]
    left = [row[0] for row in values]
    deep = min(right)
    matching = sum(1 for row in values if probe.looks_like_a_wink(row[0], row[1]))
    at_deepest = min(values, key=lambda row: row[1])
    return (
        f"{len(values)} frames; right reached {deep:.3f} (rule needs under {rule.shut_ratio}), "
        f"left was {at_deepest[0]:.3f} there (needs {rule.asymmetry}x the right, so over "
        f"{at_deepest[1] * rule.asymmetry:.3f}); left reached {min(left):.3f}; "
        f"{matching} frames matched the rule"
    )


def pitch_summary(
    values: list[tuple[float, float, float | None]], rule: GEST.WinkConfig, bands: int = 5
) -> list[str]:
    """Does the wink rule pass or fail depending on where the head is?

    Reported live: raising the chin freezes the pointer but produces no click,
    and lowering it toward the lens makes the same wink work. That is a
    testable claim, because eye openness here is polygon AREA in px^2 -- a
    PROJECTED area, which shrinks as the eye is seen more edge-on -- while the
    gate normalises it against a baseline that moves at 0.02 a frame and only
    while the ratio is already above 0.55. If lifting the chin pushes both
    eyes under that, they read as shut together, the baseline stops recovering
    and the asymmetry rule cannot pass however hard the person winks.

    So: split the session by head pitch and show, per band, how deep the right
    eye got and how often the rule was satisfied. If the matching frames sit
    in one band and the rest sit in another, the camera angle is the story.
    """

    known = [row for row in values if len(row) > 2 and row[2] is not None]
    if not known:
        return ["  head pitch             : no head pose on any frame"]
    probe = GEST.RightWinkDetector(rule)
    pitches = [row[2] for row in known]
    lo, hi = min(pitches), max(pitches)
    lines = [
        f"  head pitch             : {len(known)} frames with a pose, "
        f"{lo:+.3f} to {hi:+.3f} (higher = chin up)",
    ]
    matched = [row[2] for row in known if probe.looks_like_a_wink(row[0], row[1])]
    if matched:
        ordered = sorted(matched)
        lines.append(
            f"    frames matching the rule: {len(matched)}, pitch "
            f"{ordered[0]:+.3f} to {ordered[-1]:+.3f}, median {ordered[len(ordered) // 2]:+.3f}"
        )
    else:
        lines.append("    frames matching the rule: none, so there is no pitch to compare")
    if hi - lo < 1e-6:
        lines.append("    the head never moved enough to split into bands")
        return lines
    width = (hi - lo) / bands
    lines.append(
        f"    {'pitch band':>18} {'frames':>7} {'deepest R':>10} "
        f"{'median R':>9} {'matched':>8}"
    )
    for b in range(bands):
        low = lo + b * width
        high = hi if b == bands - 1 else low + width
        rows = [(row[0], row[1]) for row in known if low <= row[2] <= high]
        if not rows:
            continue
        rights = sorted(rv for _lv, rv in rows)
        hits = sum(1 for lv, rv in rows if probe.looks_like_a_wink(lv, rv))
        lines.append(
            f"    {low:+.3f}..{high:+.3f} {len(rows):>7} {rights[0]:>10.3f} "
            f"{rights[len(rights) // 2]:>9.3f} {hits:>8}"
        )
    return lines


def ratio_watcher(config: GEST.WinkConfig | None = None) -> Any:
    """A per-frame view of what the wink rule makes of the current eyes.

    Exists so a live view can SHOW it. Reported live: 'I winked' against a
    session that recorded zero winks, with nothing on screen or in the log
    that could say whether the signal ever came near the rule.
    """

    detector = GEST.RightWinkDetector(config)

    def matches(state: Any) -> bool:
        ratio = getattr(state, "openness_ratio", None)
        return bool(ratio is not None and detector.looks_like_a_wink(*ratio))

    return matches


@dataclass
class SessionTelemetry:
    """What happened in a session, counted where it can fail."""

    tally: dict[str, int] = field(default_factory=_new_tally)
    # Every frame's ratios, so a session that recorded no winks can still say
    # how close the signal came. Two floats and a pitch per frame, nothing else.
    ratios_seen: list[tuple[float, float, float | None]] = field(default_factory=list)
    # Where the gaze actually went while scroll mode was on.
    seen_in: dict[str, int] = field(default_factory=dict)

    def count(self, name: str, n: int = 1) -> None:
        self.tally[name] += n

    def say(self, line: str) -> None:
        """Operator-facing line, printed at the moment it happens."""

        print(line)

    def print_report(self, session: Any) -> None:
        """The end-of-session report. ``session`` exposes the components read."""

        tally = self.tally
        safety = session.safety
        controller = session.controller
        executor = session.executor
        options = session.options
        print("\nsession:")
        print(f"  mode at the end        : {safety.label}")
        print(f"  toggled by             : {options.toggle_by}")
        print(
            f"  clicks                 : {tally['clicks']} "
            f"(double {tally['doubles']}, right {tally['right_clicks']})"
        )
        print(f"  click type at the end  : {controller.board.click_type}")
        print(f"  winks detected         : {tally['winks']}")
        print(f"  winks with nowhere to go: {tally['winks_with_no_aim']}")
        print(f"  winks while paused     : {tally['suppressed_while_paused']}")
        print(f"  wink action            : {options.wink_click}")
        print(f"  wink rule              : {session.wink_cfg}")
        print(
            f"  scrolling              : {tally['notches']} notches over "
            f"{tally['scroll_sessions']} entries"
        )
        print(f"  scroll rule            : {controller.scroll_cfg}")
        print(f"  winks while scrolling  : {tally['winks_while_scrolling']} (none clicked)")
        print(f"  winks while the menu was open: {tally['winks_in_menu']} (none clicked)")
        print(
            f"  winks dropped at a change of mode: "
            f"{tally['winks_dropped_at_mode_edge']}"
        )
        # The menu and the keyboard, each counted where it can fail. "I
        # looked at the tile and nothing happened" and "the tile never saw
        # me" are the same sentence from the person and two numbers here.
        print(
            f"  menu                   : opened {tally['menu_opened']}, "
            f"chose {tally['menu_choices']}, commands sent {tally['commands']}"
        )
        print(
            f"  long closes while it was already open: "
            f"{tally['menu_close_ignored']} (ignored; see section 57 on chin-up)"
        )
        print(f"  menu tiles             : {controller.board.summary()}")
        for warning in MENU.layout_warnings():
            print(f"    {warning}")
        print(
            f"  keyboard               : {tally['keys_chosen']} keys chosen, "
            f"{tally['chars_typed']} characters typed, "
            f"{tally['keyboard_target_lost']} times the target window changed"
        )
        print(f"  winks that arrived too late: {tally['winks_stale']}")
        print(
            f"  winks dropped mid-frame by a mode change: "
            f"{getattr(session.pipeline, 'winks_dropped_mid_frame', 0)}"
        )
        print(f"  scanning               : {controller.keyboard.summary()}")
        print(f"  key adapter            : {executor.keys.summary()}")
        print(f"  hebrew font            : {getattr(session.display, 'font_name', 'headless')}")
        print(f"  action router          : {executor.router.summary()}")
        print(f"  scroll adapter         : {executor.scroller.summary()}")
        looked = sum(self.seen_in.values())
        if looked:
            where = ", ".join(
                f"{name} {count} ({100 * count / looked:.0f}%)"
                for name, count in self.seen_in.items()
                if count
            )
            print(f"  while scrolling, the gaze was in: {where}")
        else:
            print("  while scrolling, the gaze was in: scroll mode was never entered")
        print(f"  eye ratios             : {ratio_summary(self.ratios_seen, session.wink_cfg)}")
        for line in pitch_summary(self.ratios_seen, session.wink_cfg):
            print(line)
        print(f"  overlay                : {session.display.overlay}")
        print(f"  cursor                 : {executor.cursor.summary()}")
        print(f"  click adapter          : {executor.clicker.summary()}")
