"""What the live window shows each frame (ARCH-01 stage F).

``LivePresenter`` reads the session's components and calls the display. It
owns no state that decides anything: it cannot pause, click, predict or detect.
The lines and draw arguments are exactly those the live view has always used
(``tests/golden/live_trace.json`` hashes every draw call).
"""

from __future__ import annotations

from typing import Any

from gazelink_core.interaction import actions as ACT
from gazelink_core.interaction import gesture as GEST
from gazelink_core.interaction import keyboard as KB
from gazelink_core.interaction import menu as MENU


def menu_lines(menu: GEST.RecoveryMenu) -> list[str]:
    if not menu.open:
        return []
    return ["", "MENU  (short close cycles, long close confirms)"] + [
        f"  {'>' if i == menu.index else ' '} {option.label}"
        for i, option in enumerate(menu.options)
    ]


class LivePresenter:
    def __init__(self, *, session: Any) -> None:
        self.session = session

    def render(self, state: Any, view: Any) -> None:
        s = self.session
        controller = s.controller
        safety = s.safety
        executor = s.executor
        tally = s.telemetry.tally
        options = s.options
        settings = s.settings
        fresh = view.fresh
        controlled = safety.controlled
        hud = [
            f"profile {s.profile.name}   frames {state.frames}"
            + (f"   fps {state.fps:.1f}" if state.fps else "   fps --"),
            f"filter {settings.kind.value} fc={settings.one_euro_min_cutoff_hz} "
            f"beta={settings.one_euro_beta_hz_per_px_s}",
            (
                f"{safety.label}   clicks {tally['clicks']}"
                f" (double {tally['doubles']}, right {tally['right_clicks']})"
                + ("   HELD" if executor.pointer_held else "")
                + "   -- Esc stops everything"
                if controlled
                else (
                    f"CURSOR {'PAUSED' if executor.pointer_paused else 'MOVING'}"
                    " -- Esc stops everything"
                    if executor.pointer_enabled
                    else "Esc to stop.  Nothing is recorded and no OS input is sent."
                )
            ),
            (
                (
                    "SCROLLING.  Look UP or DOWN to scroll, at the middle to stop.  "
                    "Close BOTH eyes for the menu."
                    if controller.ui_mode is ACT.UiMode.SCROLL
                    else (
                        f"{'SPACE' if options.toggle_by == 'key' else 'A long close'} switches "
                        f"PAUSED/ACTIVE.  Wink RIGHT to {controller.board.click_type}-click."
                        + ("  Close BOTH eyes for the menu." if options.menu_enabled else "")
                    )
                )
                if controlled
                else "Close your eyes about half a second to open the menu."
            ),
        ]
        # The same readout the practice window has: a wink the signal never saw
        # and a wink that was seen and dropped look identical without it.
        if controlled and state.openness_ratio is not None:
            hud.append(
                f"your eyes  L{state.openness_ratio[0] * 100:.0f}%"
                f"  R{state.openness_ratio[1] * 100:.0f}%"
                + ("   <<< WINK" if s.wink_rule_matches(state) else "")
                + f"   winks {tally['winks']}"
            )
        if controller.notice:
            hud.append(controller.notice)
        hud += menu_lines(controller.recovery_menu)
        display = s.display
        if controller.ui_mode is ACT.UiMode.MENU:
            # Its own screen, not the live view with extra lines: the tiles are
            # OPAQUE where drawn and transparent everywhere else.
            board = controller.board
            display.draw_board(
                board.buttons,
                {item.key: item.label for item in board.items},
                hovered=board.hovered,
                progress=board.progress,
                centre=[
                    f"{'מושהה' if not safety.cursor_enabled else 'פעיל'}"
                    f"   לחיצה: {MENU.CLICK_LABEL[board.click_type]}",
                    "התפריט נשאר פתוח עד שתבחר",
                    "'סגור תפריט' נמצא אחרי שתי לחיצות על 'עוד'   ·   Esc עוצר הכל",
                ],
                point=fresh,
                tracking=fresh is not None,
                ready=board.seeded,
            )
        elif controller.ui_mode is ACT.UiMode.KEYBOARD:
            keyboard = controller.keyboard
            items = (
                [g.label for g in keyboard.groups]
                if keyboard.level is KB.Level.GROUPS
                else [k.label for k in keyboard.keys]
            )
            if keyboard.level is KB.Level.PARKED:
                items = [g.label for g in keyboard.groups]
            display.draw_scan(
                items,
                keyboard.index,
                centre=(
                    ["הסריקה נעצרה - קרוץ כדי להמשיך"]
                    if keyboard.parked
                    else [
                        f"פריסה: {KB.LAYOUT_LABEL[keyboard.layout_name]}"
                        f"   יעד: {executor.keyboard_target_name}",
                        # A refused wink has to SAY it was refused.
                        "מוקדם מדי - חכה שהסימון יופיע ואז קרוץ"
                        if keyboard.refused_recently(s.clock())
                        else "קריצה בוחרת",
                    ]
                ),
                typed=keyboard.typed,
                parked=keyboard.parked,
                point=fresh,
                tracking=fresh is not None,
            )
        else:
            display.draw_live(
                fresh,
                view.raw_fresh,
                view.unfiltered,
                hud,
                tracking=fresh is not None,
                # Shown only in scroll mode: bands the rest of the time would be
                # clutter over whatever the person is reading.
                zones=(
                    [*controller.scroll_bands]
                    if controlled and controller.ui_mode is ACT.UiMode.SCROLL
                    else []
                ),
                active_zone=(
                    controller.repeater.zone if controller.ui_mode is ACT.UiMode.SCROLL else None
                ),
            )
