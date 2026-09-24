"""The pygame window every live tool draws into (ARCH-01 stage D).

Moved verbatim from ``gf_record``: a display adapter. It draws state and
reports keys; it predicts nothing, detects nothing and emits no OS input.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gazelink_core.calibration import schema as S
from gazelink_core.gaze.visibility import OVERLAY_STALE_S, visible_overlay_point, visible_point
from gazelink_core.tracking.gazefollower_library import _library_dir
from gazelink_core.ui.prompt_layout import prompt_layout

HEBREW_FONTS = ("segoeui.ttf", "arial.ttf", "tahoma.ttf")

# Hebrew, including the presentation forms. Used only to decide whether a
# string needs reversing for display.
_HEBREW_RANGE = ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))


def has_hebrew(text: str) -> bool:
    return any(any(lo <= ord(c) <= hi for lo, hi in _HEBREW_RANGE) for c in text)


def rtl(text: str) -> str:
    """Reverse a Hebrew string so a left-to-right renderer draws it right-to-left.

    pygame draws glyphs in the order it is given them and knows nothing about
    direction, so Hebrew comes out backwards unless the string is reversed
    first. Strings with no Hebrew are returned untouched.

    LIMITATION, stated rather than hidden: this is a whole-string reversal and
    not a bidirectional algorithm, so a line mixing Hebrew and English -- or
    Hebrew and a number -- will have the Latin run backwards. There is no bidi
    implementation in this project and adding one is not part of this work.
    It affects the DISPLAY only: what ``gf_keys`` sends is the original string
    in logical order, so the text arriving in the application is correct
    either way.
    """

    return text[::-1] if has_hebrew(text) else text


def hebrew_font(pygame: Any, size: int) -> tuple[Any, str]:
    """A font that can draw Hebrew, and the name of whichever one was found.

    Returns the built-in font as a last resort WITH a name that says it cannot
    draw Hebrew, because a caller printing the name is the only warning the
    operator gets.
    """

    import os  # noqa: PLC0415

    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in HEBREW_FONTS:
        candidate = folder / name
        try:
            if candidate.exists():
                return pygame.font.Font(str(candidate), size), name
        except Exception:  # noqa: BLE001 - try the next one; a missing font is not fatal
            continue
    return (
        pygame.font.Font(None, size),
        "pygame built-in (LATIN ONLY - Hebrew will draw as empty boxes)",
    )


class Display:
    """pygame fullscreen on the CHOSEN monitor; white like the library's UI.

    ``origin`` is the monitor's position in the virtual desktop. On a second
    screen it is not (0, 0), and a window opened without it lands on the
    primary display -- where the targets would be drawn on one monitor while
    the geometry describes another.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        headless: bool,
        origin: tuple[int, int] = (0, 0),
        overlay_available: bool = False,
        click_through: bool = False,
        show_unfiltered_overlay: bool = False,
        overlay_stale_s: float = OVERLAY_STALE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.headless = headless
        self.width, self.height = width, height
        self.origin = origin
        self._overlay_available = overlay_available
        self._show_unfiltered_overlay = show_unfiltered_overlay
        self._overlay_stale_s = overlay_stale_s
        self._clock = clock
        if headless:
            return
        import os  # noqa: PLC0415

        # SDL reads this at video-subsystem init, so it must be set first.
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{origin[0]},{origin[1]}"
        import pygame  # noqa: PLC0415

        pygame.init()
        try:
            pygame.mixer.init()
        except Exception:  # noqa: BLE001 - sound is a courtesy
            pass
        self.pg = pygame
        flags = pygame.NOFRAME if origin != (0, 0) else pygame.FULLSCREEN
        if click_through:
            # Borderless rather than FULLSCREEN: an exclusive fullscreen
            # window is not a thing Windows will let clicks fall through.
            flags = pygame.NOFRAME
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("GAZELINK - GazeFollower recording")
        self.overlay = None
        if click_through:
            from gazelink_core.platform import overlay as OV  # noqa: PLC0415

            self.overlay = OV.make_click_through(pygame.display.get_wm_info()["window"])
            self.transparent_key = OV.TRANSPARENT_KEY
        self.font = pygame.font.Font(None, 44)
        self.big = pygame.font.Font(None, 64)
        # A SECOND pair, used only by the screens that show Hebrew. The
        # existing two are left exactly as they are: every protocol screen in
        # this project was laid out against their metrics, and a TrueType face
        # at the same nominal size is a different height -- CLAUDE.md 8 says
        # not to silently replace something that works.
        self.hebrew, self.font_name = hebrew_font(pygame, 34)
        self.hebrew_big, _ = hebrew_font(pygame, 48)
        res = Path(_library_dir()) / "res"
        self.dot = None
        self.beep = None
        try:
            self.dot = pygame.transform.smoothscale(
                pygame.image.load(str(res / "image" / "dot.png")), (70, 70)
            )
        except Exception:  # noqa: BLE001
            self.dot = None
        try:
            self.beep = pygame.mixer.Sound(str(res / "audio" / "beep.wav"))
        except Exception:  # noqa: BLE001
            self.beep = None

    def poll_escape(self) -> bool:
        if self.headless:
            return False
        for event in self.pg.event.get():
            if event.type == self.pg.KEYDOWN and event.key == self.pg.K_ESCAPE:
                return True
        return False

    def poll_keys(self) -> set[str]:
        """Every key pressed since the last call, by name.

        Separate from :meth:`poll_escape` because that one drains the queue
        and reports only one key, so a caller that needs a second key cannot
        use both. Screens that only care about Esc keep using poll_escape.
        """

        if self.headless:
            return set()
        names = {self.pg.K_ESCAPE: "escape", self.pg.K_SPACE: "space"}
        pressed: set[str] = set()
        for event in self.pg.event.get():
            if event.type == self.pg.KEYDOWN and event.key in names:
                pressed.add(names[event.key])
        return pressed

    def wait_for_key(self, lines: Sequence[str], *, timeout_s: float | None = None) -> str:
        """Hold the instruction on screen until the operator is ready.

        Returns "go" on Space or Enter, "abort" on Esc, "timeout" if a
        timeout was given and expired. A countdown used to do this job, which
        meant the instruction vanished before it could be read; a protocol
        that starts before the person knows what it asks for produces data
        about their confusion rather than about their gaze.

        Headless runs have nobody to press a key, so they proceed at once.
        """

        if self.headless:
            return "go"
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        prompt = list(lines) + ["", "Press SPACE or ENTER to start   (Esc to abort)"]
        # Drain anything queued before the prompt appeared, so a stray key
        # press during the previous protocol cannot skip this one.
        self.pg.event.clear()
        while True:
            # Draw BEFORE polling: otherwise a key already in flight ends the
            # wait on the first pass and the instruction is never shown at all,
            # which is the failure this screen exists to prevent.
            self.draw_message(prompt)
            for event in self.pg.event.get():
                if event.type == self.pg.KEYDOWN:
                    if event.key == self.pg.K_ESCAPE:
                        return "abort"
                    if event.key in (self.pg.K_SPACE, self.pg.K_RETURN, self.pg.K_KP_ENTER):
                        return "go"
                elif event.type == self.pg.QUIT:
                    return "abort"
            if deadline is not None and time.monotonic() > deadline:
                return "timeout"
            time.sleep(0.02)

    def play_beep(self) -> None:
        if not self.headless and self.beep is not None:
            self.beep.play()

    def draw_message(self, lines: Sequence[str]) -> None:
        if self.headless:
            return
        self.screen.fill((255, 255, 255))
        y = self.height // 2 - 40 * len(lines)
        for line in lines:
            surf = self.big.render(line, True, (20, 20, 20))
            self.screen.blit(surf, (self.width // 2 - surf.get_width() // 2, y))
            y += 80
        self.pg.display.flip()

    def draw_target(self, state: Any, hud: Sequence[str]) -> None:
        """Draw the target, plus what the system currently sees.

        Shown for EVERY frame of EVERY protocol, calibration points included:
        a live dot following (or failing to follow) the target is a sanity
        check no post-hoc number can substitute for, and hiding it during
        calibration would exempt exactly the phase most worth watching.
        """

        if self.headless:
            return
        self.screen.fill((255, 255, 255))
        target = state.target
        if target is not None:
            cx = int(round(target.x * self.width))
            cy = int(round(target.y * self.height))
            if self.dot is not None:
                self.screen.blit(self.dot, (cx - 35, cy - 35))
            else:
                self.pg.draw.circle(self.screen, (30, 30, 30), (cx, cy), 24)
                self.pg.draw.circle(self.screen, (255, 255, 255), (cx, cy), 6)
            if state.protocol in S.CALIBRATION_PROTOCOLS and state.phase in (
                S.PHASE_COLLECT,
                S.PHASE_WAIT,
            ):
                label = self.font.render(str(state.progress), True, (255, 255, 255))
                self.screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))
        raw = visible_point(
            state.raw_norm, state.frame_updated_s, self._clock(), self._overlay_stale_s
        )
        self._draw_live_point(raw, (255, 160, 0), radius=6)  # raw model output: orange
        overlay = visible_overlay_point(state, self._clock(), self._overlay_stale_s)
        if self._show_unfiltered_overlay and overlay is not None:
            self._draw_live_point(state.overlay_raw_norm, (150, 70, 180), radius=7, cross=True)
        self._draw_live_point(
            overlay, (30, 110, 255), radius=10, cross=True
        )  # filtered calibrated: blue
        y = 20
        for line in hud:
            surf = self.font.render(line, True, (60, 60, 60))
            self.screen.blit(surf, (20, y))
            y += 40
        legend_y = self.height - 60
        self.pg.draw.circle(self.screen, (255, 160, 0), (30, legend_y), 6)
        self.screen.blit(
            self.font.render("raw model output (no calibration)", True, (90, 90, 90)),
            (46, legend_y - 12),
        )
        self.pg.draw.circle(self.screen, (30, 110, 255), (30, legend_y + 28), 6)
        self.screen.blit(
            self.font.render(
                "filtered calibrated"
                if self._overlay_available
                else "calibrated: no --overlay-model loaded",
                True,
                (90, 90, 90),
            ),
            (46, legend_y + 16),
        )
        self.pg.display.flip()

    def draw_live(
        self,
        point: tuple[float, float] | None,
        raw_model: tuple[float, float] | None,
        unfiltered: tuple[float, float] | None,
        hud: Sequence[str],
        *,
        tracking: bool,
        zones: Sequence[Any] = (),
        active_zone: str | None = None,
    ) -> None:
        """The free-running view: no target on screen, just what is reported.

        "Not tracking" is drawn as an explicit banner rather than as an absent
        dot.  With nothing on screen, a frozen prediction and a lost face look
        identical, and the whole point of watching the dot is to be able to
        tell those apart.

        ``zones`` are drawn as OUTLINES, never filled.  This window sits over
        the desktop as a colour-keyed hole, so a filled rectangle would hide
        the thing the person is trying to read -- and the whole purpose of a
        scroll band is to be looked at while reading past it.
        """

        if self.headless:
            return
        # In overlay mode the background is the colour Windows was told to
        # treat as a hole, so the desktop shows through and only the dots and
        # the text are visible. Anywhere else it is the ordinary white sheet.
        overlay = self.overlay is not None and self.overlay.see_through
        self.screen.fill(self.transparent_key if overlay else (255, 255, 255))
        for zone in zones:
            rect = self.pg.Rect(
                int(zone.x0 * self.width),
                int(zone.y0 * self.height),
                int((zone.x1 - zone.x0) * self.width),
                int((zone.y1 - zone.y0) * self.height),
            )
            live = zone.key == active_zone
            # Outline only. Never (255, 0, 255): that is the colour Windows
            # was told to treat as a hole, so a border in it would vanish.
            # Thick and high contrast on purpose. A 3 px grey line over a
            # busy page is invisible, and a control the person cannot find is
            # the same as one that is not there.
            self.pg.draw.rect(
                self.screen,
                (30, 110, 255) if live else (70, 110, 160),
                rect,
                14 if live else 6,
            )
            label = self.font.render(zone.key, True, (20, 20, 20))
            plate = self.pg.Surface((label.get_width() + 16, label.get_height() + 8))
            plate.fill((245, 245, 245))
            self.screen.blit(plate, (rect.centerx - plate.get_width() // 2, rect.centery - 20))
            self.screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - 16))
        self._draw_live_point(raw_model, (255, 160, 0), radius=6)
        if self._show_unfiltered_overlay:
            self._draw_live_point(unfiltered, (150, 70, 180), radius=7, cross=True)
        self._draw_live_point(point, (30, 110, 255), radius=10, cross=True)
        if not tracking:
            banner = self.big.render("tracking lost", True, (200, 40, 40))
            self.screen.blit(banner, (self.width // 2 - banner.get_width() // 2, 60))
        y = 20
        # Over an unknown desktop the text needs its own ground, or it is
        # unreadable exactly when it matters -- the mode line included.
        colour = (20, 20, 20) if overlay else (60, 60, 60)
        for line in hud:
            surf = self.font.render(line, True, colour)
            if overlay:
                plate = self.pg.Surface((surf.get_width() + 16, surf.get_height() + 8))
                plate.fill((245, 245, 245))
                self.screen.blit(plate, (12, y - 4))
            self.screen.blit(surf, (20, y))
            y += 40
        self.pg.display.flip()

    def draw_practice(
        self,
        buttons: Sequence[Any],
        point: tuple[float, float] | None,
        *,
        hovered: str | None,
        progress: float,
        prompt: Sequence[str],
        flash: str | None = None,
        pointer: tuple[float, float] | None = None,
        tracking: bool,
        prompt_anchor: str = "top",
    ) -> None:
        """The dwell practice screen: targets, the gaze point, and a ring.

        Every target is drawn identically. The prompt names the one to look
        at, in text, away from the targets themselves -- nothing about the
        requested target changes its appearance, position or size, because a
        target that stood out would measure attention capture rather than
        whether the gaze can be put where the person intends.

        The ring fills on whichever target the gaze is actually resting on,
        which makes a wrong selection visible while it happens instead of only
        in the report afterwards.
        """

        if self.headless:
            return
        self.screen.fill((250, 250, 250))
        for button in buttons:
            rect = self.pg.Rect(
                int(button.x0 * self.width),
                int(button.y0 * self.height),
                int(button.width * self.width),
                int(button.height * self.height),
            )
            filled = flash == button.key
            self.pg.draw.rect(self.screen, (210, 228, 246) if filled else (232, 232, 236), rect)
            self.pg.draw.rect(self.screen, (120, 130, 140), rect, 3)
            label = self.big.render(button.key, True, (60, 60, 70))
            self.screen.blit(
                label,
                (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2),
            )
            if hovered == button.key and progress > 0.0:
                self._draw_progress_ring(rect.centerx, rect.centery + 140, progress)
        # The POINTER, drawn separately from the gaze point and in a
        # different colour. Without it a simulated run shows nothing at all
        # where the cursor would be -- the real one does not move in
        # simulation, so "the cursor is not there" was indistinguishable from
        # "the cursor is broken". The two dots also make the smoothing and the
        # dead zone visible: the pointer trails the gaze on purpose.
        self._draw_live_point(pointer, (220, 90, 30), radius=18)
        self._draw_live_point(point, (30, 110, 255), radius=12, cross=True)
        if not tracking:
            banner = self.big.render("tracking lost", True, (200, 40, 40))
            self.screen.blit(banner, (self.width // 2 - banner.get_width() // 2, 30))
        # The instruction goes in the CENTRE, in the dead zone between the
        # targets -- never in a corner. Reading it is itself a gaze, and a
        # corner prompt puts that gaze inside whichever target is nearest:
        # measured, every wrong activation fired deep inside the wrong target
        # (x 0.300-0.392 or 0.579-0.641), never at a boundary, and the target
        # nearest the top-left prompt was chosen twice as often as the other.
        # Whoever reaches 900 ms first wins, so where the person must look to
        # READ the task decides the answer before they can act on it.
        #
        # That centre is dead space only for the two-column layouts. In the
        # nine-zone grid the top centre sits straight above UP-C, closer than
        # the measured vertical bias, so ``prompt_anchor="sides"`` puts the
        # prompt outside the working band instead (see ``prompt_layout``).
        # "top" stays the default and is drawn exactly as before.
        font = self.big if prompt_anchor == "top" else self.font
        sizes = [font.size(line) for line in prompt]
        rects = prompt_layout(sizes, prompt_anchor, self.width, self.height)
        per_copy = len(prompt)
        for i, (x, y, _w, _h) in enumerate(rects):
            line = prompt[i % per_copy] if per_copy else ""
            surf = font.render(line, True, (40, 40, 50))
            self.screen.blit(surf, (x, y))
        self.pg.display.flip()

    def _plate(self, surf: Any, x: int, y: int, pad: int = 8) -> None:
        """Draw text on its own opaque ground, so it is readable over anything.

        The board screens sit over the person's own work as a colour-keyed
        hole. Anything drawn without a ground is unreadable exactly when it
        matters -- and a FULL-screen fill is not the answer either: it would
        hide the page the person is about to type into or scroll.
        """

        plate = self.pg.Surface((surf.get_width() + pad * 2, surf.get_height() + pad))
        plate.fill((245, 245, 245))
        self.screen.blit(plate, (x - pad, y - pad // 2))
        self.screen.blit(surf, (x, y))

    def _centre_lines(self, lines: Sequence[str], top: int) -> int:
        """Status text down the MIDDLE of the screen. Never in a corner.

        Measured on the practice screen and written down there: reading a
        corner prompt is itself a gaze, it lands inside whichever target is
        nearest, and the target beside the prompt was chosen twice as often as
        the other. The middle of these boards is the dead zone between the
        tiles, which is the one place a gaze can rest without choosing
        anything.
        """

        y = top
        for line in lines:
            surf = self.hebrew.render(rtl(line), True, (20, 20, 20))
            self._plate(surf, self.width // 2 - surf.get_width() // 2, y)
            y += surf.get_height() + 14
        return y

    def draw_board(
        self,
        buttons: Sequence[Any],
        labels: Mapping[str, str],
        *,
        hovered: str | None,
        progress: float,
        centre: Sequence[str] = (),
        point: tuple[float, float] | None = None,
        tracking: bool = True,
        ready: bool = True,
        not_ready_line: str = "הבט למרכז כדי להפעיל את האריחים",
    ) -> None:
        """Four tiles over the desktop, and a ring on whichever is filling.

        Opaque only where something is drawn: the tiles, the labels and the
        centre text get their own ground, and everything between them stays
        the colour key so the page underneath is still visible. A full fill
        like ``draw_practice`` would hide the thing the person opened the menu
        to do something about.

        ``ready`` is false until the caller's arming rule is satisfied, and it
        is SHOWN rather than merely enforced: tiles the person can see but
        cannot yet choose, with nothing saying why, is the same experience as
        tiles that are broken. ``not_ready_line`` says WHICH rule, because the
        menu arms on the dead centre and the persistent bar arms on content --
        telling a bar user to look at the centre would be an instruction that
        does not work.
        """

        if self.headless:
            return
        overlay = self.overlay is not None and self.overlay.see_through
        self.screen.fill(self.transparent_key if overlay else (255, 255, 255))
        for button in buttons:
            rect = self.pg.Rect(
                int(button.x0 * self.width),
                int(button.y0 * self.height),
                int(button.width * self.width),
                int(button.height * self.height),
            )
            live = hovered == button.key and ready
            plate = self.pg.Surface((rect.width, rect.height))
            plate.fill((214, 231, 247) if live else (242, 242, 245))
            self.screen.blit(plate, rect.topleft)
            self.pg.draw.rect(
                self.screen,
                (30, 110, 255) if live else (120, 130, 140),
                rect,
                10 if live else 4,
            )
            label = self.hebrew_big.render(
                rtl(labels.get(button.key, button.key)), True, (30, 30, 40)
            )
            self.screen.blit(
                label,
                (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2),
            )
            if live and progress > 0.0:
                self._draw_progress_ring(rect.centerx, rect.bottom - 70, progress, radius=40)
        top = self.height // 2 - 60
        lines = list(centre)
        if not ready:
            lines = [*lines, not_ready_line]
        self._centre_lines(lines, top)
        self._draw_live_point(point, (30, 110, 255), radius=10, cross=True)
        if not tracking:
            banner = self.big.render("tracking lost", True, (200, 40, 40))
            self.screen.blit(banner, (self.width // 2 - banner.get_width() // 2, 60))
        self.pg.display.flip()

    def draw_scan(
        self,
        items: Sequence[str],
        index: int,
        *,
        centre: Sequence[str] = (),
        typed: str = "",
        parked: bool = False,
        point: tuple[float, float] | None = None,
        tracking: bool = True,
    ) -> None:
        """The scanning strip: every choice in a row, one of them highlighted.

        Laid out RIGHT to left, because the labels are Hebrew: the first item
        the sweep reaches is the first one a Hebrew reader's eye reaches.

        Drawn across the middle of the screen on purpose. The gaze does not
        choose anything here -- a wink does -- so the one thing the strip must
        do is be READ, and the middle is where the eyes already are.
        """

        if self.headless:
            return
        overlay = self.overlay is not None and self.overlay.see_through
        self.screen.fill(self.transparent_key if overlay else (255, 255, 255))
        rendered = [
            self.hebrew_big.render(rtl(text), True, (30, 30, 40)) for text in items
        ]
        pad, gap = 18, 14
        widths = [surf.get_width() + pad * 2 for surf in rendered]
        total = sum(widths) + gap * max(0, len(widths) - 1)
        height = (max((s.get_height() for s in rendered), default=40)) + pad
        y = self.height // 2 - height // 2
        # Right to left: the strip starts at the right edge of its own span.
        x = self.width // 2 + total // 2
        for i, surf in enumerate(rendered):
            width = widths[i]
            x -= width
            rect = self.pg.Rect(x, y, width, height)
            live = i == index and not parked
            plate = self.pg.Surface((rect.width, rect.height))
            plate.fill((214, 231, 247) if live else (242, 242, 245))
            self.screen.blit(plate, rect.topleft)
            if live:
                self.pg.draw.rect(self.screen, (30, 110, 255), rect, 8)
            self.screen.blit(
                surf,
                (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2),
            )
            x -= gap
        below = y + height + 24
        if typed:
            echo = self.hebrew.render(rtl(typed), True, (20, 60, 20))
            self._plate(echo, self.width // 2 - echo.get_width() // 2, below)
            below += echo.get_height() + 20
        self._centre_lines(centre, below)
        self._draw_live_point(point, (30, 110, 255), radius=10, cross=True)
        if not tracking:
            banner = self.big.render("tracking lost", True, (200, 40, 40))
            self.screen.blit(banner, (self.width // 2 - banner.get_width() // 2, 60))
        self.pg.display.flip()

    def _draw_progress_ring(self, cx: int, cy: int, fraction: float, radius: int = 60) -> None:
        """Dwell progress, as an arc filling clockwise from the top.

        CLAUDE.md 4.5 requires visible feedback for dwell progress: without
        it, waiting for an activation and waiting for nothing look identical.
        """

        self.pg.draw.circle(self.screen, (200, 205, 210), (cx, cy), radius, 6)
        span = max(0.0, min(1.0, fraction)) * 2.0 * math.pi
        if span <= 0.0:
            return
        rect = self.pg.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
        start = math.pi / 2.0
        self.pg.draw.arc(self.screen, (30, 110, 255), rect, start - span, start, 8)

    def _draw_live_point(
        self,
        point_norm: tuple[float, float] | None,
        colour: tuple[int, int, int],
        *,
        radius: int,
        cross: bool = False,
    ) -> None:
        """One moving dot: what the system currently reports, on or off screen.

        A point outside [0, 1] is drawn clamped to the edge with a ring, so
        "predicting off the display" is visibly different from "not tracking
        at all" (nothing drawn) rather than silently invisible.
        """

        if point_norm is None:
            return
        x, y = point_norm
        off_screen = not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        cx = int(round(min(1.0, max(0.0, x)) * self.width))
        cy = int(round(min(1.0, max(0.0, y)) * self.height))
        if cross:
            self.pg.draw.line(self.screen, colour, (cx - radius, cy), (cx + radius, cy), 3)
            self.pg.draw.line(self.screen, colour, (cx, cy - radius), (cx, cy + radius), 3)
        else:
            self.pg.draw.circle(self.screen, colour, (cx, cy), radius, 0 if not off_screen else 2)
        if off_screen:
            self.pg.draw.circle(self.screen, (200, 40, 40), (cx, cy), radius + 6, 2)

    def close(self) -> None:
        if self.headless:
            return
        try:
            self.pg.quit()
        except Exception:  # noqa: BLE001
            pass

def sleep_with_escape(display: Display, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if display.poll_escape():
            return True
        time.sleep(0.02)
    return False


# The recorder's historical name.
_sleep_with_escape = sleep_with_escape
