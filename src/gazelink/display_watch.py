"""Which physical display are we on, and has it changed since calibration?

A gaze model maps normalized coordinates onto one screen.  That map is only
meaningful against the display it was trained on, yet the product read the
display once at launch and never looked again: change the resolution, change
the Windows scaling percentage, or drag the window to another monitor, and
every gaze point kept being scored against a ruler that no longer existed.  No
error, no warning -- output that looks plausible and means nothing.

Two separate questions live here, and conflating them is the bug this module
removes:

* **Identity** -- *is this the same physical monitor?*  The obvious key is
  wrong.  ``QScreen.name()`` on Windows is a connector slot such as
  ``DISPLAY1``, not a panel.  Unplug one monitor, plug a different one into the
  same port at the same resolution, and a name-and-resolution key matches, so a
  calibration belonging to another screen is accepted in silence.  Identity
  therefore comes from EDID (manufacturer, model, serial) and carries an
  explicit confidence; only :attr:`DisplayIdentityConfidence.CONFIRMED` may
  auto-match a stored profile.  Size, resolution and connector name are never
  sufficient on their own -- that is precisely the failure being designed out.
* **Geometry** -- *is the drawing surface still the ruler we scored against?*
  Exact equality on :class:`~gazelink.domain.ScreenGeometry`, matching the
  definition already trusted by ``CalibrationStore.load_latest_model``, so
  startup and runtime cannot disagree about what "the same screen" means.

:class:`DisplayGuard` **latches**: once tripped it stays tripped even if the
display reverts.  A session that spanned a change is already suspect, and
silently resuming is the failure mode rather than the recovery.

Coordinate contract, stated once because two spaces already coexist in this
codebase undocumented: ``ScreenGeometry.width_px`` and ``height_px`` are
**logical** pixels in Qt's coordinate space -- the same space
``normalized_to_pixel`` returns and every window draws in.  ``dpi_scale`` is
the only conversion to device pixels (``device = logical * dpi_scale``, as
``eyegestures_run`` computes it).  Nothing here mixes the two.

Nothing in this module imports Qt, so all of it is testable without a display.
The screen readers accept any object exposing the few methods they use, which
a plain fake satisfies in tests.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from gazelink.domain import (
    ContractValidationError,
    ReasonCode,
    ScreenGeometry,
    ScreenOrientation,
)

__all__ = [
    "DisplayChange",
    "DisplayGuard",
    "DisplayIdentity",
    "DisplayIdentityConfidence",
    "DisplayWatcher",
    "OutputFreeze",
    "describe_screen",
    "detect_change",
    "ensure_high_dpi_policy",
    "identify_screen",
    "pick_screen",
]

# EDID fields that a panel leaves unset commonly arrive as one of these rather
# than as an empty string.  Treating them as absent is the whole point: a
# placeholder serial that several units share would otherwise mint a confident
# identity for displays that cannot actually be told apart.
_PLACEHOLDER_EDID_VALUES = frozenset(
    {"0", "00000000", "n/a", "na", "none", "null", "unknown", "unspecified"}
)


class DisplayIdentityConfidence(StrEnum):
    """How firmly a display can be told apart from a different one.

    The distinction is load-bearing: only ``CONFIRMED`` licenses reusing a
    stored calibration without asking the user first.
    """

    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"
    UNIDENTIFIED = "UNIDENTIFIED"


class _ScreenLike(Protocol):
    """The slice of ``QScreen`` this module reads.

    Declared structurally so tests can pass a plain fake and so nothing here
    has to import Qt.
    """

    def name(self) -> str: ...

    def geometry(self) -> Any: ...

    def devicePixelRatio(self) -> float: ...


def _clean(value: object) -> str:
    """Normalize an EDID string, mapping "no evidence" onto the empty string.

    EDID text arrives padded, blank, or filled with a vendor placeholder.  A
    blank or placeholder serial is absence of evidence, not a value to key on,
    so it funnels into the ambiguous path instead of manufacturing a false
    identity.
    """

    if not isinstance(value, str):
        return ""
    text = value.strip().strip("\x00").strip()
    if not text or text.lower() in _PLACEHOLDER_EDID_VALUES:
        return ""
    return text


def _call(obj: object, method: str) -> object:
    """Call an optional accessor, treating absence and failure alike.

    Qt bindings differ in which of these accessors exist, and a missing one
    means what an empty one means: no evidence.  Absorbing the difference here
    keeps every caller from repeating the check.
    """

    function = getattr(obj, method, None)
    if not callable(function):
        return ""
    try:
        return function()
    except Exception:  # noqa: BLE001 - a binding that raises is simply no evidence
        return ""


@dataclass(frozen=True, slots=True)
class DisplayIdentity:
    """A physical monitor, identified as firmly as the hardware permits.

    ``key`` is what gets compared and stored.  It is built only from fields
    that survived :func:`_clean`, so an absent serial cannot silently widen the
    key into something a different panel also matches.
    """

    confidence: DisplayIdentityConfidence
    manufacturer: str = ""
    model: str = ""
    serial: str = ""
    connector: str = ""
    key: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, DisplayIdentityConfidence):
            raise ContractValidationError("confidence must be a DisplayIdentityConfidence")

    @property
    def may_auto_match(self) -> bool:
        """May a stored profile be reused for this display without asking?

        Only for a confirmed identity.  Anything weaker must be verified by the
        user, because the alternative is loading another monitor's calibration
        and eventually moving a cursor with it.
        """

        return self.confidence is DisplayIdentityConfidence.CONFIRMED

    def matches(self, other: DisplayIdentity) -> bool:
        """Is this established to be the same physical display?

        Two ``AMBIGUOUS`` identities with equal keys are deliberately **not** a
        match.  That is the case of two monitors of the same model, which is
        ordinary on a two-screen desk and is exactly what a resolution-based
        key gets wrong.
        """

        if not isinstance(other, DisplayIdentity):
            raise ContractValidationError("other must be a DisplayIdentity")
        if not (self.may_auto_match and other.may_auto_match):
            return False
        return self.key == other.key

    def differs_from(self, other: DisplayIdentity) -> bool:
        """Has the display changed into a *different* one since we looked?

        Deliberately not :meth:`matches`.  That method answers "may I reuse a
        stored calibration?", which demands a confirmed serial; this one asks
        the much weaker question "is this still the same thing as a moment
        ago?", which is answered by comparing whatever evidence exists.

        Conflating the two froze gaze permanently on every laptop: a panel with
        no EDID is never ``CONFIRMED``, so ``not matches()`` was true on the
        first poll and the guard latched although nothing had changed.

        Known limit, stated rather than hidden: two *unidentified* displays both
        key to the empty string, so swapping one for the other is invisible here
        and is caught only if the geometry differs.  That is exactly why
        :attr:`may_auto_match` refuses to load a stored profile for them.
        """

        if not isinstance(other, DisplayIdentity):
            raise ContractValidationError("other must be a DisplayIdentity")
        return self.key != other.key

    def describe(self) -> str:
        """A short phrase naming this display for a person."""

        if self.confidence is DisplayIdentityConfidence.UNIDENTIFIED:
            return f"unidentified display on {self.connector or 'an unknown connector'}"
        label = " ".join(part for part in (self.manufacturer, self.model) if part)
        label = label or self.connector or "display"
        if self.confidence is DisplayIdentityConfidence.AMBIGUOUS:
            return f"{label} (no serial reported; indistinguishable from an identical unit)"
        return label

    def to_dict(self) -> dict[str, str]:
        return {
            "confidence": self.confidence.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial": self.serial,
            "connector": self.connector,
            "key": self.key,
        }


def identify_screen(screen: object) -> DisplayIdentity:
    """Read EDID identity from a ``QScreen``-like object.

    Qt exposes the EDID fields directly (``manufacturer``, ``model`` and
    ``serialNumber``), so no raw EDID parsing and no new dependency is needed.
    Any of them may be missing or empty -- on laptop panels and behind some
    KVM switches all three are -- and each absence downgrades the confidence
    rather than being papered over.
    """

    manufacturer = _clean(_call(screen, "manufacturer"))
    model = _clean(_call(screen, "model"))
    serial = _clean(_call(screen, "serialNumber"))
    connector = _clean(_call(screen, "name"))

    if serial and (manufacturer or model):
        return DisplayIdentity(
            confidence=DisplayIdentityConfidence.CONFIRMED,
            manufacturer=manufacturer,
            model=model,
            serial=serial,
            connector=connector,
            key=f"{manufacturer}|{model}|{serial}",
        )
    if manufacturer or model:
        return DisplayIdentity(
            confidence=DisplayIdentityConfidence.AMBIGUOUS,
            manufacturer=manufacturer,
            model=model,
            connector=connector,
            key=f"{manufacturer}|{model}|",
        )
    return DisplayIdentity(
        confidence=DisplayIdentityConfidence.UNIDENTIFIED,
        connector=connector,
    )


def describe_screen(screen: _ScreenLike) -> ScreenGeometry:
    """The one place a ``QScreen`` becomes a :class:`ScreenGeometry`.

    Five windows each carried their own copy of this conversion, which is how
    they came to disagree about which display they were scoring against.  One
    function means one rule.

    The returned sizes are logical pixels; see the module docstring.
    """

    rectangle = screen.geometry()
    width = int(rectangle.width())
    height = int(rectangle.height())
    return ScreenGeometry(
        screen_id=_clean(_call(screen, "name")) or "primary",
        width_px=width,
        height_px=height,
        dpi_scale=float(screen.devicePixelRatio()),
        orientation=ScreenOrientation.LANDSCAPE if width >= height else ScreenOrientation.PORTRAIT,
    )


@dataclass(frozen=True, slots=True)
class DisplayChange:
    """What changed about the display, in terms a person can act on."""

    expected_geometry: ScreenGeometry
    actual_geometry: ScreenGeometry
    expected_identity: DisplayIdentity | None = None
    actual_identity: DisplayIdentity | None = None
    identity_changed: bool = False
    display_lost: bool = False

    @property
    def reason_code(self) -> ReasonCode:
        return ReasonCode.DISPLAY_CHANGED

    @property
    def message(self) -> str:
        """One sentence naming the change and why gaze stopped."""

        if self.display_lost:
            return (
                "The display is no longer available. Gaze is frozen because there "
                "is no screen to place it on."
            )
        if self.identity_changed and self.expected_identity and self.actual_identity:
            return (
                f"The display changed: calibrated on {self.expected_identity.describe()}, "
                f"now on {self.actual_identity.describe()}. Gaze is frozen because the "
                "calibration belongs to a different screen."
            )
        expected = self.expected_geometry
        actual = self.actual_geometry
        details: list[str] = []
        if (expected.width_px, expected.height_px) != (actual.width_px, actual.height_px):
            details.append(
                f"{expected.width_px}x{expected.height_px} to "
                f"{actual.width_px}x{actual.height_px} px"
            )
        if expected.dpi_scale != actual.dpi_scale:
            details.append(f"scaling {expected.dpi_scale:.2f}x to {actual.dpi_scale:.2f}x")
        if expected.orientation is not actual.orientation:
            details.append(
                f"{expected.orientation.value.lower()} to {actual.orientation.value.lower()}"
            )
        if expected.screen_id != actual.screen_id:
            details.append(f"{expected.screen_id} to {actual.screen_id}")
        changed = "; ".join(details) or "the display configuration"
        return (
            f"The display changed ({changed}). Gaze is frozen because every point would "
            "be scored against a screen that no longer exists."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reason_code": self.reason_code.value,
            "identity_changed": self.identity_changed,
            "display_lost": self.display_lost,
            "expected_geometry": self.expected_geometry.to_dict(),
            "actual_geometry": self.actual_geometry.to_dict(),
            "expected_identity": (
                None if self.expected_identity is None else self.expected_identity.to_dict()
            ),
            "actual_identity": (
                None if self.actual_identity is None else self.actual_identity.to_dict()
            ),
        }


def detect_change(
    expected_geometry: ScreenGeometry,
    actual_geometry: ScreenGeometry,
    *,
    expected_identity: DisplayIdentity | None = None,
    actual_identity: DisplayIdentity | None = None,
) -> DisplayChange | None:
    """Has the display we calibrated against been replaced or reconfigured?

    Geometry is compared by exact equality: a one-pixel or one-DPI-step
    difference already invalidates a trained model, so there is no tolerance
    worth tuning.  Identity, when both sides are known, is compared through
    :meth:`DisplayIdentity.matches`, which refuses to confirm anything weaker
    than an EDID serial -- so an unidentifiable display reads as changed and
    the caller has to ask rather than assume.
    """

    if not isinstance(expected_geometry, ScreenGeometry):
        raise ContractValidationError("expected_geometry must be a ScreenGeometry")
    if not isinstance(actual_geometry, ScreenGeometry):
        raise ContractValidationError("actual_geometry must be a ScreenGeometry")

    identity_changed = False
    if expected_identity is not None and actual_identity is not None:
        identity_changed = expected_identity.differs_from(actual_identity)

    if not identity_changed and expected_geometry == actual_geometry:
        return None
    return DisplayChange(
        expected_geometry=expected_geometry,
        actual_geometry=actual_geometry,
        expected_identity=expected_identity,
        actual_identity=actual_identity,
        identity_changed=identity_changed,
    )


class DisplayGuard:
    """Holds the display we calibrated against and latches when it changes.

    Latching is the safety property.  A monitor that changes and changes back
    must not silently resume: the session that spanned the change is already
    suspect, and a user who cannot use their hands has no way to notice that
    gaze quietly became wrong again.  Clearing the latch is explicit and
    belongs to a recalibration, never to a signal handler.
    """

    def __init__(
        self,
        expected_geometry: ScreenGeometry,
        *,
        expected_identity: DisplayIdentity | None = None,
    ) -> None:
        self._tripped: DisplayChange | None = None
        self._set_expected(expected_geometry, expected_identity)

    def _set_expected(self, geometry: ScreenGeometry, identity: DisplayIdentity | None) -> None:
        if not isinstance(geometry, ScreenGeometry):
            raise ContractValidationError("expected_geometry must be a ScreenGeometry")
        if identity is not None and not isinstance(identity, DisplayIdentity):
            raise ContractValidationError("expected_identity must be a DisplayIdentity")
        self._expected_geometry = geometry
        self._expected_identity = identity

    @property
    def expected_geometry(self) -> ScreenGeometry:
        return self._expected_geometry

    @property
    def expected_identity(self) -> DisplayIdentity | None:
        return self._expected_identity

    @property
    def tripped(self) -> DisplayChange | None:
        """The change that tripped this guard, or ``None`` while healthy."""

        return self._tripped

    @property
    def is_tripped(self) -> bool:
        return self._tripped is not None

    def check(
        self,
        actual_geometry: ScreenGeometry,
        *,
        actual_identity: DisplayIdentity | None = None,
    ) -> DisplayChange | None:
        """Compare the live display and latch on the first disagreement.

        Returns the latched change on every later call, including calls where
        the display has gone back to matching.  Callers can therefore poll this
        from a timer without having to remember anything themselves.
        """

        if self._tripped is not None:
            return self._tripped
        change = detect_change(
            self._expected_geometry,
            actual_geometry,
            expected_identity=self._expected_identity,
            actual_identity=actual_identity,
        )
        if change is not None:
            self._tripped = change
        return self._tripped

    def trip_display_lost(self) -> DisplayChange:
        """Latch because the display went away entirely.

        Qt hands back ``None`` for the screen when a monitor is unplugged, so
        there is no geometry to compare and :meth:`check` has nothing to work
        with.  That is still unambiguously a display change, and the most
        abrupt one: continuing to emit gaze for a screen that is gone is the
        worst version of the failure this class exists to prevent.
        """

        if self._tripped is None:
            self._tripped = DisplayChange(
                expected_geometry=self._expected_geometry,
                actual_geometry=self._expected_geometry,
                expected_identity=self._expected_identity,
                display_lost=True,
            )
        return self._tripped

    def reset(
        self,
        expected_geometry: ScreenGeometry,
        *,
        expected_identity: DisplayIdentity | None = None,
    ) -> None:
        """Adopt a new display as the expected one and clear the latch.

        Only a deliberate recalibration should call this.  It is kept separate
        from :meth:`check` so that no automatic path can quietly re-arm a guard
        that tripped for a real reason.
        """

        self._set_expected(expected_geometry, expected_identity)
        self._tripped = None


def pick_screen(application: Any, screen_name: str | None = None) -> Any:
    """The display to open a full-screen window on.

    Every screen in this package used to take ``primaryScreen()`` and nothing
    else, which is wrong in two ways.  A user whose working monitor is the
    secondary one could never calibrate it; and, worse, the recovery flow
    offers "calibrate for this screen" from a window the user may have dragged
    onto another display -- so accepting the primary would calibrate a monitor
    they are not looking at, and tell them it had done what they asked.

    ``screen_name`` is matched against ``QScreen.name()``.  A name that no
    longer matches anything falls back to the primary and says so, because the
    monitor may genuinely have been unplugged between the request and this
    call.
    """

    screens = list(application.screens()) if hasattr(application, "screens") else []
    primary = application.primaryScreen()
    if screen_name:
        for screen in screens:
            if _clean(_call(screen, "name")) == screen_name:
                return screen
        print(f"Display {screen_name!r} is no longer available; using the primary display instead.")
    return primary


class OutputFreeze:
    """Stops everything a screen is drawing once its display has changed.

    Each window has its own set of moving parts -- a frame timer, gaze dots, a
    calibration target, a prediction overlay -- and each had its own hand
    written freeze.  That duplication produced the same two bugs in several
    places at once, so the behaviour lives here instead:

    * **Every action runs on every call.**  The first version returned early
      once a flag was set, which meant a Restart could re-arm the timer and
      nothing would ever stop it again: the screen span forever, drawing
      nothing and explaining nothing.  Being idempotent in *effect* is not the
      same as skipping the work.
    * **Announce once.**  :meth:`engage` returns ``True`` only the first time,
      so a caller can show its message without repeating it every frame.

    Actions are plain callables so this stays Qt-free and testable.  An action
    that raises must not stop the others from running: a half-frozen screen
    still showing a stale marker is the failure being prevented.
    """

    def __init__(self, *actions: Any) -> None:
        for action in actions:
            if not callable(action):
                raise ContractValidationError("every freeze action must be callable")
        self._actions = tuple(actions)
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def engage(self) -> bool:
        """Stop and clear everything. ``True`` the first time only."""

        for action in self._actions:
            with suppress(Exception):
                action()
        first = not self._engaged
        self._engaged = True
        return first


def ensure_high_dpi_policy() -> None:
    """Stop Qt rounding ``devicePixelRatio`` differently per monitor.

    Must run before the first ``QApplication`` is constructed; Qt ignores it
    afterwards, so every entry point calls it immediately before creating one.

    Without ``PassThrough``, Qt snaps the scale factor to a rounded value that
    can differ between two monitors set to the same Windows scaling percentage.
    ``dpi_scale`` is part of the geometry equality key, so that rounding would
    manufacture *false* display-change trips -- freezing a session for no
    reason, which trains people to ignore the warning that matters.

    This is necessary but not sufficient: the written coordinate contract in
    this module's docstring is what actually keeps logical and device pixels
    apart.
    """

    from PySide6.QtCore import Qt  # noqa: PLC0415
    from PySide6.QtGui import QGuiApplication  # noqa: PLC0415

    if QGuiApplication.instance() is not None:
        # Too late to matter, and re-setting it would be a silent no-op that
        # reads like it worked.
        return
    with suppress(AttributeError, RuntimeError):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )


class DisplayWatcher:
    """Binds a :class:`DisplayGuard` to a live Qt window.

    Qt is imported inside the methods, matching the rest of the package, so
    importing this module still costs nothing on a headless machine and the
    pure logic above stays testable without a display server.

    Two detection paths on purpose.  The signals catch a change promptly, but
    which of them a given Windows build actually emits varies -- ``screenAdded``
    and ``primaryScreenChanged`` in particular are unreliable across versions.
    :meth:`poll` therefore re-checks from the ordinary frame timer, so a missed
    signal delays the freeze by one frame instead of losing it entirely.  A
    missed freeze is the failure that matters; a duplicate one costs nothing
    because the guard latches.

    ``current_screen`` is a callable rather than a fixed ``QScreen`` because the
    window can be dragged to another display, and the screen we must compare is
    the one the window is on *now*.
    """

    def __init__(
        self,
        guard: DisplayGuard,
        current_screen: Any,
        on_change: Any,
    ) -> None:
        if not isinstance(guard, DisplayGuard):
            raise ContractValidationError("guard must be a DisplayGuard")
        if not callable(current_screen):
            raise ContractValidationError("current_screen must be callable")
        if not callable(on_change):
            raise ContractValidationError("on_change must be callable")
        self._guard = guard
        self._current_screen = current_screen
        self._on_change = on_change
        self._connections: list[Any] = []
        self._notified = False

    @property
    def guard(self) -> DisplayGuard:
        return self._guard

    def poll(self) -> DisplayChange | None:
        """Re-check the live display; call this from the frame timer.

        Safe to call at any rate: the guard latches, and ``on_change`` fires
        exactly once no matter how many paths detect the same change.
        """

        screen = self._current_screen()
        if screen is None:
            # An unplugged monitor: Qt has no screen left to describe, but this
            # is a display change and one of the most abrupt, so it must freeze
            # here rather than be waited out. Returning the bare latch instead
            # also skipped the notification, leaving gaze running silently.
            change: DisplayChange | None = self._guard.trip_display_lost()
        else:
            change = self._guard.check(
                describe_screen(screen), actual_identity=identify_screen(screen)
            )
        if change is not None and not self._notified:
            self._notified = True
            self._on_change(change)
        return change

    def attach(self, widget: Any = None) -> None:
        """Subscribe to every display signal this Qt build exposes.

        Each connection is attempted independently: bindings differ in which
        signals exist, and one missing signal must not cost us the others.
        """

        from PySide6.QtGui import QGuiApplication  # noqa: PLC0415

        application = QGuiApplication.instance()
        if application is not None:
            for signal_name in ("screenAdded", "screenRemoved", "primaryScreenChanged"):
                self._connect(getattr(application, signal_name, None))

        screen = self._current_screen()
        if screen is not None:
            for signal_name in (
                "geometryChanged",
                "physicalDotsPerInchChanged",
                "orientationChanged",
            ):
                self._connect(getattr(screen, signal_name, None))

        if widget is not None:
            handle = getattr(widget, "windowHandle", None)
            window = handle() if callable(handle) else None
            if window is not None:
                # Fires when the window is dragged onto another display, which
                # no application-level signal reports.
                self._connect(getattr(window, "screenChanged", None))

    def detach(self) -> None:
        """Drop every subscription. Safe to call more than once."""

        for signal in self._connections:
            # Already gone is fine: the C++ object may have outlived its Python
            # wrapper, or Qt may have torn the connection down first.
            with suppress(RuntimeError, TypeError):
                signal.disconnect(self._on_signal)
        self._connections.clear()

    def _connect(self, signal: Any) -> None:
        if signal is None:
            return
        try:
            signal.connect(self._on_signal)
        except (RuntimeError, TypeError):
            return
        self._connections.append(signal)

    def _on_signal(self, *_: Any) -> None:
        self.poll()
