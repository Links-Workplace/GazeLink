"""The display we score against must be the display we calibrated on.

No Qt, no camera, no display server: a fake screen supplies every value these
tests need.  The properties asserted here are safety properties rather than
convenience ones -- each corresponds to a way a person could end up driving a
cursor with another screen's calibration, or to a freeze they could not escape
from.
"""

from __future__ import annotations

import pytest

from gazelink.display_watch import (
    DisplayGuard,
    DisplayIdentity,
    DisplayIdentityConfidence,
    DisplayWatcher,
    OutputFreeze,
    describe_screen,
    detect_change,
    identify_screen,
    pick_screen,
)
from gazelink.domain import (
    ContractValidationError,
    GazePoint,
    ReasonCode,
    ScreenGeometry,
    ScreenOrientation,
)
from gazelink.gaze_engine import normalized_to_pixel

pytestmark = pytest.mark.unit


class _Rect:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class _FakeScreen:
    """The slice of ``QScreen`` the module reads, and nothing more.

    ``missing`` drops accessors entirely rather than returning empty strings,
    because Qt bindings differ in which of them exist and the code must treat
    an absent accessor exactly like an empty one.
    """

    def __init__(
        self,
        *,
        name: str = "DISPLAY1",
        width: int = 1920,
        height: int = 1080,
        ratio: float = 1.0,
        manufacturer: str = "ACME",
        model: str = "U2720Q",
        serial: str = "SN-0001",
        missing: tuple[str, ...] = (),
        raising: tuple[str, ...] = (),
    ) -> None:
        self._values = {
            "name": name,
            "manufacturer": manufacturer,
            "model": model,
            "serialNumber": serial,
        }
        self._width = width
        self._height = height
        self._ratio = ratio
        self._missing = set(missing)
        self._raising = set(raising)

    def __getattr__(self, item: str) -> object:
        if item in {"name", "manufacturer", "model", "serialNumber"}:
            if item in self._missing:
                raise AttributeError(item)
            if item in self._raising:

                def _raise() -> str:
                    raise RuntimeError("binding failure")

                return _raise
            return lambda: self._values[item]
        raise AttributeError(item)

    def geometry(self) -> _Rect:
        return _Rect(self._width, self._height)

    def devicePixelRatio(self) -> float:
        return self._ratio


def _geometry(**overrides: object) -> ScreenGeometry:
    values: dict[str, object] = {
        "screen_id": "DISPLAY1",
        "width_px": 1920,
        "height_px": 1080,
        "dpi_scale": 1.0,
    }
    values.update(overrides)
    return ScreenGeometry(**values)  # type: ignore[arg-type]


def _identity(**overrides: object) -> DisplayIdentity:
    values: dict[str, object] = {
        "confidence": DisplayIdentityConfidence.CONFIRMED,
        "manufacturer": "ACME",
        "model": "U2720Q",
        "serial": "SN-0001",
        "connector": "DISPLAY1",
        "key": "ACME|U2720Q|SN-0001",
    }
    values.update(overrides)
    return DisplayIdentity(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# describe_screen: one conversion, replacing five hand-written copies
# --------------------------------------------------------------------------


def test_describe_screen_matches_a_hand_built_geometry() -> None:
    """The shared helper cannot drift from the call sites it replaced."""

    screen = _FakeScreen(name="DISPLAY2", width=2560, height=1440, ratio=1.25)
    assert describe_screen(screen) == ScreenGeometry(
        screen_id="DISPLAY2",
        width_px=2560,
        height_px=1440,
        dpi_scale=1.25,
        orientation=ScreenOrientation.LANDSCAPE,
    )


def test_describe_screen_reports_portrait_when_taller_than_wide() -> None:
    screen = _FakeScreen(width=1080, height=1920)
    assert describe_screen(screen).orientation is ScreenOrientation.PORTRAIT


def test_describe_screen_falls_back_when_the_name_is_unusable() -> None:
    """``ScreenGeometry`` rejects an empty ``screen_id``; a blank name must not crash."""

    assert describe_screen(_FakeScreen(name="   ")).screen_id == "primary"


# --------------------------------------------------------------------------
# Identity: the hole that exists today
# --------------------------------------------------------------------------


def test_same_resolution_and_connector_but_different_serial_do_not_match() -> None:
    """Swapping monitors on one port must not reuse the old calibration.

    This is the failure the pixel-and-name key allows today: both displays
    report ``DISPLAY1`` at 1920x1080, so an equality check on geometry alone
    accepts a calibration trained on the other panel.
    """

    first = identify_screen(_FakeScreen(serial="SN-0001"))
    second = identify_screen(_FakeScreen(serial="SN-0002"))

    assert first.confidence is DisplayIdentityConfidence.CONFIRMED
    assert second.confidence is DisplayIdentityConfidence.CONFIRMED
    assert not first.matches(second)

    change = detect_change(
        _geometry(),
        _geometry(),
        expected_identity=first,
        actual_identity=second,
    )
    assert change is not None
    assert change.identity_changed
    assert change.reason_code is ReasonCode.DISPLAY_CHANGED


@pytest.mark.parametrize("serial", ["", "   ", "0", "00000000", "unknown", "N/A"])
def test_blank_or_placeholder_serial_is_ambiguous_not_confirmed(serial: str) -> None:
    """A placeholder serial is absence of evidence, not an identity."""

    identity = identify_screen(_FakeScreen(serial=serial))
    assert identity.confidence is DisplayIdentityConfidence.AMBIGUOUS
    assert not identity.may_auto_match


def test_two_identical_models_without_serials_never_auto_match() -> None:
    """Two of the same monitor is an ordinary desk, and must not be guessed at."""

    left = identify_screen(_FakeScreen(name="DISPLAY1", serial=""))
    right = identify_screen(_FakeScreen(name="DISPLAY2", serial=""))

    assert left.key == right.key  # indistinguishable by every field available
    assert not left.matches(right)
    assert not left.matches(left)  # not even against itself: still unverified


def test_missing_edid_entirely_is_unidentified() -> None:
    """Laptop panels and some KVMs report nothing; that must be explicit."""

    identity = identify_screen(_FakeScreen(missing=("manufacturer", "model", "serialNumber")))
    assert identity.confidence is DisplayIdentityConfidence.UNIDENTIFIED
    assert not identity.may_auto_match
    assert identity.key == ""


def test_a_raising_accessor_is_treated_as_no_evidence() -> None:
    """A binding that throws must degrade the identity, not crash the app."""

    identity = identify_screen(_FakeScreen(raising=("serialNumber",)))
    assert identity.confidence is DisplayIdentityConfidence.AMBIGUOUS


def test_confirmed_identity_matches_itself() -> None:
    identity = identify_screen(_FakeScreen())
    assert identity.matches(identify_screen(_FakeScreen()))


def test_matches_rejects_a_non_identity() -> None:
    with pytest.raises(ContractValidationError):
        _identity().matches(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# detect_change: geometry
# --------------------------------------------------------------------------


def test_identical_geometry_is_not_a_change() -> None:
    assert detect_change(_geometry(), _geometry()) is None


@pytest.mark.parametrize(
    "override",
    [
        {"width_px": 1921},
        {"height_px": 1081},
        {"dpi_scale": 1.25},
        {"screen_id": "DISPLAY2"},
        {"width_px": 1080, "height_px": 1920, "orientation": ScreenOrientation.PORTRAIT},
    ],
    ids=["width", "height", "dpi", "screen_id", "orientation"],
)
def test_each_geometry_field_alone_trips_detection(override: dict[str, object]) -> None:
    change = detect_change(_geometry(), _geometry(**override))
    assert change is not None
    assert change.message


def test_detect_change_rejects_non_geometry_arguments() -> None:
    with pytest.raises(ContractValidationError):
        detect_change(object(), _geometry())  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError):
        detect_change(_geometry(), object())  # type: ignore[arg-type]


def test_message_names_both_sides_of_a_resolution_change() -> None:
    change = detect_change(_geometry(), _geometry(width_px=1280, height_px=720))
    assert change is not None
    assert "1920x1080" in change.message
    assert "1280x720" in change.message


# --------------------------------------------------------------------------
# DisplayGuard: the latch
# --------------------------------------------------------------------------


def test_guard_is_quiet_while_the_display_is_unchanged() -> None:
    guard = DisplayGuard(_geometry())
    assert guard.check(_geometry()) is None
    assert not guard.is_tripped


def test_guard_stays_tripped_after_the_display_reverts() -> None:
    """The safety property: changing back must not silently resume.

    A user who cannot use their hands has no way to notice that gaze quietly
    became valid again against a session that already spanned a change.
    """

    guard = DisplayGuard(_geometry())
    tripped = guard.check(_geometry(width_px=1280, height_px=720))
    assert tripped is not None

    still_tripped = guard.check(_geometry())
    assert still_tripped is tripped
    assert guard.is_tripped


def test_guard_latches_on_identity_even_when_geometry_is_identical() -> None:
    guard = DisplayGuard(_geometry(), expected_identity=_identity())
    change = guard.check(
        _geometry(),
        actual_identity=_identity(serial="SN-0002", key="ACME|U2720Q|SN-0002"),
    )
    assert change is not None
    assert change.identity_changed


def test_guard_keeps_the_first_change_not_the_latest() -> None:
    """The first divergence is the one that invalidated the session."""

    guard = DisplayGuard(_geometry())
    first = guard.check(_geometry(width_px=1280, height_px=720))
    second = guard.check(_geometry(width_px=800, height_px=600))
    assert first is second
    assert second is not None
    assert second.actual_geometry.width_px == 1280


def test_reset_adopts_a_new_display_and_clears_the_latch() -> None:
    """Only a deliberate recalibration re-arms the guard."""

    guard = DisplayGuard(_geometry())
    guard.check(_geometry(width_px=1280, height_px=720))
    assert guard.is_tripped

    guard.reset(_geometry(width_px=1280, height_px=720))
    assert not guard.is_tripped
    assert guard.check(_geometry(width_px=1280, height_px=720)) is None


def test_guard_rejects_a_non_geometry_expectation() -> None:
    with pytest.raises(ContractValidationError):
        DisplayGuard(object())  # type: ignore[arg-type]


def test_guard_rejects_a_non_identity_expectation() -> None:
    with pytest.raises(ContractValidationError):
        DisplayGuard(_geometry(), expected_identity=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Coordinate contract: logical pixels, one documented conversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1.0, 1.25, 1.5, 2.0])
def test_scored_pixels_are_logical_and_independent_of_dpi_scale(ratio: float) -> None:
    """``dpi_scale`` must not leak into the scored coordinate.

    ``normalized_to_pixel`` returns logical pixels -- the space the windows
    draw in.  ``dpi_scale`` is the sole conversion to device pixels, and the
    two must not silently blend, which is the drift this pins down.
    """

    geometry = _geometry(dpi_scale=ratio)
    scored = normalized_to_pixel(GazePoint(0.5, 0.5), geometry)

    assert scored == normalized_to_pixel(GazePoint(0.5, 0.5), _geometry(dpi_scale=1.0))
    device_x = scored.x_px * geometry.dpi_scale
    assert device_x == pytest.approx(scored.x_px * ratio)


def test_describe_screen_geometry_is_logical_not_device_pixels() -> None:
    """A 2x display reports its logical size, matching what Qt lays out in."""

    screen = _FakeScreen(width=1920, height=1080, ratio=2.0)
    geometry = describe_screen(screen)
    assert (geometry.width_px, geometry.height_px) == (1920, 1080)
    assert geometry.dpi_scale == 2.0


# --------------------------------------------------------------------------
# DisplayWatcher: the timer backstop, and notifying exactly once
# --------------------------------------------------------------------------


def test_watcher_polls_the_screen_the_window_is_on_now() -> None:
    """Dragging the window to another display must be noticed.

    No application-level signal reports this, which is why the watcher asks
    for the current screen on every poll rather than holding one.
    """

    current = _FakeScreen(name="DISPLAY1")
    seen: list[object] = []
    watcher = DisplayWatcher(
        DisplayGuard(describe_screen(current), expected_identity=identify_screen(current)),
        lambda: current,
        seen.append,
    )

    assert watcher.poll() is None

    current = _FakeScreen(name="DISPLAY2", width=1280, height=720, serial="SN-0002")
    change = watcher.poll()

    assert change is not None
    assert len(seen) == 1
    assert seen[0] is change


def test_watcher_notifies_once_however_many_paths_detect_it() -> None:
    """Signals and the timer both fire; the user must not see two freezes."""

    current = _FakeScreen()
    seen: list[object] = []
    watcher = DisplayWatcher(DisplayGuard(describe_screen(current)), lambda: current, seen.append)

    current = _FakeScreen(width=1280, height=720)
    for _ in range(5):
        watcher.poll()

    assert len(seen) == 1


def test_watcher_treats_a_vanished_screen_as_a_change_not_a_shrug() -> None:
    """An unplugged monitor makes Qt hand back ``None``: freeze, do not wait.

    This originally asserted that ``poll`` returned ``None`` here, which
    encoded the bug -- gaze kept running against geometry that no longer
    existed.  The safe answer is a latched change.
    """

    watcher = DisplayWatcher(DisplayGuard(_geometry()), lambda: None, lambda _: None)
    change = watcher.poll()
    assert change is not None
    assert change.display_lost


def test_watcher_keeps_reporting_the_latch_after_the_screen_goes_away() -> None:
    screen: object = _FakeScreen()
    watcher = DisplayWatcher(DisplayGuard(_geometry()), lambda: screen, lambda _: None)
    screen = _FakeScreen(width=800, height=600)
    tripped = watcher.poll()
    assert tripped is not None

    screen = None
    assert watcher.poll() is tripped


def test_watcher_rejects_bad_wiring() -> None:
    """A non-callable here would silently disable the whole escape hatch."""

    with pytest.raises(ContractValidationError):
        DisplayWatcher(object(), lambda: None, lambda _: None)  # type: ignore[arg-type]
    with pytest.raises(ContractValidationError):
        DisplayWatcher(DisplayGuard(_geometry()), object(), lambda _: None)
    with pytest.raises(ContractValidationError):
        DisplayWatcher(DisplayGuard(_geometry()), lambda: None, object())


def test_detach_is_safe_before_attach_and_twice() -> None:
    watcher = DisplayWatcher(DisplayGuard(_geometry()), lambda: None, lambda _: None)
    watcher.detach()
    watcher.detach()


# --------------------------------------------------------------------------
# Regressions: two paths that froze gaze wrongly, or failed to freeze it
# --------------------------------------------------------------------------


def test_a_display_without_edid_does_not_trip_against_itself() -> None:
    """An ordinary laptop must not freeze on the first poll.

    Regression: change detection used ``matches()``, which demands a confirmed
    serial.  A panel with no EDID is never confirmed, so "has it changed?"
    answered yes immediately and gaze froze permanently on the most common
    hardware there is, with nothing having changed at all.
    """

    screen = _FakeScreen(missing=("manufacturer", "model", "serialNumber"))
    identity = identify_screen(screen)
    assert identity.confidence is DisplayIdentityConfidence.UNIDENTIFIED

    guard = DisplayGuard(describe_screen(screen), expected_identity=identity)
    assert guard.check(describe_screen(screen), actual_identity=identity) is None
    assert not guard.is_tripped


def test_an_ambiguous_display_does_not_trip_against_itself() -> None:
    """Same regression, one tier up: EDID present but no serial."""

    screen = _FakeScreen(serial="")
    identity = identify_screen(screen)
    assert identity.confidence is DisplayIdentityConfidence.AMBIGUOUS

    guard = DisplayGuard(describe_screen(screen), expected_identity=identity)
    assert guard.check(describe_screen(screen), actual_identity=identity) is None


def test_reuse_still_refuses_what_change_detection_permits() -> None:
    """The two questions must stay separate, not be collapsed back together.

    Unchanged is not the same as safe to reuse: an unidentified display is
    still not allowed to load a stored calibration.
    """

    identity = identify_screen(_FakeScreen(missing=("manufacturer", "model", "serialNumber")))
    assert not identity.differs_from(identity)  # unchanged
    assert not identity.may_auto_match  # but still unverified
    assert not identity.matches(identity)


def test_a_swapped_monitor_still_trips_after_the_fix() -> None:
    """The fix must not have bought quiet by disabling detection."""

    first = identify_screen(_FakeScreen(serial="SN-0001"))
    second = identify_screen(_FakeScreen(serial="SN-0002"))
    guard = DisplayGuard(_geometry(), expected_identity=first)

    change = guard.check(_geometry(), actual_identity=second)
    assert change is not None
    assert change.identity_changed


def test_an_unplugged_display_freezes_and_notifies() -> None:
    """Regression: a screen Qt reports as ``None`` left gaze running.

    ``poll`` returned the bare latch, which was ``None`` on the first such
    call, so an unplugged monitor produced neither a freeze nor a message and
    gaze carried on against geometry that no longer existed.
    """

    screen: object = _FakeScreen()
    seen: list[object] = []
    watcher = DisplayWatcher(DisplayGuard(describe_screen(screen)), lambda: screen, seen.append)
    assert watcher.poll() is None

    screen = None
    change = watcher.poll()

    assert change is not None
    assert change.display_lost
    assert len(seen) == 1
    assert "no longer available" in change.message


def test_display_lost_latches_like_any_other_change() -> None:
    """Re-plugging must not silently resume; the latch rules apply here too."""

    screen: object = None
    watcher = DisplayWatcher(DisplayGuard(_geometry()), lambda: screen, lambda _: None)
    lost = watcher.poll()
    assert lost is not None

    screen = _FakeScreen()
    assert watcher.poll() is lost


# --------------------------------------------------------------------------
# OutputFreeze: the real class the windows use, not a copy of it
# --------------------------------------------------------------------------


class _Timer:
    def __init__(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def start(self, _interval: int = 0) -> None:
        self.running = True


class _Marker:
    def __init__(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def show(self) -> None:
        self.visible = True


def test_freeze_stops_and_clears_every_part_it_was_given() -> None:
    """A marker left drawn sits at a position computed from the old geometry."""

    timer, target, overlay = _Timer(), _Marker(), _Marker()
    freeze = OutputFreeze(timer.stop, target.hide, overlay.hide)

    assert freeze.engage() is True
    assert not timer.running
    assert not target.visible
    assert not overlay.visible


def test_freeze_reasserts_itself_after_something_restarts_the_timer() -> None:
    """Regression: an early return skipped the work on every later call.

    ``_on_restart`` re-armed the timer, the next tick saw the latched change,
    and the freeze returned immediately without stopping anything -- the screen
    span forever, drawing nothing and explaining nothing.
    """

    timer, target = _Timer(), _Marker()
    freeze = OutputFreeze(timer.stop, target.hide)
    freeze.engage()

    timer.start()  # what Restart does
    target.show()
    freeze.engage()

    assert not timer.running
    assert not target.visible


def test_freeze_reports_first_engagement_only() -> None:
    """So the caller can explain once instead of every frame."""

    freeze = OutputFreeze(lambda: None)
    assert freeze.engage() is True
    assert freeze.engage() is False
    assert freeze.engaged


def test_one_failing_action_does_not_strand_the_others() -> None:
    """A half-frozen screen still showing a stale marker is the failure."""

    target = _Marker()

    def _explode() -> None:
        raise RuntimeError("Qt object already deleted")

    freeze = OutputFreeze(_explode, target.hide)
    freeze.engage()

    assert not target.visible


def test_freeze_rejects_something_that_cannot_be_called() -> None:
    with pytest.raises(ContractValidationError):
        OutputFreeze(object())


def test_every_live_window_freezes_through_this_class() -> None:
    """Guards the wiring, not just the class.

    The previous regression tests re-implemented the freeze inside the test
    file, so reverting the production fix left them green.  This asserts the
    windows actually delegate here, which is what makes the tests above mean
    anything.
    """

    import inspect

    from gazelink import (
        calibration_window,
        feature_check_window,
        gaze_window,
        test_window,
        validation_window,
    )

    for module in (
        gaze_window,
        calibration_window,
        validation_window,
        feature_check_window,
        test_window,
    ):
        source = inspect.getsource(module)
        assert "OutputFreeze(" in source, f"{module.__name__} builds no OutputFreeze"
        assert "self._freeze.engage()" in source, f"{module.__name__} never engages it"


# --------------------------------------------------------------------------
# pick_screen: recalibration must target the display the user is on
# --------------------------------------------------------------------------


class _FakeApplication:
    def __init__(self, screens: tuple[object, ...], primary: object) -> None:
        self._screens = screens
        self._primary = primary

    def screens(self) -> tuple[object, ...]:
        return self._screens

    def primaryScreen(self) -> object:
        return self._primary


def test_pick_screen_returns_the_named_display_not_the_primary() -> None:
    """The bug: "calibrate for this screen" opened on the primary monitor.

    After dragging the window to the second display, calibrating the first one
    and reporting success is worse than refusing -- the user ends up with a
    model for a screen they are not looking at.
    """

    first = _FakeScreen(name="DISPLAY1")
    second = _FakeScreen(name="DISPLAY2", width=2560, height=1440)
    application = _FakeApplication((first, second), primary=first)

    assert pick_screen(application, "DISPLAY2") is second


def test_pick_screen_defaults_to_primary_without_a_name() -> None:
    first = _FakeScreen(name="DISPLAY1")
    second = _FakeScreen(name="DISPLAY2")
    application = _FakeApplication((first, second), primary=first)

    assert pick_screen(application, None) is first


def test_pick_screen_falls_back_when_the_display_was_unplugged() -> None:
    """The monitor can genuinely vanish between the request and the call."""

    first = _FakeScreen(name="DISPLAY1")
    application = _FakeApplication((first,), primary=first)

    assert pick_screen(application, "DISPLAY2") is first


def test_calibration_opens_on_the_requested_display() -> None:
    """Guards the wiring, not just the helper.

    Asserts the calibration entry point takes a target screen and resolves it
    through ``pick_screen`` rather than reaching for the primary display.
    """

    import inspect

    from gazelink import calibration_window

    source = inspect.getsource(calibration_window)
    assert "screen_name" in source
    assert "pick_screen(application, screen_name)" in source
    assert "application.primaryScreen()" not in source


def test_windows_size_themselves_by_the_screen_their_geometry_describes() -> None:
    """The drawing surface and the scoring ruler must be the same display.

    Regression: ``run_guided_calibration`` picked the requested monitor for its
    geometry, but ``show()`` sized the window from ``self._widget.screen()`` --
    the default display. Targets computed for one monitor were then drawn on
    another, which is a worse failure than opening on the wrong screen outright
    because the two disagree while both look correct.

    ``test_window`` already learned this and documents it; these two had not.
    """

    import inspect

    from gazelink import calibration_window, gaze_window

    for module in (calibration_window, gaze_window):
        source = inspect.getsource(module)
        assert "self._widget.setGeometry(self._screen.geometry())" in source, (
            f"{module.__name__} must size itself by the screen it was given"
        )
        assert "screen = self._widget.screen()\n        if screen is not None:" not in source, (
            f"{module.__name__} still sizes itself by whatever display it landed on"
        )
