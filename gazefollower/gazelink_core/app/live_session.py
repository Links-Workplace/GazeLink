"""One live session: build the components, run the loop, shut down (ARCH-01 G).

``LiveSession`` composes and orchestrates. It holds no interaction or safety
decision of its own:

* ``SafetyController``      control mode, face freshness, activation, stopping
* ``InteractionController`` UI mode, menu, keyboard, scroll, winks
* ``ActionExecutor``        the only caller of the input adapters
* ``FramePipeline``         gate, prediction, filter, gestures (camera thread)
* ``LivePresenter``         what is drawn
* ``SessionTelemetry``      counters and the report

Every external dependency comes from the injected ``LiveEnvironment``. Every
acquisition registers its undo in a ``CleanupStack`` the moment it succeeds;
shutdown stops the source and blocks new actions, releases held input, closes
the display and the library, and only then reports (TECHNICAL_SPEC 4.5).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from gazelink_core.app.environment import LiveEnvironment
from gazelink_core.app.lifecycle import CleanupStack
from gazelink_core.app.options import LiveOptions
from gazelink_core.app.telemetry import SessionTelemetry, ratio_watcher
from gazelink_core.calibration import profile as PROF
from gazelink_core.domain.observation import FrameObservation
from gazelink_core.gaze import preflight as PRE
from gazelink_core.interaction import actions as ACT
from gazelink_core.interaction import bar as BAR
from gazelink_core.interaction import keyboard_spatial as SKB
from gazelink_core.interaction import control as CTL
from gazelink_core.interaction import gesture as GEST
from gazelink_core.interaction import keyboard as KB
from gazelink_core.interaction import menu as MENU
from gazelink_core.interaction import scroll as SCR
from gazelink_core.interaction.controller import InteractionController
from gazelink_core.interaction.executor import ActionExecutor
from gazelink_core.interaction.safety import SafetyController
from gazelink_core.platform import click as CK
from gazelink_core.platform import cursor as CUR
from gazelink_core.platform import keys as KEYS
from gazelink_core.platform import real_input as RI
from gazelink_core.platform import screen_check as SC
from gazelink_core.tracking import gazefollower_library as LIB
from gazelink_core.ui.presenter import LivePresenter

# A distinct exit code so a caller can tell "the operator asked to
# recalibrate" from "the view was closed" without parsing stdout.
RECALIBRATE_REQUESTED = 10
LOOP_TICK_S = 0.005
MODEL_REFUSED_PAUSE_S = 3.0


class LiveSession:
    def __init__(self, profile: PROF.Profile, options: LiveOptions, env: LiveEnvironment) -> None:
        self.profile = profile
        self.options = options
        self.env = env
        self.clock = env.clock.now
        self.cleanup = CleanupStack()
        self.telemetry = SessionTelemetry()
        self.session_started = False
        self.input_adapters: list[Any] = []
        self.source: Any = None
        self.pipeline: Any = None
        self.display: Any = None
        self.safety: SafetyController | None = None
        self.executor: ActionExecutor | None = None
        self.controller: InteractionController | None = None
        self.presenter: LivePresenter | None = None
        self.wink_cfg: GEST.WinkConfig | None = None
        self.settings: Any = None
        self.wink_rule_matches: Any = None

    # -- the whole session --------------------------------------------------------

    def run(self) -> int:
        options, env, profile = self.options, self.env, self.profile
        options.validate()
        # Built once and used everywhere, so the readout, the detector and the
        # report cannot end up judging by three different rules.
        wink_cfg = profile.wink_config()
        if options.wink_hold_ms is not None:
            wink_cfg = dataclasses.replace(wink_cfg, hold_ms=options.wink_hold_ms)
        self.wink_cfg = wink_cfg
        monitor = desktop = None
        if options.move_cursor:
            # Two separate gates, on purpose. The first is the operator saying
            # they meant it; the second is the machine agreeing the ruler is
            # trustworthy. Neither substitutes for the other.
            if not options.confirmed:
                raise SystemExit(
                    "--move-cursor moves the REAL Windows pointer. Add --i-mean-it to confirm. "
                    "Without it this refuses to start; nothing is moved."
                )
            _declared, how = env.ensure_dpi_aware()
            print(f"dpi awareness: {how}")
            report = env.check_screen(profile)
            if not report.verified:
                print(SC.format_report(report))
                raise SystemExit(
                    "the screen is not verified, so a gaze fraction cannot be trusted as a pixel. "
                    "M3-00: no real cursor on an unverified ruler. Nothing was moved."
                )
            monitor = env.pick_monitor(None)
            desktop = env.virtual_desktop()

        rig = profile.rig_geometry()
        self.settings = profile.filter_settings()
        model_dir = profile.model_path()
        if not model_dir.exists():
            raise SystemExit(
                f"profile {profile.name!r} points at a model that is not there: {model_dir}"
            )
        model = env.load_model(model_dir)
        print(f"profile {profile.name}: model {model_dir} ({len(model.schema.columns)} columns)")

        if profile.capture_pipeline != PROF.CAPTURE_LIBRARY and (
            options.move_cursor or options.click_by != "off"
        ):
            # The eyelid rule and the click timings were measured on 640x480
            # landmarks. Real input from another capture waits until they are
            # re-checked on it in simulation (TASKS section 62, plan review 2).
            raise SystemExit(
                f"profile {profile.name!r} uses {profile.capture_pipeline!r}: moving the cursor or "
                "clicking is not validated for this capture yet. Run without --move-cursor and "
                "--click-by."
            )
        if options.move_cursor and env.real_input and not RI.is_armed():
            # Enabled adapters on real Windows senders with nothing armed: every
            # press would be refused mid-session. Refused before anything opens.
            raise SystemExit(
                "real OS input is not armed: run through gf_live's command line with "
                "--move-cursor --i-mean-it, or pass a fake environment."
            )

        # The tracking source owns the camera and the library; from here on the
        # session sees FrameObservations only.
        self.source = env.open_source(profile)
        self.cleanup.push("library shutdown", lambda: print(self.source.close()))
        returned: int | None = None
        try:
            returned = self._build_and_run(model, rig, monitor, desktop)
        finally:
            self._shutdown()
        return returned if returned is not None else self._final_summary()

    # -- building ---------------------------------------------------------------------

    def _build_and_run(self, model: Any, rig: Any, monitor: Any, desktop: Any) -> int | None:
        options, env, profile = self.options, self.env, self.profile
        clock = env.clock
        # The eyelid rules come from the PROFILE: they belong to this face and
        # this camera geometry, not to whoever last edited the defaults.
        self.pipeline = env.make_runner(
            model, None, rig, self.settings, wink=self.wink_cfg, gate=profile.gate_config(),
            clock=clock.now,
        )
        # The harmless option is first, so a confirm that fires when it should
        # not costs nothing. The pause sits before recalibration: when the
        # pointer is moving, stopping it is the thing most wanted in a hurry.
        options_menu = [GEST.MenuOption("dismiss", "Keep watching")]
        if options.move_cursor:
            options_menu.append(GEST.MenuOption("pause", "Pause / resume the cursor"))
        options_menu.append(GEST.MenuOption("recalibrate", "Recalibrate for this screen"))
        recovery_menu = GEST.RecoveryMenu(options_menu)
        preflight: list[np.ndarray[Any, Any]] = []
        live = {"on": False}
        pipeline = self.pipeline

        def subscriber(obs: FrameObservation) -> None:
            # After the source is stopped no observation arrives here.
            if not live["on"]:
                if obs.gaze_status and len(preflight) < PRE.PREFLIGHT_MAX_FRAMES:
                    row = PRE.design_row(model, obs.features, lambda: obs.head6)
                    if row is not None:
                        preflight.append(row)
                return
            pipeline.on_observation(obs)

        self.source.subscribe(subscriber)
        origin = (0, 0) if options.monitor is None else options.monitor.origin
        self.display = env.make_display(
            rig.device_w_px,
            rig.device_h_px,
            headless=options.headless,
            origin=origin,
            overlay_available=True,
            show_unfiltered_overlay=options.show_unfiltered,
            # A fullscreen window IS the thing that gets clicked, so a real
            # click over it never reaches the application underneath.
            click_through=options.click_by == "wink",
        )
        self.cleanup.push("display close", self.display.close)
        cursor = CUR.CursorAdapter(
            enabled=options.move_cursor,
            setter=env.set_pointer,
            getter=env.get_pointer,
            clock=clock.now,
            limits=CUR.CursorLimits(
                max_step_px=options.cursor_max_step_px,
                smoothing=options.cursor_smoothing,
                dead_zone_px=options.cursor_dead_zone_px,
            ),
        )
        # Released before the display and the library: the pointer goes back
        # where it was found even if either of those then misbehaves.
        self.cleanup.push("pointer restore", cursor.release)
        # Clicking, scrolling and typing are OS input: off unless asked for
        # twice (--move-cursor and --i-mean-it) and only in wink mode.
        clicker = CK.ClickAdapter(
            enabled=options.input_enabled, sender=env.click_sender, clock=clock.now,
            sleep=clock.sleep,
        )
        scroller = CK.ScrollAdapter(
            enabled=options.input_enabled, sender=env.wheel_sender, clock=clock.now
        )
        scroll_cfg = SCR.ScrollConfig(
            **{
                name: value
                for name, value in (
                    ("arm_ms", options.scroll_arm_ms),
                    ("repeat_ms", options.scroll_repeat_ms),
                )
                if value is not None
            }
        )
        board = MENU.MenuModel(
            dwell_ms=options.menu_dwell_ms,
            click_type=options.wink_click if options.wink_click in MENU.CLICK_TYPES else "single",
        )
        # The scan lag defaults to the detector's OWN hold: a wink is stamped
        # on the frame where the closure crossed ``hold_ms``.
        assert self.wink_cfg is not None
        scan_options: dict[str, Any] = {"wink_lag_ms": self.wink_cfg.hold_ms}
        if options.scan_ms is not None:
            scan_options["scan_ms"] = options.scan_ms
        if options.scan_settle_ms is not None:
            scan_options["settle_ms"] = options.scan_settle_ms
        if options.wink_lag_ms is not None:
            scan_options["wink_lag_ms"] = options.wink_lag_ms
        try:
            keyboard = KB.ScanningKeyboard(KB.ScanConfig(**scan_options))
        except ValueError as exc:
            raise SystemExit(f"the scanning keyboard cannot be set up: {exc}") from exc
        keys = KEYS.KeyAdapter(
            enabled=options.input_enabled,
            sender=env.key_sender,
            sleep=clock.sleep,
            foreground=env.foreground_window,
            title=env.window_title,
        )
        # Then the keys: a held Alt makes every later keystroke a command.
        self.cleanup.push("key release", keys.release)
        # The button first of all (registered last, so it runs first): a held
        # button turns every later pointer move into a drag. ``release``
        # covers a stuck button AND a deliberate carry -- it clears the drag
        # flag with it, so shutdown cannot leave one believed held.
        self.cleanup.push("button release", clicker.release)
        self.input_adapters.extend((clicker, keys))
        # Starting ACTIVE is the operator's decision. The two OS gates still
        # stand in front of it, and Esc still stops everything.
        control = (
            CTL.ToggleMachine(
                start=CTL.ToggleMachine.ACTIVE if options.start_active else CTL.Mode.PAUSED
            )
            if options.controlled
            else None
        )
        self.safety = SafetyController(
            control=control, input_enabled=options.input_enabled, clock=clock.now
        )
        to_pixels = (
            (lambda point: SC.to_desktop_pixels(point, monitor, desktop))
            if monitor is not None and desktop is not None
            else None
        )
        self.executor = ActionExecutor(
            safety=self.safety,
            router=ACT.ActionRouter(),
            cursor=cursor,
            clicker=clicker,
            scroller=scroller,
            keys=keys,
            to_pixels=to_pixels if to_pixels is not None else _no_pixels,
            telemetry=self.telemetry,
            clock=clock.now,
            hold_after_click_s=options.hold_after_click_s,
        )
        # The desk components. Built only with --desk, and given the dwell time
        # this PERSON needs rather than the one the code was written with.
        access = profile.accessibility_settings()
        desk_bar = BAR.ControlBar(dwell_ms=access.dwell_ms) if options.desk else None
        desk_keyboard = SKB.SpatialKeyboard(dwell_ms=access.dwell_ms) if options.desk else None
        if desk_bar is not None:
            for line in BAR.layout_warnings():
                # Said BEFORE the session, not inferred from a bad run after it.
                print(f"desk layout: {line}")
        self.controller = InteractionController(
            options=options,
            pipeline=self.pipeline,
            safety=self.safety,
            executor=self.executor,
            telemetry=self.telemetry,
            clock=clock.now,
            key_is_down=env.key_is_down,
            to_pixels=to_pixels,
            board=board,
            keyboard=keyboard,
            scroll_cfg=scroll_cfg,
            recovery_menu=recovery_menu,
            bar=desk_bar,
            spatial_keyboard=desk_keyboard,
        )
        self.presenter = LivePresenter(session=self)

        self.source.start()
        self.display.draw_message(["Camera warming up..."])
        if env.warm_up(self.display, LIB.CAMERA_WARMUP_S):
            return 1
        if not options.skip_model_check and not self._model_check(model, preflight):
            return 2
        live["on"] = True
        cursor.__enter__()
        clicker.__enter__()
        if control is not None:
            cursor.paused = not control.mode.cursor_enabled
            print(f"starting {control.label}")
        self.wink_rule_matches = ratio_watcher(self.wink_cfg)
        self.session_started = True
        self._loop()
        return None

    def _model_check(self, model: Any, preflight: list[np.ndarray[Any, Any]]) -> bool:
        # Always report, even with nothing to check: printing nothing when no
        # frames arrived reads exactly like a pass. "No frames" is reported
        # separately rather than by passing None to preflight_verdict, where
        # None means "this model has no kernel".
        ok = True
        if not preflight:
            print(
                "model check: no warm-up frames carried a tracked face, so the model was "
                "not checked against today's conditions. If the point misbehaves, that is "
                "the first thing to suspect."
            )
        else:
            ok, message = PRE.preflight_verdict(model.support_activation(np.vstack(preflight)))
            print(f"model check ({len(preflight)} warm-up frames): {message}")
        if not ok:
            self.display.draw_message(
                ["Model does not fit current conditions.", "", "See the terminal."]
            )
            self.env.clock.sleep(MODEL_REFUSED_PAUSE_S)
        return ok

    # -- the loop: orchestration only ----------------------------------------------------

    def _loop(self) -> None:
        env, clock = self.env, self.env.clock
        safety, controller, presenter = self.safety, self.controller, self.presenter
        assert safety is not None and controller is not None and presenter is not None
        started = clock.now()
        max_seconds = self.options.max_seconds
        while True:
            # A click-through window never takes focus, so pygame stops seeing
            # key presses the moment the overlay works. The stop is read from
            # the keyboard directly, or it would fail exactly while the
            # dangerous mode was on.
            if self.display.poll_escape() or (safety.controlled and env.escape_is_down()):
                break
            if max_seconds is not None and clock.now() - started > max_seconds:
                break
            # Read ONCE per frame: the pipeline builds a fresh snapshot on
            # every access, so reading it twice can answer two different ways.
            state = self.pipeline.state
            if controller.tick_control(state):
                break
            view = controller.tick_pointer(state)
            presenter.render(state, view)
            clock.sleep(LOOP_TICK_S)

    # -- shutdown ------------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Stop, release, close, then report. A failure in one never skips the rest."""

        stopped = True
        try:
            # First: no new action starts (an action being sent finishes), and
            # no camera frame reaches the pipeline. Guarded on its own, so a
            # stuck or failing stop cannot keep a button held.
            if self.safety is not None:
                self.safety.begin_stop()
            stopped = self.source.stop()
        except Exception as exc:  # noqa: BLE001 - releases run regardless
            print(f"cleanup step 'stop' failed: {exc!r}")
        finally:
            failures = self.cleanup.close()
            self._after_cleanup(failures, stopped)

    def _after_cleanup(self, failures: list[Any], stopped: bool) -> None:
        if stopped is False:
            print("WARNING: a camera frame was still being processed when shutdown began")
        if self.session_started and self.safety is not None and self.safety.controlled:
            try:
                self.telemetry.print_report(self)
            except Exception as exc:  # noqa: BLE001 - a report never undoes a release
                print(f"session report failed: {exc!r}")
        # Said on its own line, outside the report: the adapters suppress a
        # failed release and only remember it.
        for adapter in self.input_adapters:
            if getattr(adapter, "stuck_button", None) or getattr(adapter, "keys_stuck", False):
                print(f"WARNING: input may still be held after shutdown: {adapter.summary()}")
        for failure in failures:
            print(f"cleanup step {failure.step!r} failed: {failure.error}")
        executor = self.executor
        if self.session_started and executor is not None and (
            executor.cursor.enabled or executor.cursor.moves
        ):
            print(f"cursor: {executor.cursor.summary()}")

    def _final_summary(self) -> int:
        state = self.pipeline.state
        tracked = "unknown"
        if state.frames:
            last = "tracking" if state.tracking else "not tracking"
            tracked = f"{state.frames} frames, last frame {last}"
        print(f"live view ended: {tracked}")
        if state.frames == 0:
            print(
                "no frames reached the view at all -- the camera produced nothing usable. "
                "Check it is uncovered, the room is lit, and no other application is holding it."
            )
        if self.pipeline.errors:
            print(f"{self.pipeline.errors} frame errors, last: {self.pipeline.last_error}")
        if getattr(self.source, "errors", 0):
            print(
                f"{self.source.errors} frames could not be read from the library, "
                f"last: {self.source.last_error}"
            )
        if self.controller is not None and self.controller.chosen:
            print(f"gesture chose: {self.controller.chosen[0]}")
            return RECALIBRATE_REQUESTED
        return 0


def _no_pixels(point: tuple[float, float]) -> tuple[int, int]:
    raise RuntimeError("no verified screen: the pointer cannot be placed")
