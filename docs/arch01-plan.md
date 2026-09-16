# ARCH-01 implementation plan (lead: Claude Opus 5)

Scope: gazefollower/ live path. TECHNICAL_SPEC.md v2.0 stages A–H. ADR: docs/decisions/0002-modular-live-architecture.md.
Test command: `cd gazefollower && ../../gazefollower_eval/.venv/Scripts/python.exe -m unittest discover -s tests` (delete __pycache__ first).
Baseline 2026-09-15: 1177 tests, 1 pre-existing failure (test_recalibrate CarriedForwardTests.test_the_live_session_reads_the_carried_wink_rule: fixture equals new 35ms default, user's uncommitted change).

## Verified review findings (lead read the code)
- gf_live.run_live 584-1648; closures hold ui_mode/scroll_seeded/content_px/notice/tally/armed.
- finally (1540-1631): report prints BEFORE clicker.release/keys.release/cursor.release; a report exception skips all releases; releases not individually guarded; display.close failure skips shutdown_library.
- build_gaze_follower (684), R.Display (713), KB.ScanConfig SystemExit (782-784) run before try (845): library never shut down on those failures.
- state_face_ok (509) compares runner.clock timestamp with time.monotonic().
- armed=True constants at 1034-1067, 1339; inline wink gate 1366-1385 duplicates ActionRouter rules (needed because keyboard.select mutates before routing).
- Subscriber keeps delivering frames during shutdown (no stopping flag).
- gf_record ProtocolRunner._on_frame gate (837-839, float32, requires not idle) vs LiveRunner (238, float64). Both call predict_overlay_point; filter semantics equal.
- Tests replace ~20 module globals incl. real OS senders (tests/test_live_loop.py:307-379).

## Stage B – lifecycle & safety fixes (in current gf_live, before restructuring)
- gazelink_core/app/lifecycle.py: CleanupStack(push(name, fn)); close() runs LIFO, each step try/except, idempotent, records failures; never raises.
- run_live: acquisitions registered immediately (library shutdown right after build; display close right after creation); stopping flag set first in shutdown so subscriber drops frames; release order: clicker, keys, cursor (each guarded) → stop delivery → display.close → shutdown_library → report (guarded).
- state_face_ok uses runner.clock.
- New tests (fakes only): report raises → all three releases still called; one release raises → others still called; display factory raises after library built → shutdown called; double close is no-op; callback after stopping ignored.
- Expected before/after documented per failure scenario only.

## Stage C – seams + characterization
- gazelink_core/domain/clock.py: Clock protocol (now(), sleep()), MonotonicClock, FakeClock (sleep advances + hooks).
- gazelink_core/app/options.py: LiveOptions dataclass = current run_live kwargs and defaults (single source; CLI/profile precedence unchanged).
- run_live gains keyword seams (defaults = current behavior): clock, source_factory, pipeline_factory, display_factory, input_ports, key_reader, screen, model_loader.
- Golden trace test: FakeClock + scripted pipeline events + FakeInput, recorded BEFORE stage F: emitted input calls, mode prints, tally, router summary per scenario (loss/recovery, blink vs wink, pause/resume, menu, scroll seed, stale & duplicate wink, keyboard target loss, Esc with held button). Stored as JSON under tests/golden/.
- Numerical characterization on existing recordings (recordings/resolution/round3 + 1-2 others present): per frame gate decision, raw prediction, filtered point for live policy and record policy → golden .npz of numbers only (no embeddings/frames stored in tests; derived from existing local recordings, compared exactly).
- Harness migration: test_live_loop._run injects fakes instead of patching CK._send/KEYS._send_key/CUR._set_cursor_pos etc.
- Test-wide guard: env GAZELINK_FORBID_OS_INPUT=1 set by a tests/__init__-level helper → real senders raise.

## Stage D – extract core from recording/fitting tools
- gazelink_core/gaze/predict.py: predict_one, predict_overlay_point, overlay_design_row (from gf_record).
- gazelink_core/gaze/visibility.py: visible_point, OVERLAY_STALE_S.
- gazelink_core/gaze/preflight.py: preflight_verdict, PREFLIGHT_* constants, PreflightCollector with explicit policy LIVE (single model, no guard) / RECORD (x+y, guarded).
- gazelink_core/calibration/model.py: FittedModel + FitConfig + resolve_gamma/_new_svr/ridge helpers (from gf_fit; gf_fit re-imports, same class object).
- gazelink_core/ui/pygame_display.py: Display + hebrew helpers + prompt_layout (verbatim move).
- gazelink_core/tracking/gazefollower_library.py: make_pass_through_calibration, shutdown_library, call_with_timeout wrapper, build config.
- gf_record/gf_fit re-export old names. gf_live stops importing gf_record/gf_fit. gf_setup fingerprint list gains core files.

## Stage E – tracking source + shared pipeline
- domain/observation.py FrameObservation (frozen, read-only features array preserving library dtype, library-frame openness + availability flag, face_present, gaze_status, head6|None, pnp|None (optional), raw_gaze_cm|None, tracking_state str, timestamp_ns, frame_seq, observed_s).
- tracking/gazefollower_source.py GazeFollowerSource: build library (incl hi components), subscribe → translate → publish, warm-up, stop (stopping flag), shutdown; context manager. translate() is the ONLY getattr on library objects.
- tracking/replay_source.py ReplaySource from schema.Recording rows.
- gaze/sample_gate.py SamplePolicy(LIVE float64 / RECORD float32+non-idle), gaze_usable(), apply_filter().
- gaze/pipeline.py FramePipeline (LiveRunner logic on FrameObservation); gf_live.LiveRunner = compat wrapper translating face/gaze.
- ProtocolRunner consumes FrameObservation (on_frame(face,gaze) wrapper kept for tests); shadow path translated through same adapter function.
- gf_resolution_view uses the source/pipeline; library setup copies removed.

## Stage F – interaction / safety / execution
- interaction/safety.py SafetyController: owns ToggleMachine, face freshness (injected clock), activation (move_cursor+confirmed), stopping; Permission snapshot (cursor_enabled, selection_armed, face_ok, may_scroll, stopping, reasons).
- ActionRouter gains intent screening (`screen_gesture`) implementing the stale/menu/scroll/paused pre-check once (same order as today, no router counters change), decide() still re-validates at execution.
- interaction/controller.py InteractionController: UiMode machine (enter_mode), menu/keyboard/scroll/content anchor, wink & long-close handling; uses MenuModel, ScanningKeyboard, ScrollRepeater, RecoveryMenu unchanged; calls executor, never ports.
- interaction/executor.py ActionExecutor: router.decide + ports with armed from Permission; pointer follow/freeze/jump/hold; keyboard target lock/status; returns ActionResult.
- app/telemetry.py SessionTelemetry: tally + session report text (identical lines).
- ui/presenter.py LivePresenter: HUD/board/scan view models → DisplayPort draw calls (identical arguments).
- Tests: forbidden transitions, permission change between decision and execution, release while paused, no input when stale/low confidence; golden trace identical.

## Stage G – app wiring & structure
- app/live_session.py LiveSession: build, lifecycle, loop tick orchestration only; run_live and CLI thin wrappers.
- Move library modules into core with sys.modules aliases (gf_gesture, gf_dwell, gf_scroll, gf_menu, gf_keyboard, gf_actions, gf_control → interaction; gf_click, gf_keys, gf_cursor, gf_overlay, gf_screen_check, gf_display → platform; gf_gaze_filter → gaze; gf_head_features, gf_common → domain/gaze; gf_schema, gf_profile, gf_presets? → calibration). Break gf_screen_check→gf_live cycle (resolve_profile to app/profiles).
- Boundary test: core never imports gf_* tools / argparse CLI / pygame|ctypes in domain+interaction; no face_info/gaze_info getattr outside tracking adapter.

## Stage H
- Full suite, golden traces, numeric equivalence, ruff/mypy counts on changed files vs HEAD, p50/p95 callback timing on replay before/after; TASKS/ADR update; hardware checklist listed as pending.

## Risks
- Monkeypatch tests depend on module identity → sys.modules alias approach.
- FakeClock loop determinism: the loop's only wait is clock.sleep(0.005) → fake advances there.
- float dtype policy must be preserved bit-exactly.

## Amendments approved by the user (2026-09-15) — binding
1. **Input guard must not depend on test discovery.** `tests/__init__.py` is not guaranteed to load under `unittest discover -s tests`. Instead the real Win32 senders (`_send`, `_send_wheel`, `_send_key`, `_set_cursor_pos`) refuse unless the process explicitly armed real input (`platform.real_input.arm(reason)`), which only CLI `main()` entry points call after `--i-mean-it`. Default = disarmed, so any test (or programmatic caller) that forgets a fake raises instead of emitting. A dedicated test asserts the guard is active in a fresh interpreter (subprocess) and that each sender raises when disarmed.
2. **Permission evaluated at execution time.** `ActionExecutor` never trusts a stored `Permission`. It holds the `SafetyController` and asks for the current permission (with the injected clock) inside `execute`, immediately before calling a port. Test: permission revoked between the controller's decision and execution → no input, ActionResult REFUSED.
3. **Synchronised stopping blocks new actions.** `SafetyController.begin_stop()` and `ActionExecutor.execute()` share one lock: an action already inside `execute` completes before shutdown proceeds to releases; any `execute` after `begin_stop` is refused. The frame callback also drops frames after stop. Tests: execute after stop → refused; a thread blocked inside a port during `begin_stop` → stop waits (bounded), releases run after it, and no second action starts.

## Opus plan review round 1 (claude-opus-5[1m], read-only sub-agent, effort unverified) — resolutions
- B1 early-exit crash (report read `ratios_seen`/`seen_in` before assignment → UnboundLocalError skipped every release): confirmed; fixed by `session_started` guard in stage B. Report is not printed on warm-up/preflight exit (it crashed before). Dedicated test added.
- B2 release order: push order is display → pointer → keys → button, LIFO gives button → keys → pointer → display → library; `stopping.set()` runs before `cleanup.close()`. Exact-order test added.
- B3 stopping unread: subscriber now checks `stopping` first. Test added.
- B4 clock fallback for fakes: accepted as a stage-B stopgap only for runner doubles without `.clock`; stage F moves freshness into SafetyController with the injected clock and removes the getattr.
- B5 golden data privacy: golden files store only SHA-256 digests of per-frame output arrays + counts, never predictions/points/features. Real-recording checks skip when recordings are absent.
- B6 dead patches after extraction: harness migrates to explicit injection in stage C, before any stage-D extraction; every fake records that it was called (canaries).
- B7 guard vs releases: the arming guard blocks button-down, key-down, wheel and pointer moves; button-up and key-up (release paths) are always allowed. Cursor restore is a move and stays guarded (suppressed in release when disarmed). Tests for both.
- Extra: recorder `overlay_updated_s` only on valid frames vs live `updated_s` every frame → part of SamplePolicy. Wink queue unbounded → bounded with overflow counter (documented behaviour change at overflow only). Pause/resume via SPACE/eyes does not invalidate queued winks → explicit bug fix in stage F with test (spec §15). Scroll path while paused characterised before adding permission check. Hidden core→tool imports (gf_display→gf_targets, gf_screen_check→gf_live, gf_fit model loading, sys.path inserts) are stage D/G targets in the boundary test.
- Equivalence claim scope: replay proves equivalence of the refactored code path against the pre-refactor code on the same inputs (hashes captured before stage E); it does not prove camera-to-cursor behaviour or live float64 numerics from float32 recordings.
- Fingerprint: core files added to the code fingerprint with a manifest note; compare tools report the incompatibility explicitly.
- Arming: `gf_live.main` and `gf_click_practice.main` arm after their confirmations; `run_live` refuses enabled adapters when input is not armed; tests arm only in scoped context managers.
- Stop lock: re-entrant; release/pause/stop-scroll bypass stopping/permission refusal; resume never bypasses activation or tracking; bounded wait timeout → releases re-run after the port returns and stuck input reported.
