# Display identity: knowing what screen we are on

## Context

GAZELINK maps normalized gaze (0..1) onto a screen. That mapping is only meaningful against the exact display it was calibrated on. Today the product reads the display **once, at launch, always `primaryScreen()`**, and never looks again.

The goal is not only to catch a monitor change mid-session. It is that the product, **installed on any computer**, works out for itself what screen it is running on and behaves correctly — because that is what M6 ("a new user can install, calibrate, and use GAZELINK safely without a developer present") actually requires.

### What the code does today

- Five live screens each duplicate the same four-line snapshot of `application.primaryScreen()`:
  [calibration_window.py:282](src/gazelink/calibration_window.py#L282), [gaze_window.py:123](src/gazelink/gaze_window.py#L123), [test_window.py:213](src/gazelink/test_window.py#L213), [validation_window.py:46](src/gazelink/validation_window.py#L46), [feature_check_window.py:165](src/gazelink/feature_check_window.py#L165).
- A **startup** guard exists but is weaker than it looks: [`load_latest_model`](src/gazelink/gaze_engine.py#L559) returns a model only when `model.screen_geometry == live_geometry` (exact equality incl. `screen_id`, `dpi_scale`). It catches a *changed* display, but `screen_id` is a connector slot, so it does **not** catch a *swapped* one (finding 1). And it fails with a developer string — `"No compatible gaze model. Run --guided-calibration first."` — not a product flow.
- A **per-frame** guard is dead: [`GazeEstimator.estimate`](src/gazelink/gaze_engine.py#L333) compares two frozen snapshots, so it can never fire mid-session.
- **Zero** Qt display signals connected anywhere in `src/` (`screenChanged`, `screenAdded`, `screenRemoved`, `primaryScreenChanged`, `geometryChanged` → no matches).
- **No physical size anywhere in the product.** `ScreenGeometry` ([domain.py](src/gazelink/domain.py)) carries only `width_px`, `height_px`, `dpi_scale`, `orientation`, `screen_id`. Centimetres exist only in the research harness, where they are typed in by hand (`--screen-width-cm`).
- No monitor selection — always primary. A user whose working screen is the secondary cannot calibrate on it.

### Review findings folded in (2026-09-08)

Five corrections from review, each checked against the code:

1. **`screen_id` is not an identity.** Qt documents `QScreen::name` as not a unique identifier; on Windows it is `\\.\DISPLAY1` — a *connector slot*, not a monitor. Swap monitors on the same port at the same resolution and the equality key at [gaze_engine.py:566](src/gazelink/gaze_engine.py#L566) matches, so a calibration from a different physical screen is silently accepted. This is the most severe hole and it exists **today**, independent of this work. It also reverses the Tier-2 demotion below: EDID matters after all — for **identity**, not size — so it moves into Tier 1.
2. **Same screen does not prove the calibration still fits.** Identity only rules out the wrong profile; it says nothing about the camera having been nudged or the user sitting differently. The mechanism for this **already exists and is tested**: [`measure_feature_drift`](src/gazelink/live_validation.py#L144) and `build_calibration_feature_reference` compare live feature medians against interpolated calibration anchors and raise `FEATURE_DRIFT_SUSPECTED`. Wire it to profile load; do not build anything new.
3. **A freeze with no hands-free way out is a trap.** If gaze is frozen, gaze cannot be used to recover. An eyelid signal is calibration-independent and is the right channel — but the naive form does not work: [`extract_eye_features`](src/gazelink/features.py#L141) returns `_invalid` once `lid_gap` reaches epsilon, so `openness` is unavailable at exactly the moment of a deliberate close. The usable signal is *face tracked + eye geometry invalid, sustained*, read from the raw `tick.observation`. Detail and the prototype requirement are in Tier 1.
4. **Coordinate units are undefined.** [eyegestures_run.py:450](src/gazelink/eyegestures_run.py#L450) computes `physical_width_px = width_px * dpi_scale`, so `ScreenGeometry.width_px` is *logical* px — yet [`normalized_to_pixel`](src/gazelink/gaze_engine.py#L363) consumes it directly and nothing states the contract. Two spaces already coexist undocumented. `PassThrough` alone does not fix this; the contract must be written down.
5. **Never report degrees without both size and distance.** Angular accuracy needs screen size *and* viewing distance. If either is unknown or EDID-derived, report pixels and label the estimate; do not present it as measured.

### The physical-size finding

A gaze model is trained on one physical rig — screen size, viewing distance, camera position. A model trained on a 27" at 60 cm is invalid on a 15" laptop at 45 cm even at identical resolution, so a profile can never transfer between physically different screens.

But **the product does not need centimetres to work.** `grep` over `src/gazelink/` finds no `width_mm`, no `screen_w_cm`, no physical units at all — `ScreenGeometry` carries pixels only, and the product functions. The reason: the model is learned in **normalized** coordinates on whatever screen it was calibrated on. It needs the calibration to have happened on this geometry; it does not need to know how big that geometry is in centimetres.

Centimetres are needed only to **report** accuracy in degrees of visual angle — a research and QA concern, which the research harness already solves with manual entry (`--screen-width-cm`).

What the code uses today as its "same rig" key — pixel geometry plus `screen_id` — is **not sufficient**, per finding 1. EDID is therefore needed for identity even though it is not needed for size; see `DisplayIdentity` in Tier 1.

**This rules out any user-facing measurement step.** GAZELINK's users cannot use their hands; a task that requires dragging a rectangle to match a credit card contradicts the mission, and — decisively — solves a problem the product does not have. Physical size must therefore be **passive, best-effort, and never a gate**.

Research into what Windows can supply passively:

| Source | Unit | Limitation |
|---|---|---|
| WMI `WmiMonitorBasicDisplayParams.MaxHorizontalImageSize` | **integer cm** | 1 cm quantisation — a 52.7 cm screen reports 53 |
| EDID Detailed Timing Descriptor | **mm** | 10× the precision, but requires parsing the raw EDID blob |
| Qt `QScreen.physicalSize()` | mm | same EDID underneath |
| `screeninfo` `width_mm` / `height_mm` | mm | same EDID; already used by [gf_display.py:83](gazefollower/gf_display.py#L83) |

**All four are the same EDID number.** There is no independent OS source. EDID can be absent, zeroed, or simply wrong, because the manufacturer writes it — which is precisely why the research harness already carries [`SCREEN_SIZE_TOLERANCE_MM = 20.0`](gazefollower/gf_setup.py#L61) with the comment *"somebody measured the bezel, or the wrong monitor"*.

So EDID is taken as best-effort and **labelled**, never trusted silently and never confirmed by the user. Where a reported figure depends on it, say so ("≈1.4°, from EDID") rather than presenting an unverifiable number as measured.

For the record, the known ruler-free verification method is the **credit-card task** — resize an on-screen rectangle to match an ISO/IEC 7810 ID-1 card (85.60 × 53.98 mm by standard), validated in the Virtual Chinrest paper (Li, Joo, Yeatman & Reinecke, *Scientific Reports*, 2020). It requires hands, so it can only ever be an **optional step in the installer**, performed by whoever physically mounts the camera and positions the monitor — work that is hands-on regardless. It must never appear in the user's daily path or gate any function.

**Viewing distance** (today `--viewing-distance-cm`, typed by hand) is subject to the same rule. The same paper's blind-spot task reaches 3.25 cm mean error but requires closing one eye and tracking a moving dot — unusable here. GAZELINK already has a camera and MediaPipe landmarks, so inter-pupillary distance in pixels gives a continuous estimate with **no user task at all**. Absolute IPD varies per person (~57–70 mm), so use it for **relative** tracking ("has the user moved since calibration?"), which is the safety-relevant question anyway. Prototype and measure before committing.

### Intended outcome

Install on any machine → the product enumerates displays, lets the user choose one **by gaze or by keyboard, with no hands required**, calibrates on it, and thereafter refuses to run against a display that is not the one it was calibrated on — freezing safely and saying why, instead of degrading silently. Physical size is collected passively where the hardware offers it and never asked for.

## Approach

Three tiers, in dependency order. **Tier 1 is the safety floor and is worth shipping alone.** Tiers 2–3 are what make it a product rather than a rig.

### Tier 1 — Know the current display, and freeze when it changes

New `src/gazelink/display_watch.py`, Qt-free core (mirroring the deliberate no-Qt rule in [screen_mapping.py](src/gazelink/screen_mapping.py)) so it is testable headless:

- `describe_screen(screen) -> ScreenGeometry` — the **single** place converting a `QScreen` to a `ScreenGeometry`. Accepts any object with `name()`, `geometry()`, `devicePixelRatio()`, so a fake works in tests. Replaces all five duplicated copies; that duplication is why the rule can drift per-screen today.
- **`DisplayIdentity`** — the stable key, and the fix for finding 1. It carries an explicit **confidence**, and only the top tier is treated as an identification:
  - `CONFIRMED` — EDID manufacturer + product code + **serial number**. Only this may silently auto-match a stored profile.
  - `AMBIGUOUS` — EDID present but no usable serial, so the key is `(manufacturer, product, mm, resolution, connector)`. Two identical monitors of the same model are **indistinguishable** here, and so are many panels that ship a blank serial. This is *not* an identification: it requires explicit user verification before a stored profile is used.
  - `UNIDENTIFIED` — no usable EDID at all (common on laptops and behind some KVMs). Same rule: never auto-match.
  
  Size, resolution and connector name are **never** treated as sufficient on their own — that is precisely the failure being designed out. When identity is not `CONFIRMED`, the product asks rather than assumes; the confirmation must be reachable through the hands-free channel described below, or it re-creates the trap it is meant to prevent. Keep this key separate from `ScreenGeometry` so the geometry contract stays a pure pixel description.
- `detect_change(expected, actual) -> DisplayChange | None` — compares identity **and** geometry. Geometry uses `ScreenGeometry` equality, the same definition already trusted at [gaze_engine.py:566](src/gazelink/gaze_engine.py#L566), so startup and runtime cannot disagree. No loose tolerance: a DPI or 1-px difference already invalidates a trained model.
- **Coordinate contract** (finding 4): state in the module docstring that `ScreenGeometry.width_px`/`height_px` are **logical** px in Qt's coordinate space, that `normalized_to_pixel` returns logical px, and that `dpi_scale` is the only conversion to device px. Add a test asserting the relationship so the two spaces cannot drift silently as they do now.
- `DisplayGuard` — **latches** on first change and stays tripped. A monitor that changes and changes back must not silently resume; the session that spanned the change is already suspect.
- Add `ReasonCode.DISPLAY_CHANGED` to [domain.py](src/gazelink/domain.py). The enum already separates causes at this granularity (five distinct `CORRECTION_*` codes); reusing `CALIBRATION_INVALID` would hide the cause from logs and from the M5 event API.

Qt binding (one place): connect `QGuiApplication.screenAdded/screenRemoved/primaryScreenChanged`, the active `QScreen.geometryChanged/physicalDotsPerInchChanged`, and the window's `QWindow.screenChanged` (fires when the window is dragged to another display). Re-check on the ordinary timer tick as a backstop — signal coverage varies across Windows versions and a missed signal must not mean a missed freeze.

Wire into all five windows following the refusal precedent already set by [`_verify_surface_matches_ruler`](src/gazelink/test_window.py#L437) — it refuses rather than warns, which is the right call. On trip: stop **gaze output and any control**, clear the prediction overlay, show a red message naming both geometries, and mark any in-flight recording aborted so a partial dataset is never promoted. **The tick timer and the vision loop keep running** — they are what the recovery gesture depends on. (`test_window`'s existing refusal does stop its timer; that is correct for a recording screen with nothing to recover to, but must not be copied to the live screens.)

Also fix here: `gaze_window` and `calibration_window` size themselves from `self._widget.screen()` ([gaze_window.py:360](src/gazelink/gaze_window.py#L360), [calibration_window.py:429](src/gazelink/calibration_window.py#L429)) while scoring against the `primaryScreen()` snapshot — two potentially different displays. Capture one `QScreen` and use it for both, as `test_window` already does ([comment at test_window.py:425](src/gazelink/test_window.py#L425)).

Set an explicit `setHighDpiScaleFactorRoundingPolicy(PassThrough)` before the first `QApplication`. Nothing in the repo sets any High-DPI policy today (`highdpi|dpiaware|SetProcessDpi` → no matches), and without it `devicePixelRatio` can round differently per monitor and cause **false** trips. This is necessary but not sufficient — the written coordinate contract above is the actual fix.

**Hands-free recovery** (finding 3) — without this the freeze is a trap, so it ships *with* Tier 1, not after it. Three constraints, all verified against the code:

- **Do not stop the tick timer.** The timer drives `runtime.tick()`, so stopping it stops vision itself and kills the very signal recovery depends on. Freeze **gaze output and control**, keep the capture/vision loop running. The earlier wording ("stop the timer") was wrong and is corrected throughout.
- **Read `tick.observation`, not `tick.accepted_observation`.** [`RuntimeTick.observation`](src/gazelink/runtime.py#L39) is non-optional and carries the raw frame result, while `accepted_observation` is `None` whenever the confidence policy rejects the frame ([confidence.py:250](src/gazelink/confidence.py#L250)). Recovery must survive rejection, so it reads the raw field. Assert this in a test — reading the wrong field is a silent, total failure of the escape hatch.
- **`openness` is NOT available at full closure — the earlier claim was wrong.** [`extract_eye_features`](src/gazelink/features.py#L141) returns `_invalid` when `lid_gap <= _GEOMETRY_EPSILON`, and bails immediately when iris landmarks are absent — both true of a shut eye. So `left_eye` is `None` exactly when the gesture is being made. The detectable signal is therefore **"face still tracked, eye geometry invalid, sustained for N ms"**, not a reading of openness. This is a different mechanism and it **must be prototyped and measured on real video before Tier 1 is called done**; if it proves unreliable, the freeze needs a different escape and this plan must change rather than ship a trap.
- **Cycle vs confirm must be unambiguous.** Proposal to validate during the prototype: a *short* sustained close (~400–800 ms) advances the highlight, a *long* one (~1.5 s+) confirms, with a visible countdown so the user sees which they are in and an audible/visual cancel if they open early. Both thresholds live in the one config place, not scattered.
- This overlaps M3-03 ("מצמוץ, קריצה ושהייה"). One owner must hold the gesture state machine — do not build a second, parallel detector here. Either M3-03 lands first, or this defines the shared component that M3-03 later extends.
- The frozen screen offers at most **two or three quadrant-sized targets** ("recalibrate" / "exit"), so a gaze-driven hit needs only the crudest accuracy — but the close-gesture path must work with no gaze at all.
- Keyboard remains available as a parallel path, never the only one.

**Profile-load fitness check** (finding 2): after a profile matches on identity, run the existing [`measure_feature_drift`](src/gazelink/live_validation.py#L144) against the stored calibration anchors before trusting it. `FEATURE_DRIFT_SUSPECTED` prompts a short re-validation rather than a full recalibration. Reuse only — this is already built and tested.

### Tier 2 — Physical size, passively and honestly

Small, and deliberately so: nothing here gates the user or asks them for anything.

- Extend `ScreenGeometry` with optional `width_mm` / `height_mm` and `size_source` (`EDID` | `UNKNOWN`). Optional keeps every existing serialized model loadable — verify `from_dict`/`to_dict` round-trips before changing the contract, since the spec already pins strict schema-rejection behaviour.
- Read EDID mm via Qt `physicalSize()` (no new dependency). Do **not** build a raw-EDID DTD parser speculatively; the coarse value is enough for labelling.
- Keep the mm **size** fields out of the geometry equality key: a monitor reporting mm on one boot and nothing on the next must not invalidate a good model. Identity is handled separately by `DisplayIdentity` (Tier 1), which uses EDID serial/product — not the mm figures — so the two concerns do not collide.
- Use it for exactly two things: labelling reported degrees as EDID-derived, and a sanity note when a newly selected display is physically very different from the calibrated one.
- **Never report degrees unless both physical size and viewing distance are known and non-estimated** (finding 5). Otherwise report pixels, and label any angular figure as an estimate with its source. An unverifiable number presented as measured is worse than no number.

Optional installer-only step (not on the user's path, not required by anything): the credit-card measurement described above, for installations that want defensible degree reporting.

### Tier 3 — Choose the display, and a real "wrong screen" flow

- Monitor selection in the product, reusing the logic already working in [gf_display.py](gazefollower/gf_display.py) (`list_monitors`, `pick_monitor`, and the desktop-origin handling that stops a window landing on the wrong screen).
- Replace the developer error string at startup with a guided flow: *"This profile was calibrated on <name> (52×29 cm). You are now on <name> (34×19 cm). Recalibrate for this screen?"* → straight into guided calibration. Profiles stored per display so moving between two known monitors does not mean recalibrating each time.
- Viewing-distance estimation from IPD — **prototype and measure first**, do not design it in yet.

### Still out of scope

Virtual-desktop coordinate space and multi-monitor gaze routing belong with the M3 cursor adapter (there is zero OS-input code in the repo today). Automatic model migration across physically different screens is **not possible** for the reason given above — recalibration is the correct answer, not a limitation to engineer around.

## Files

| File | Change |
|---|---|
| `src/gazelink/display_watch.py` | **New.** Pure core + Qt watcher (Tier 1). |
| `src/gazelink/domain.py` | `ReasonCode.DISPLAY_CHANGED`; later optional mm fields on `ScreenGeometry`. |
| `src/gazelink/{gaze,calibration,test,validation,feature_check}_window.py` | Use `describe_screen`; bind watcher; freeze handler; single-`QScreen` fix. |
| `src/gazelink/display_watch.py` | Tier 2 EDID read lands here too — no separate module needed. |
| `tests/test_display_watch.py` | **New.** |

## Verification

**Automated** (`pytest`, ruff check/format, mypy — the gates this repo already runs), headless with a fake screen:
- `detect_change` returns `None` when equal, non-`None` for each of width, height, `dpi_scale`, `screen_id`, orientation changing independently.
- `DisplayGuard` **stays tripped after the geometry reverts** — the latch is the safety property; assert it explicitly.
- **Identity**: two displays with identical resolution and connector name but different EDID serial must NOT match (finding 1 — this is the test that would fail today). Absent or blank serial resolves to `AMBIGUOUS`, absent EDID to `UNIDENTIFIED`, and **neither auto-matches a stored profile** — assert that explicitly for both, since "falls back to resolution" is the exact bug being removed.
- **Coordinate contract**: assert `normalized_to_pixel` returns logical px and that `logical × dpi_scale` gives device px, so the two spaces cannot drift.
- **Recovery**: assert the freeze leaves the tick timer running, that the detector reads `tick.observation` and never `accepted_observation`, and that it still fires when the confidence policy rejects every frame. A fixture of frames with `left_eye = None` and the face tracked must trigger recovery — this is the closed-eye case, and the test must use `None`, not a small `openness`, because `None` is what the code actually produces.
- `describe_screen` equals a hand-built `ScreenGeometry` from the same fake, so the shared helper cannot drift from the five call sites it replaces.
- Freeze handler clears the overlay and blocks gaze output while **leaving the tick timer running** (assert against fakes, not internals).
- Tier 2: existing serialized models still load after the `ScreenGeometry` change, **and** a display whose mm are present on one run and absent on the next still matches the same model (mm must not be part of the identity key).

Confirm the full suite still passes (TASKS.md reports 532 as the current baseline) and that no test opens a camera or emits OS input.

**Manual** — required, cannot be claimed without the hardware. Per the standing preference, each is verified by watching the live prediction overlay on screen, not by reading logs alone:
1. `--gaze-check` running → change resolution → point freezes immediately, red message appears.
2. → change scaling 100%→125% → same.
3. → drag window to the second monitor → same.
4. → unplug the second monitor → clean message, no crash.
5. → change resolution *back* → **stays frozen** (latch holds).
6. Restart on the original display → normal operation, model loads.
7. Tier 2: on at least one desktop monitor and one laptop, confirm EDID mm are read when present and that a display reporting nothing degrades to `UNKNOWN` without blocking calibration or gaze.

## Task tracking

Add to TASKS.md as an M3 prerequisite, blocking M3-02 "הזזת הסמן לפי מבט" ([TASKS.md:373-396](TASKS.md#L373)) — cursor movement must not be built on an unverified ruler — and as an M6 dependency for install-anywhere. Record Tier 2/3 as separate entries so Tier 1 can land alone. Do not mark DONE without the manual scenarios.

**Review:** the user has directed that review goes through them, not Codex, for this work. Do not invoke Codex review for it.

## Scope decision (settled with the user, 2026-09-08)

**Product only — `src/gazelink/`.** `gazefollower/` is left alone: its start-time monitor selection already works ([gf_display.py](gazefollower/gf_display.py) enumerates displays, `--monitor` selects, desktop origin is honoured, declared cm are cross-checked against EDID), and it is driven by an operator who picks the monitor deliberately on each run.

Known and accepted as a consequence: the harness keeps the same mid-run gap and the same name-based identity weakness. That is tolerable for an operator-supervised research run and is **not** tolerable for a product a person relies on unattended. Revisit only if the harness starts running unsupervised.

**This plan is not a diagnosis.** No ROUND5 fault has been reported and none was found in the repo; the user confirmed there is no known fault. The work is a structural gap being closed before M3, not a fix for an observed symptom — nothing here should be cited as having explained a past run.
