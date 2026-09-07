# Preregistered protocol — full-screen accuracy on a 24-inch monitor

**Status: written before any measurement on this monitor. No results exist yet.**
Frozen 7 September 2026. Changing anything below after seeing results starts a
new experiment with a new identifier; it does not amend this one.

## 1. The question

The central-band run on the ultrawide showed the engine's raw horizontal
output is informative near the screen centre and flat toward the edges. A
24-inch 16:9 monitor asks the eye to travel roughly the same horizontal
distance as that central band, and less vertically.

**Does the engine deliver usable accuracy across an ENTIRE standard monitor?**

This is a new measurement, not a re-analysis. Nothing below is established by
the ultrawide work; that work only motivated the question.

## 2. What is and is not carried over

| Carried over | Not carried over |
|---|---|
| The recording protocol (9-point calibration, 1.5 s settle, 1.5 s collect, blink gate) | Any fitted calibration model. A new screen needs fresh personal calibration. |
| The `central-band-svr` preset — method, features, hyperparameters | Its accuracy numbers. Those are TUNING-SET results and have never been validated independently. |
| The screening thresholds in `gf_select.py` | The 120 px reference. It belongs to a 4096 px-wide geometry and is replaced here by an angular gate. |

### Honest status of the preset

`central-band-svr` (z-scored features and labels, RBF SVR, C=100, gamma=0.0005)
was the best-behaved candidate in the round2 central-band **tuning** sweep:
Euclidean median 141 px, Euclidean P90 239 px, no off-screen predictions.

When it was subsequently scored on that round's independent test set for the
first time, it gave Euclidean median 144 px, X median 97 px, Y median 68 px,
zero off-screen — but Euclidean **P90 613 px**, which does not pass the
screening gate defined below. The median transferred almost unchanged from
tuning to test, which is reassuring about overfitting; the tail did not.

It is carried forward as a candidate, not as a validated configuration.

## 3. Setup to record (every session)

The recorder captures all of this into `setup.json` and every `*.meta.json`.

| Item | How |
|---|---|
| Monitor identity, resolution, desktop origin | `--monitor` (index or name); recorded automatically |
| Physical width and height | `--screen-width-cm`, `--screen-height-cm`; cross-checked against the OS report, mismatch >2 cm warns |
| OS scaling, logical vs device pixels | `--dpi-scale` |
| Viewing distance (eye to screen centre) | `--viewing-distance-cm` |
| Camera distance (eye to camera) | `--eye-distance-cm` — a different number, and the one the specification's 45–70 cm range refers to |
| Camera model, resolution, measured frame rate | probed automatically |
| Camera position and angle | `--camera-x-cm`, `--camera-y-cm`; tilt measured separately if `--camera-pitch-deg` is used at analysis time |
| Preset and run identifier | `--preset`, `--round` |

**Measure, do not estimate**, the screen's active area, the viewing distance
and the camera position. Every angular number depends on them.

If the camera distance falls outside 45–70 cm, that is an extension beyond the
specification's stated range and must be written into the run note.

## 4. Session structure

Three sessions, on at least two different days, each with **fresh
calibration**. The protocol and configuration are frozen before session 1.

| Protocol | What you do | Purpose |
|---|---|---|
| **A** | 9-point calibration, head still | The personal calibration |
| **TUNE** | tuning targets, head still | Configuration selection only. Never scored as a result. |
| **FULL** | 15 targets over the whole screen — centre, sides, top, bottom, four corners — head still, comfortable neutral posture | The main measurement |
| **MOVE** | the same 15 targets, sitting naturally with small head movements | Robustness. A separate block, never merged with FULL. |

Target order is randomised with a recorded seed. Head movement is **not** an
input gesture: the product is eye-only, and MOVE measures tolerance to natural
posture, not a required action.

Exclusions and missing predictions are recorded, never silently dropped:
availability is computed over every eligible row.

## 5. What gets reported

Per session, per target, and grouped by centre / edge-h / edge-v / corner:

- X, Y and Euclidean error — mean and median
- Euclidean P90, P95, P99 and maximum
- Angular error from the measured geometry, with the assumptions stated
- Invalid and off-screen prediction rates
- Availability, frame rate, and latency where measurable
- FULL versus MOVE, side by side

Raw predictions are reported separately from any smoothing. No clipping to the
screen, and no dropping of samples to improve a number.

## 6. Acceptance gates — fixed before data collection

Every threshold is either reused with its source named, or labelled **proposed**
with its reasoning. Angular gates are primary, because they survive a change of
screen; the pixel equivalents shown are for a 1920×1080 24-inch panel at 73 cm,
where **1.5° ≈ 69 px** and the diagonal is 2203 px.

### Gate 1 — Calibration safety and numerical stability

*Does the calibration produce output that is safe to act on?* Evaluated on
**TUNE**, per session, by `gf_select.py`.

| Rule | Threshold | Source |
|---|---|---|
| Euclidean P90 | ≤ 12 % of the diagonal (264 px, ≈ 5.7°) | Proposed |
| Euclidean P95 | ≤ 18 % of the diagonal | Proposed |
| Euclidean P99 | ≤ 25 % of the diagonal (551 px) | Proposed |
| Single worst error | ≤ 1 diagonal | Proposed — beyond this it is a numerical failure, not an accuracy one |
| Off-screen predictions | ≤ 2 % | Proposed |
| Availability | ≥ 95 % of eligible rows | Targets document, G4 |
| Worst single target's median | ≤ 35 % of the diagonal | Proposed — catches "nine good targets, one hopeless" |
| Non-finite output or a failed fit | rejected outright | — |

**If no candidate passes, the report says "no acceptable calibration".** The
least bad candidate is never chosen.

### Gate 2 — Repeatability across all three sessions

*Is the result a property of the system rather than of one lucky sitting?*

| Rule | Threshold | Source |
|---|---|---|
| Gate 1 passes | in all 3 sessions | — |
| Median Euclidean error, each session | ≤ 3.0° | Proposed — twice the specification's mean goal, as a per-session floor |
| Spread of the session medians | max − min ≤ 1.5° | Proposed — larger means the calibration, not the engine, decides the result |
| Availability, each session | ≥ 95 % | Targets document, G4 |

### Gate 3 — Readiness for a supervised interaction trial

*Is it good enough to try a real task, under supervision?*

| Rule | Threshold | Source |
|---|---|---|
| Mean Euclidean angular error, FULL, all sessions pooled | ≤ 2.0° | Proposed — a large-target trial is informative below this even though it misses the product goal |
| Euclidean P90, FULL | ≤ 4.0° | Proposed |
| Every screen region (centre, edge-h, edge-v, corner) | median ≤ 3.0° | Proposed — a dead corner makes a task unusable regardless of the average |
| Off-screen rate | ≤ 2 % | Proposed |
| MOVE median degradation vs FULL | ≤ 1.0° | Proposed |
| Frame rate | ≥ 25/s median | Below the 30/s goal, but enough for a supervised trial |

### Gate 4 — Compliance with the specification's accuracy target

*Does it meet the product requirement?*

| Rule | Threshold | Source |
|---|---|---|
| **Mean gaze error across the screen ≤ 1.5°** | all 3 sessions | **Specification 3.2 / 12** |
| Every region's mean | ≤ 1.5° | Proposed reading of "across the screen" |
| Availability | ≥ 95 % | Targets document, G4 |
| Frame rate ≥ 30/s and latency P95 < 50 ms | as specified | Specification 3.2 — latency currently **NOT MEASURABLE**: the library timestamps a frame on receipt, not exposure |

Gate 4 cannot be fully closed on this hardware while latency is unmeasurable.
That limitation is reported, not worked around.

### On the 120 px reference

The project's 120 px figure was an experimental reference on a 4096 px-wide
screen (≈ 2.6° at 60 cm). It is **not** carried to this monitor; the angular
gates above replace it. Historical pixel numbers stay in their own reports.

## 7. Deciding

- **All four gates pass** → the engine meets the specification on a 24-inch
  screen. Proceed to the interaction trial.
- **Gates 1–3 pass, Gate 4 does not** → PARTIAL. A supervised trial with large
  targets is justified; the accuracy goal is not met, and the gap is reported
  in degrees.
- **Gates 1–2 pass, Gate 3 does not** → the calibration is stable but not
  accurate enough for interaction. Report the limit; do not proceed.
- **Gate 1 fails in any session** → no acceptable calibration. Report and stop.

No threshold moves after results are seen. If one turns out to be wrong, that
is a finding, and testing the revised threshold is a new experiment.

## 8. What stays out of scope here

Snapping, magnification and dwell assistance are evaluated **separately** from
raw gaze accuracy: they change what a user can hit without changing what the
engine estimates, and mixing them would make the accuracy number meaningless.

No real OS input is emitted at any point in this experiment.
