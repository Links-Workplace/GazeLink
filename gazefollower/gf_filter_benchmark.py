"""Tune a calibrated-point filter on TUNE and evaluate the winner once on T1.

Only aggregate metrics are written. Recordings and derived face embeddings are
read locally and never copied into the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_head_features as H  # noqa: E402
import gf_schema as S  # noqa: E402


def predict(model: FIT.FittedModel, rec: S.Recording, rig: C.RigGeometry) -> np.ndarray:
    """Predict every model-eligible row and preserve missing rows as NaN."""

    mask, _ = FIT.scoring_rows(rec, model.schema.head_names)
    out = np.full((rec.n_rows, 2), np.nan, dtype=np.float64)
    design = S.assemble(
        rec.features[mask],
        rec.head[mask] if model.schema.head_names else None,
        model.schema.head_names,
        H.HEAD6_NAMES,
    )
    out[mask] = model.predict_norm(design, rig)
    return out


def replay(
    prediction_norm: np.ndarray,
    timestamp_ns: np.ndarray,
    settings: GF.FilterSettings,
) -> tuple[np.ndarray, float]:
    """Replay in capture order and return output plus mean filter CPU time."""

    output = np.full_like(np.asarray(prediction_norm, dtype=np.float64), np.nan)
    flt = GF.GazePointFilter(settings)
    elapsed_ns = 0
    measured = 0
    for i, point in enumerate(prediction_norm):
        if not np.all(np.isfinite(point)):
            flt.reset()
            continue
        started = time.perf_counter_ns()
        try:
            output[i] = flt.update((float(point[0]), float(point[1])), float(timestamp_ns[i]) / 1e9)
        except ValueError:
            flt.reset()
            continue
        elapsed_ns += time.perf_counter_ns() - started
        measured += 1
    return output, (elapsed_ns / measured / 1e6 if measured else float("nan"))


def _presentation_windows(indexes: np.ndarray, target_id: np.ndarray) -> list[np.ndarray]:
    """Split ``indexes`` into contiguous runs of one target id: one per showing.

    A target id is not a presentation.  ``--repeat-first`` re-shows the first
    targets at the end of the run keeping their ids, so grouping by id alone
    pools two visits separated by half a minute into a single "fixation" --
    which hides exactly the drift those runs were recorded to expose, and
    inflates the dispersion of every repeated target.  Capture order is
    monotonic, so each showing is one contiguous run of rows.
    """

    if indexes.size == 0:
        return []
    ids = target_id[indexes]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    bounds = [int(v) for v in starts] + [int(indexes.size)]
    return [indexes[bounds[i] : bounds[i + 1]] for i in range(len(starts))]


def jitter_metrics(
    points_norm: np.ndarray,
    rec: S.Recording,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Scatter around each presentation's own median, without target bias.

    One fixation is one contiguous showing of a target, not one target id: a
    repeated target is two fixations, measured separately.

    Two summaries are returned, and they are not interchangeable.

    ``p95_radius_px`` pools every sample from every fixation and takes one
    percentile over the pool.  On a 10-fixation recording that tail is set by
    whichever one or two fixations went worst, so the pooled number describes
    the worst moments of the run rather than typical steadiness.  Measured on
    round6/T1 it reads 380.8px while eight of the ten fixations sit near
    125px.

    ``median_of_fixation_p95_px`` takes each fixation's own P95 first and then
    the median across fixations, so one bad fixation cannot carry the whole
    statistic.  Report both: the pooled value answers "how bad does it get",
    the per-fixation median answers "how steady is it usually".  Tuning
    against the pooled value alone optimises for two fixations out of ten.
    """

    collecting = rec.rows_collecting()
    scale = np.array([width, height], dtype=np.float64)
    radii: list[float] = []
    jumps: list[float] = []
    per_fixation_p95: list[float] = []
    per_fixation_median: list[float] = []
    valid = np.all(np.isfinite(points_norm), axis=1)
    for window in _presentation_windows(np.flatnonzero(collecting), rec.target_id):
        indexes = window[valid[window]]
        if indexes.size < 3:
            continue
        points_px = points_norm[indexes] * scale
        centre = np.median(points_px, axis=0)
        this_fixation = np.linalg.norm(points_px - centre, axis=1)
        radii.extend(this_fixation.tolist())
        per_fixation_p95.append(float(np.percentile(this_fixation, 95)))
        per_fixation_median.append(float(np.median(this_fixation)))
        paired = zip(points_px, points_px[1:], indexes, indexes[1:], strict=False)
        for before, after, before_index, after_index in paired:
            if after_index == before_index + 1:
                jumps.append(float(np.linalg.norm(after - before)))
    return {
        "n": len(radii),
        "n_fixations": len(per_fixation_p95),
        "median_radius_px": _percentile(radii, 50.0),
        "p95_radius_px": _percentile(radii, 95.0),
        "median_of_fixation_p95_px": _percentile(per_fixation_p95, 50.0),
        "worst_fixation_p95_px": max(per_fixation_p95) if per_fixation_p95 else None,
        "median_of_fixation_median_px": _percentile(per_fixation_median, 50.0),
        "p95_frame_jump_px": _percentile(jumps, 95.0),
    }


def support_activation(
    model: FIT.FittedModel,
    rec: S.Recording,
    rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """How far the recorded features sit outside the calibration's support.

    An RBF SVR answers with its constant bias once a query is far from every
    support vector, so a fixation whose features leave the calibrated region
    does not merely get noisier -- it snaps to a fixed wrong point, or freezes
    there.  Measured on round6/T1 the only two fixations whose activation
    reached 0.0 were exactly the two whose dispersion exploded (535px and
    698px against ~125px elsewhere), and every sample of round2/T2, recorded
    while the head was deliberately nodding, sits at 0.0.

    Returns the maximum kernel activation per row: 1.0 sits on a support
    vector, 0.0 means the prediction carries no calibration information at
    all.  Ridge models have no support vectors and report ``available: False``
    rather than a fabricated number.
    """

    index = np.arange(rec.n_rows) if rows is None else np.flatnonzero(rows)
    # The design must carry the head columns the model was fitted with, or the
    # schema check rejects it outright.  Hardcoding "no head columns" made
    # this function usable only for embedding-only models, which is not a
    # property the caller can see from the outside.
    head_names = model.schema.head_names
    design = S.assemble(
        rec.features[index],
        rec.head[index] if head_names else None,
        head_names,
        H.HEAD6_NAMES,
    )
    activation = model.support_activation(design)
    if activation is None:
        return {"available": False, "reason": "not an SVR; no support vectors"}
    return {
        "available": True,
        "gamma": float(model.schema.svr["gamma"]),
        "n_support_vectors": int(model._svr_x.getSupportVectors().shape[0]),
        "median_activation": float(np.median(activation)),
        "min_activation": float(activation.min()),
        "fraction_below_0p01": float(np.mean(activation < 0.01)),
        "fraction_below_0p10": float(np.mean(activation < 0.10)),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), percentile))


def synthetic_step_ms(settings: GF.FilterSettings, fps: float = 30.0) -> float | None:
    """Filter-only time to reach 90% of a 40%-screen horizontal step.

    Kept as the headline number for continuity with the earlier report.  Step
    size 0.4 of screen width, sampled at ``fps``; the result is the filter's
    own settling time and is NOT camera-to-display latency.
    """

    return _step_ms(settings, fps=fps, start=0.3, end=0.7, axis=0)


def _step_ms(
    settings: GF.FilterSettings,
    *,
    fps: float,
    start: float,
    end: float,
    axis: int,
) -> float | None:
    """Frames-to-90% for one synthetic step, in milliseconds.

    ``axis`` 0 steps horizontally, 1 vertically; the other axis is held at
    0.5.  A step may be negative, which is why the threshold test is written
    in terms of the distance already covered rather than a fixed comparison.
    """

    flt = GF.GazePointFilter(settings)
    dt_s = 1.0 / fps
    held = 0.5
    before = (start, held) if axis == 0 else (held, start)
    after = (end, held) if axis == 0 else (held, end)
    for i in range(30):
        flt.update(before, i * dt_s)
    span = end - start
    for frame in range(1, 241):
        value = flt.update(after, (29 + frame) * dt_s)[axis]
        if abs(value - start) >= 0.9 * abs(span):
            return frame * dt_s * 1000.0
    return None


# Eight representative steps: three horizontal sizes (0.10, 0.20, 0.40 of
# screen width) each in both directions, plus one vertical size (0.20 of
# height) in both directions -- 6 horizontal + 2 vertical.  Horizontal is
# sampled more finely because the central-band experiment moves mainly in x.
# Tuning against a single step size would reward a filter that happens to
# suit that one distance.
STEP_CASES: tuple[tuple[str, float, float, int], ...] = (
    ("x_small_+0.10", 0.45, 0.55, 0),
    ("x_small_-0.10", 0.55, 0.45, 0),
    ("x_mid_+0.20", 0.40, 0.60, 0),
    ("x_mid_-0.20", 0.60, 0.40, 0),
    ("x_large_+0.40", 0.30, 0.70, 0),
    ("x_large_-0.40", 0.70, 0.30, 0),
    ("y_mid_+0.20", 0.40, 0.60, 1),
    ("y_mid_-0.20", 0.60, 0.40, 1),
)


def step_response_profile(settings: GF.FilterSettings, fps: float = 30.0) -> dict[str, Any]:
    """Settling time across every step in ``STEP_CASES``.

    ``worst_ms`` is what the acceptance check should read: a filter is only as
    responsive as its slowest representative move.  ``None`` for a case means
    it never reached 90% inside the 8-second window.
    """

    per_case = {
        name: _step_ms(settings, fps=fps, start=start, end=end, axis=axis)
        for name, start, end, axis in STEP_CASES
    }
    measured = [v for v in per_case.values() if v is not None]
    return {
        "fps": fps,
        "per_case_ms": per_case,
        "worst_ms": max(measured) if measured else None,
        "median_ms": float(np.median(measured)) if measured else None,
        "unreached_cases": [name for name, value in per_case.items() if value is None],
    }


def recorded_transition_ms(
    points_norm: np.ndarray,
    rec: S.Recording,
    width: int,
    height: int,
    min_move_px: float = 200.0,
) -> dict[str, Any]:
    """Time to cover 90% of the move the signal itself actually made.

    Measured between consecutive collected targets: take the signal's median
    position while holding the previous target and its median while holding
    the next one, then time how long after the new target's first sample the
    signal first reaches 90% of the way between those two medians.

    Anchoring on the signal's own medians rather than on target coordinates
    makes this independent of calibration bias -- a constant offset shifts
    both medians equally -- so it measures movement, not accuracy.  It still
    contains the operator's reaction time and saccade, so it is an upper
    bound on filter lag and is NOT camera-to-display latency.  Pairs whose
    medians differ by less than ``min_move_px`` are skipped: with no real
    move to time, the number would be noise.
    """

    # The move itself happens in the settle window, which rows_collecting()
    # excludes -- timing from the scored window alone would always report ~0,
    # because by then the eye has already arrived.  So the search window runs
    # from target onset (STABILIZING) while the reference medians are taken
    # from the steady COLLECTING samples only.
    collecting = rec.rows_collecting()
    onset = collecting | (rec.phase == S.PHASE_STABILIZING)
    scale = np.array([width, height], dtype=np.float64)
    times: list[float] = []
    unreached = 0
    skipped = 0
    onset_indexes = np.flatnonzero(onset)
    collect_indexes = np.flatnonzero(collecting)
    if onset_indexes.size and collect_indexes.size:

        def median_px(window: np.ndarray) -> np.ndarray | None:
            points = points_norm[window]
            valid = points[np.all(np.isfinite(points), axis=1)]
            return None if valid.size == 0 else np.median(valid * scale, axis=0)

        # Indexed by presentation, not keyed by target id.  With
        # --repeat-first a dict keyed by id keeps only the second visit, so
        # the first transition into that target would be timed against an
        # anchor recorded half a minute later.
        collect_windows = _presentation_windows(collect_indexes, rec.target_id)
        onset_windows = _presentation_windows(onset_indexes, rec.target_id)
        steady = [median_px(w) for w in collect_windows]

        def onset_window_for(collect_window: np.ndarray) -> np.ndarray | None:
            first = int(collect_window[0])
            for candidate in onset_windows:
                if int(candidate[0]) <= first <= int(candidate[-1]):
                    return candidate
            return None

        for position in range(1, len(collect_windows)):
            start_px, end_px = steady[position - 1], steady[position]
            window = onset_window_for(collect_windows[position])
            if start_px is None or end_px is None or window is None:
                continue
            move = end_px - start_px
            distance = float(np.linalg.norm(move))
            if distance < min_move_px:
                skipped += 1
                continue
            unit = move / distance
            t0 = float(rec.timestamp_ns[window[0]])
            reached = False
            for row in window:
                point = points_norm[row]
                if not np.all(np.isfinite(point)):
                    continue
                travelled = float(np.dot(point * scale - start_px, unit))
                if travelled >= 0.9 * distance:
                    times.append((float(rec.timestamp_ns[row]) - t0) / 1e6)
                    reached = True
                    break
            if not reached:
                unreached += 1
    return {
        "min_move_px": min_move_px,
        "n_transitions_measured": len(times),
        "n_transitions_never_reached": unreached,
        "n_transitions_skipped_small_move": skipped,
        "median_ms": _percentile(times, 50.0),
        "p90_ms": _percentile(times, 90.0),
        "limits": (
            "includes operator reaction and saccade; upper bound on filter lag, "
            "not end-to-end latency"
        ),
    }


def score(
    points: np.ndarray,
    rec: S.Recording,
    rig: C.RigGeometry,
    filter_ms: float,
) -> dict[str, Any]:
    eligible = FIT.eligible_rows(rec)
    width = int(rec.meta["target_geometry"]["width_px"])
    height = int(rec.meta["target_geometry"]["height_px"])
    evaluation = FIT.evaluate(
        points[eligible],
        rec.target_xy[eligible],
        rec.target_id[eligible],
        width,
        height,
        fresh=rec.rows_fresh()[eligible],
        timestamp_ns=rec.timestamp_ns[eligible],
    ).flat()
    return {
        "accuracy": evaluation,
        "jitter": jitter_metrics(points, rec, width, height),
        "transition": recorded_transition_ms(points, rec, width, height),
        "filter_cpu_mean_ms": filter_ms,
    }


def candidates(width: int, height: int) -> list[GF.FilterSettings]:
    common = {"width_px": width, "height_px": height}
    result = [GF.FilterSettings(**common, kind=GF.FilterKind.OFF)]
    result.extend(
        GF.FilterSettings(**common, kind=GF.FilterKind.EMA, ema_cutoff_hz=cutoff)
        for cutoff in (0.8, 1.2, 2.0, 3.0, 5.0)
    )
    result.extend(
        GF.FilterSettings(
            **common,
            kind=GF.FilterKind.ONE_EURO,
            one_euro_min_cutoff_hz=min_cutoff,
            one_euro_beta_hz_per_px_s=beta,
        )
        # Low min_cutoff buys fixation stability; positive beta buys the
        # cutoff back during fast motion.  The adaptive question is whether
        # some (low fc, non-zero beta) pair keeps most of the stability of
        # fc=0.4/beta=0 without its ~1s settling, so the low end of fc is
        # sampled together with a finer beta ladder than the first sweep used.
        for min_cutoff in (0.2, 0.3, 0.4, 0.6, 0.8, 1.2)
        for beta in (
            0.0,
            0.0002,
            0.0005,
            0.001,
            0.0015,
            0.002,
            0.003,
            0.004,
            0.006,
            0.008,
            0.012,
            0.02,
        )
    )
    result.extend(
        GF.FilterSettings(
            **common,
            kind=GF.FilterKind.KALMAN,
            kalman_acceleration_noise_px2_s4=process,
            kalman_measurement_noise_px2=measurement,
        )
        for process in (100.0, 400.0, 1600.0, 6400.0)
        for measurement in (225.0, 900.0, 3600.0)
    )
    return result


def _candidate_name(settings: GF.FilterSettings) -> str:
    if settings.kind is GF.FilterKind.EMA:
        return f"ema_fc={settings.ema_cutoff_hz:g}"
    if settings.kind is GF.FilterKind.ONE_EURO:
        return (
            f"one-euro_fc={settings.one_euro_min_cutoff_hz:g}"
            f"_beta={settings.one_euro_beta_hz_per_px_s:g}"
        )
    if settings.kind is GF.FilterKind.KALMAN:
        return (
            f"kalman_q={settings.kalman_acceleration_noise_px2_s4:g}"
            f"_r={settings.kalman_measurement_noise_px2:g}"
        )
    return "off"


# Accuracy tolerance, fixed here BEFORE any candidate is scored so that the
# threshold cannot be talked into fitting whichever candidate looks nice:
# a candidate may not worsen TUNE median or P95 error by more than 5%.
ACCURACY_TOLERANCE = 1.05
# Responsiveness goal from the plan, applied to the SLOWEST representative
# step rather than to the single 40% step the first sweep used.
STEP_GOAL_MS = 100.0
JITTER_GOAL_FRACTION = 0.5


def choose(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Pick the steadiest candidate that still meets the response goal.

    Selection reads ``step_worst_ms`` -- a filter is only as responsive as its
    slowest representative move -- so a candidate cannot qualify by being fast
    on one convenient step size alone.
    """

    baseline = rows[0]
    base_accuracy = baseline["tune"]["accuracy"]
    base_jitter = baseline["tune"]["jitter"]["p95_radius_px"]
    accepted = []
    for row in rows[1:]:
        accuracy = row["tune"]["accuracy"]
        step = row["step_profile"]["worst_ms"]
        accurate = (
            accuracy["median_euclid_px"] <= base_accuracy["median_euclid_px"] * ACCURACY_TOLERANCE
            and accuracy["p95_euclid_px"] <= base_accuracy["p95_euclid_px"] * ACCURACY_TOLERANCE
        )
        if accurate and step is not None and step <= STEP_GOAL_MS:
            accepted.append(row)
    if not accepted:
        return baseline, (
            "no filtered candidate met the 5% accuracy tolerance together with "
            f"a worst-case {STEP_GOAL_MS:g}ms step response"
        )
    selected = min(accepted, key=lambda row: row["tune"]["jitter"]["p95_radius_px"])
    reduction = 1.0 - selected["tune"]["jitter"]["p95_radius_px"] / base_jitter
    return selected, (
        f"met the proposed >={JITTER_GOAL_FRACTION:.0%} TUNE jitter reduction"
        if reduction >= JITTER_GOAL_FRACTION
        else (
            f"passed accuracy/response guards but reached only {reduction:.1%} TUNE "
            f"jitter reduction, short of the proposed {JITTER_GOAL_FRACTION:.0%} goal"
        )
    )


def run(tune_dir: Path, test_dir: Path, model_dir: Path) -> dict[str, Any]:
    model = FIT.FittedModel.load(model_dir)
    tune = S.Recording.load(tune_dir, "TUNE")
    test = S.Recording.load(test_dir, "T1")
    tune_rig = C.RigGeometry.from_dict(tune.meta["rig"])
    test_rig = C.RigGeometry.from_dict(test.meta["rig"])
    if (tune_rig.device_w_px, tune_rig.device_h_px) != (test_rig.device_w_px, test_rig.device_h_px):
        raise ValueError("TUNE and T1 use different device pixel geometry")

    tune_prediction = predict(model, tune, tune_rig)
    rows = []
    for settings in candidates(tune_rig.device_w_px, tune_rig.device_h_px):
        output, cpu_ms = replay(tune_prediction, tune.timestamp_ns, settings)
        rows.append(
            {
                "name": _candidate_name(settings),
                "settings": asdict(settings),
                "synthetic_step_90_ms": synthetic_step_ms(settings),
                "step_profile": step_response_profile(settings),
                "tune": score(output, tune, tune_rig, cpu_ms),
            }
        )
    selected, verdict = choose(rows)

    off_settings = GF.FilterSettings(
        width_px=test_rig.device_w_px,
        height_px=test_rig.device_h_px,
        kind=GF.FilterKind.OFF,
    )
    stable_fc, stable_beta = GF.ONE_EURO_PRESETS["high-stability"]
    stable_settings = GF.FilterSettings(
        width_px=test_rig.device_w_px,
        height_px=test_rig.device_h_px,
        one_euro_min_cutoff_hz=stable_fc,
        one_euro_beta_hz_per_px_s=stable_beta,
    )
    selected_settings = GF.FilterSettings(**selected["settings"])

    test_prediction = predict(model, test, test_rig)
    # All three arms replay the identical prediction array and timestamps, so
    # the valid-sample mask is the same for each and the comparison is paired.
    raw_test, raw_cpu = replay(test_prediction, test.timestamp_ns, off_settings)
    stable_test, stable_cpu = replay(test_prediction, test.timestamp_ns, stable_settings)
    filtered_test, filtered_cpu = replay(test_prediction, test.timestamp_ns, selected_settings)
    return {
        "selection_source": str(tune_dir),
        "held_out_test": str(test_dir),
        "model": str(model_dir),
        "selection_rule": (
            f"lowest TUNE jitter P95 subject to <={ACCURACY_TOLERANCE - 1:.0%} median/P95 "
            f"error change and <={STEP_GOAL_MS:g}ms WORST-CASE step across "
            f"{len(STEP_CASES)} representative steps"
        ),
        "accuracy_tolerance": ACCURACY_TOLERANCE,
        "selected": selected["name"],
        "selected_settings": selected["settings"],
        "selected_step_profile": selected["step_profile"],
        "selection_verdict": verdict,
        "high_stability_preset": {
            "settings": asdict(stable_settings),
            "step_profile": step_response_profile(stable_settings),
            "tune": next(
                (
                    row["tune"]
                    for row in rows
                    if row["settings"]["kind"] == GF.FilterKind.ONE_EURO.value
                    and row["settings"]["one_euro_min_cutoff_hz"] == stable_fc
                    and row["settings"]["one_euro_beta_hz_per_px_s"] == stable_beta
                ),
                None,
            ),
        },
        "tune_baseline": rows[0]["tune"],
        "tune_selected": selected["tune"],
        "tune_candidates": rows,
        # Filter-independent: it describes the calibrated input every arm
        # shares, so it belongs outside the per-arm scores.
        "support_activation": {
            "tune": support_activation(model, tune, tune.rows_collecting()),
            "test": support_activation(model, test, test.rows_collecting()),
        },
        "t1_scored_once_after_selection": {
            "unfiltered": score(raw_test, test, test_rig, raw_cpu),
            "high_stability": score(stable_test, test, test_rig, stable_cpu),
            "selected_filter": score(filtered_test, test, test_rig, filtered_cpu),
        },
        "limits": [
            "synthetic step time measures only the filter, not camera-to-display latency",
            "recorded transition time includes operator reaction and model error",
            "stationary-target jitter includes residual eye settling and drift during collection",
            "round2/TUNE and round6/T1 have both already informed earlier decisions; "
            "neither is an untouched holdout",
            "hardware confirmation is still required",
        ],
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(report: dict[str, Any]) -> str:
    arms = report["t1_scored_once_after_selection"]
    before = arms["unfiltered"]
    stable = arms["high_stability"]
    after = arms["selected_filter"]
    tune_before = report["tune_baseline"]
    tune_after = report["tune_selected"]
    selected_row = next(
        row for row in report["tune_candidates"] if row["name"] == report["selected"]
    )
    stable_profile = report["high_stability_preset"]["step_profile"]
    selected_profile = report["selected_step_profile"]

    def arm_row(label: str, section: str | None, metric: str, scale: float = 1.0) -> str:
        def cell(arm: dict[str, Any]) -> str:
            values = arm if section is None else arm[section]
            value = values.get(metric)
            return "n/a" if value is None else _fmt(scale * value)

        return f"| {label} | {cell(before)} | {cell(stable)} | {cell(after)} |"

    stable_fc, stable_beta = GF.ONE_EURO_PRESETS["high-stability"]
    lines = [
        "# Calibrated overlay filter benchmark",
        "",
        f"Selected on TUNE: **{report['selected']}** — {report['selection_verdict']}.",
        "T1 was evaluated only after selection.",
        "",
        f"Selection rule: {report['selection_rule']}.",
        "",
        "## Responsiveness (filter only, synthetic)",
        "",
        f"Headline 40%-width step: **{_fmt(selected_row['synthetic_step_90_ms'])} ms** "
        f"(adaptive candidate) vs **{_fmt(stable_profile['worst_ms'])} ms** worst case "
        f"for the high-stability preset (fc={stable_fc:g}, beta={stable_beta:g}).",
        "",
        f"Adaptive worst case across {len(STEP_CASES)} steps: "
        f"**{_fmt(selected_profile['worst_ms'])} ms**; "
        f"median **{_fmt(selected_profile['median_ms'])} ms**.",
        "",
        "| step case | adaptive (ms) | high-stability (ms) |",
        "|---|---:|---:|",
        *[
            f"| {name} | {_fmt(selected_profile['per_case_ms'].get(name))} | "
            f"{_fmt(stable_profile['per_case_ms'].get(name))} |"
            for name, _, _, _ in STEP_CASES
        ],
        "",
        "These are the filter's own settling times at 30 fps. They are NOT "
        "camera-to-display latency and do not evidence the specification's "
        "50 ms end-to-end requirement.",
        "",
        "## TUNE (used for selection)",
        "",
        "| TUNE metric | unfiltered | selected filter |",
        "|---|---:|---:|",
        (
            "| jitter radius P95 (px) | "
            f"{_fmt(tune_before['jitter']['p95_radius_px'])} | "
            f"{_fmt(tune_after['jitter']['p95_radius_px'])} |"
        ),
        (
            "| median error (px) | "
            f"{_fmt(tune_before['accuracy']['median_euclid_px'])} | "
            f"{_fmt(tune_after['accuracy']['median_euclid_px'])} |"
        ),
        (
            "| P95 error (px) | "
            f"{_fmt(tune_before['accuracy']['p95_euclid_px'])} | "
            f"{_fmt(tune_after['accuracy']['p95_euclid_px'])} |"
        ),
        "",
        "## T1 (three arms, identical samples)",
        "",
        "| T1 metric | unfiltered | high-stability | adaptive |",
        "|---|---:|---:|---:|",
        arm_row("jitter P95 pooled (px, outlier-led)", "jitter", "p95_radius_px"),
        arm_row("jitter median-of-fixation-P95 (px)", "jitter", "median_of_fixation_p95_px"),
        arm_row("jitter worst fixation P95 (px)", "jitter", "worst_fixation_p95_px"),
        arm_row("jitter radius median (px)", "jitter", "median_radius_px"),
        arm_row("fixations counted", "jitter", "n_fixations"),
        arm_row("frame jump P95 (px)", "jitter", "p95_frame_jump_px"),
        arm_row("mean error (px)", "accuracy", "mean_euclid_px"),
        arm_row("median error (px)", "accuracy", "median_euclid_px"),
        arm_row("P90 error (px)", "accuracy", "p90_euclid_px"),
        arm_row("P95 error (px)", "accuracy", "p95_euclid_px"),
        arm_row("valid coverage (%)", "accuracy", "coverage", 100.0),
        arm_row("recorded transition to 90% median (ms)", "transition", "median_ms"),
        arm_row("recorded transition to 90% P90 (ms)", "transition", "p90_ms"),
        arm_row("transitions measured", "transition", "n_transitions_measured"),
        arm_row("transitions never reached", "transition", "n_transitions_never_reached"),
        (
            "| filter CPU mean (ms) | "
            f"{_fmt(before['filter_cpu_mean_ms'], 4)} | "
            f"{_fmt(stable['filter_cpu_mean_ms'], 4)} | "
            f"{_fmt(after['filter_cpu_mean_ms'], 4)} |"
        ),
        "",
        "## Calibration support (filter-independent)",
        "",
        *_support_lines(report),
        "",
        "## Limits",
        "",
        *[f"- {limit}" for limit in report["limits"]],
        "",
    ]
    return "\n".join(lines)


def _support_lines(report: dict[str, Any]) -> list[str]:
    support = report.get("support_activation", {})
    lines = [
        "Maximum RBF kernel activation of the recorded features against the "
        "calibration's support vectors. Near 0 the SVR returns its constant "
        "bias, so the point stops following the eye rather than merely "
        "getting noisier. No filter can repair those samples.",
        "",
        "| split | median activation | min | % below 0.01 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("tune", "test"):
        entry = support.get(name)
        if not entry or not entry.get("available"):
            lines.append(f"| {name} | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {_fmt(entry['median_activation'], 3)} | "
            f"{_fmt(entry['min_activation'], 3)} | "
            f"{_fmt(100.0 * entry['fraction_below_0p01'])} |"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune", type=Path, required=True, help="recording directory containing TUNE.npz"
    )
    parser.add_argument(
        "--test", type=Path, required=True, help="recording directory containing held-out T1.npz"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="aggregate JSON output")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="optional aggregate Markdown output"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args.tune, args.test, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
