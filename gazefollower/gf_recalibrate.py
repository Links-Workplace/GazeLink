"""Recalibrate, check the result against the current profile, then decide.

The sequence is: record a calibration, fit it, record a short held-out check,
score BOTH models on that same check, and only then choose.

Three things make this different from "calibrate and hope":

* **No configuration search.**  The profile's configuration is pinned and
  refitted as-is.  Searching again would change the model and the calibration
  in one step, and a difference could then be credited to either.  It also
  removes the need for a TUNE protocol at all, because nothing is being
  selected -- which makes the session shorter and the comparison cleaner.

* **The comparison is paired.**  Old and new predict the same frames of the
  same new recording, with the same filter.  The profile's stored historical
  number is context, not evidence: it was measured on different frames on a
  different day, and every single-run comparison attempted in this project
  before pairing was unreadable because run-to-run noise was larger than the
  effect being looked for.

* **The old profile survives either way.**  A new calibration lands in its own
  recording directory and becomes active only if it is chosen.

No OS input.  Recordings hold derived embeddings and stay under ``recordings/``
like every other protocol.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_live as L  # noqa: E402
import gf_profile as PROF  # noqa: E402
import gf_recal_compare as CMP  # noqa: E402
import gf_record as R  # noqa: E402
import gf_schema as S  # noqa: E402

# Long enough to read the instruction and settle, short enough that a person
# waiting through it twice does not feel stranded.
ADVANCE_TIMEOUT_S = 6.0

# Replacing a calibration that works is not free: it discards a known
# quantity for one measured over about forty seconds. "Not worse" is
# therefore the wrong bar -- a +2 px difference on half the frames is noise,
# and adopting on it would churn the active profile for nothing. These two
# are judgement calls, not measurements: an improvement must be big enough to
# matter and consistent enough to believe.
MIN_IMPROVEMENT_PX = 10.0
MIN_FRACTION_BETTER = 0.60


@dataclass
class Outcome:
    """Everything needed to decide, and to explain the decision afterwards."""

    recording_dir: Path
    model_dir: Path
    old: dict[str, Any]
    new: dict[str, Any]
    paired: dict[str, Any]
    better: bool
    summary: str


def fit_pinned(recording_dir: Path, config_name: str, *, overwrite: bool = False) -> Path:
    """Refit exactly the profile's configuration on the new calibration.

    Deliberately not ``phase0``: that sweeps 43 configurations and selects
    among them, which is a different experiment.  Here the method is held
    fixed so that what changed is the calibration and nothing else.
    """

    import gf_fit as FIT  # noqa: PLC0415
    import gf_presets as PRE  # noqa: PLC0415

    rec_a = S.Recording.load(recording_dir, "A")
    rig = C.RigGeometry.from_dict(rec_a.meta["rig"])
    config = next(
        (c for c in PRE.sweep_with_presets(()) if c.name == config_name),
        None,
    )
    if config is None:
        raise SystemExit(
            f"the profile names a configuration this build does not know: {config_name!r}"
        )
    head_names = tuple(config.head_names)
    mask, dropped = FIT.training_rows(rec_a, head_names)
    X = FIT.design_matrix(rec_a, mask, head_names)
    Y = rec_a.label_cm[mask]
    model = FIT.FittedModel.fit(
        config,
        X,
        Y,
        rig=rig,
        train_meta={
            "protocol": "A",
            "round_id": rec_a.round_id,
            "head_dropped": dropped,
            "session": rec_a.meta.get("session"),
            "fitted_by": "gf_recalibrate",
        },
    )
    target = Path(recording_dir) / "models" / config.name
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists; refusing to replace it")
    return model.save(target)


def worth_adopting(paired: dict[str, Any]) -> tuple[bool, str]:
    """Is the improvement big enough and consistent enough to swap on?

    Decided on the paired difference rather than on the two medians: the
    medians cover the same frames here, but only the paired value says how
    CONSISTENT the direction is, and a median that improves while barely half
    the frames improve is a difference the next run would reverse.
    """

    if not paired.get("n_paired"):
        return False, "no frames both models predicted; nothing was compared"
    delta = paired["median_paired_delta_px"]
    fraction = paired["fraction_of_frames_new_is_better"]
    if delta < MIN_IMPROVEMENT_PX:
        return False, (
            f"improvement {delta:+.1f} px is under the {MIN_IMPROVEMENT_PX:.0f} px needed to "
            "justify replacing a calibration that already works"
        )
    if fraction < MIN_FRACTION_BETTER:
        return False, (
            f"better on only {fraction * 100:.0f}% of frames (needs "
            f"{MIN_FRACTION_BETTER * 100:.0f}%), so the direction is not consistent enough"
        )
    return True, (f"improvement {delta:+.1f} px on {fraction * 100:.0f}% of frames: worth adopting")


def compare_on_check(
    profile: PROF.Profile,
    recording_dir: Path,
    new_model_dir: Path,
    *,
    protocol: str = "T1",
) -> Outcome:
    """Score the current and the incoming model on the same new frames."""

    import gf_fit as FIT  # noqa: F401, PLC0415 - loaded by score_one

    rec = S.Recording.load(recording_dir, protocol)
    rig = C.RigGeometry.from_dict(rec.meta["rig"])
    settings = profile.filter_settings()
    old = CMP.score_one(profile.model_path(), rec, rig, settings)
    new = CMP.score_one(new_model_dir, rec, rig, settings)
    paired = CMP.paired_difference(old["_filtered"], new["_filtered"], rec)
    old_px = old["filtered"]["median_euclid_px"]
    new_px = new["filtered"]["median_euclid_px"]
    better, why = worth_adopting(paired)
    summary = (
        f"current {old_px:.1f} px, new {new_px:.1f} px (filtered, same {paired.get('n_paired', 0)} "
        f"frames); paired difference {paired.get('median_paired_delta_px', 0.0):+.1f} px, new "
        f"better on {paired.get('fraction_of_frames_new_is_better', 0.0) * 100:.0f}% of frames"
        f"\n  -> {why}"
    )
    return Outcome(
        recording_dir=Path(recording_dir),
        model_dir=Path(new_model_dir),
        old=old,
        new=new,
        paired=paired,
        better=better,
        summary=summary,
    )


def adopt(profile: PROF.Profile, outcome: Outcome, *, name: str | None = None) -> str:
    """Save the new calibration as its own profile and make it active.

    The previous profile is neither edited nor deleted, so going back is
    selecting it again rather than recovering it.
    """

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    new_name = name or f"{profile.name}_{stamp}"
    settings = profile.filter_settings()
    fresh = PROF.from_recording(
        new_name,
        model_dir=outcome.model_dir,
        recording_dir=outcome.recording_dir,
        protocol="A",
        settings=settings,
        x_range=None if profile.x_range is None else (profile.x_range[0], profile.x_range[1]),
        scores={
            "heldout_filtered_px": outcome.new["filtered"]["median_euclid_px"],
            "heldout_unfiltered_px": outcome.new["unfiltered"]["median_euclid_px"],
            "paired_vs_previous_px": outcome.paired.get("median_paired_delta_px"),
            "previous_profile": profile.name,
            "measured_utc": stamp,
        },
    )
    PROF.save(fresh)
    PROF.activate(new_name)
    return new_name


def next_free_round(out_root: Path) -> int:
    """One past the highest round already recorded.

    A saving run must name its round, and picking one by hand is exactly the
    kind of step that makes a hands-free sequence not hands-free.
    """

    used = {
        int(p.name[5:])
        for p in Path(out_root).glob("round*")
        if p.is_dir() and p.name[5:].isdigit()
    }
    # One past the highest, not the lowest gap: filling a gap drops a new
    # recording into the middle of the historical series, where round4 now
    # sits between experiments from a different day and reads as one of them.
    return max(used) + 1 if used else 0


def require_usable_calibration(recording_dir: Path, *, protocol: str = "A") -> int:
    """Refuse to fit a calibration that is too thin to be one.

    Checked here, against the recording, rather than left to the fitter: the
    fitter can only say "need at least 4 training rows", which reads as a
    problem with the model when it is a problem with the session -- a covered
    camera, a face out of shot, an aborted run. The number that matters is
    how many points actually collected their frames.
    """

    rec = S.Recording.load(recording_dir, protocol)
    accepted = rec.rows_accepted()
    n = int(accepted.sum())
    targets_seen = len({int(t) for t in rec.target_id[accepted]}) if n else 0
    expected = len([t for t in rec.targets])
    if targets_seen < expected or n < expected * C.N_FRAMES_PER_POINT:
        raise SystemExit(
            f"the calibration is not usable: {n} accepted rows over {targets_seen} of "
            f"{expected} targets (expected about {expected * C.N_FRAMES_PER_POINT}). "
            "Nothing was changed; the previous profile is untouched."
        )
    return n


def run_sequence(
    profile: PROF.Profile,
    *,
    round_id: int | None = None,
    monitor: Any = None,
    out_root: Path | None = None,
) -> Outcome:
    """Record, fit and check. Does NOT decide -- the caller does that."""

    rig = profile.rig_geometry()
    x_range = None if profile.x_range is None else (profile.x_range[0], profile.x_range[1])
    targets, targets_tune = (
        R.band_target_files(R.PACKAGE_DIR / "targets.json", R.PACKAGE_DIR / "targets_tune.json")
        if x_range is not None
        else (R.PACKAGE_DIR / "targets.json", R.PACKAGE_DIR / "targets_tune.json")
    )
    # The geometry comes from the target file, exactly as gf_record's own main
    # does for A/T1, so the labels are computed on the same ruler as every
    # earlier recording.
    _, geometry = C.load_targets(targets)
    root = out_root or R.RECORDINGS_DIR
    result = R.run_session(
        protocols=["A", "T1"],
        round_id=round_id if round_id is not None else next_free_round(root),
        rig=rig,
        targets_t1=targets,
        targets_tune=targets_tune,
        targets_grid16=R.PACKAGE_DIR / "targets_grid16.json",
        target_geometry=geometry,
        out_root=root,
        # The current model is drawn live through both protocols, because a
        # dot that visibly follows (or fails to follow) the eye is a check no
        # number afterwards can replace.
        #
        # Its preflight check is deliberately skipped here, and only here: the
        # whole reason to recalibrate is that the current model may no longer
        # fit today's conditions, so letting that check abort the session
        # would make the fix unreachable exactly when it is needed. Nothing is
        # scored from the overlay -- it is a visual aid, and the new model is
        # fitted from the recorded features either way.
        overlay_model_dir=profile.model_path(),
        overlay_filter_settings=profile.filter_settings(),
        skip_model_check=True,
        dry_run=False,
        headless=False,
        speed=1.0,
        x_range=x_range,
        monitor=monitor,
        advance_timeout_s=ADVANCE_TIMEOUT_S,
    )
    # watchdog_tripped is checked separately from aborted: a watchdog trip
    # leaves aborted False, so a run that recorded NOTHING slipped through
    # this guard, reached the fitter, and died there with "need at least 4
    # training rows" -- a message about the model for a failure of the camera.
    if result.aborted or result.watchdog_tripped or not result.saved:
        raise SystemExit(
            f"recalibration did not complete: {result.failure or 'aborted'}. Nothing was changed."
        )
    for protocol in ("A", "T1"):
        if protocol not in result.recordings:
            raise SystemExit(f"recalibration did not record {protocol}. Nothing was changed.")
    recording_dir = Path(next(iter(result.recordings.values()))).parent
    require_usable_calibration(recording_dir)
    config_name = Path(profile.model_dir).name
    model_dir = fit_pinned(recording_dir, config_name)
    return compare_on_check(profile, recording_dir, model_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--monitor", default=None)
    parser.add_argument(
        "--adopt",
        choices=("never", "if-better", "always"),
        default="if-better",
        help=(
            "what to do with the result. 'if-better' (default) switches only when the paired "
            "comparison favours the new calibration; 'never' measures and changes nothing; "
            "'always' switches regardless, which is only sensible when the current profile is "
            "known to be unusable."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = L.resolve_profile(args.profile)
    outcome = run_sequence(profile, round_id=args.round)
    print()
    print(outcome.summary)
    print(f"new calibration: {outcome.model_dir}")
    if args.adopt == "never":
        print(f"keeping the current profile ({profile.name}); nothing was changed.")
        return 0
    if args.adopt == "if-better" and not outcome.better:
        print(
            f"the new calibration is not better on these frames, so {profile.name!r} stays active."
        )
        return 0
    name = adopt(profile, outcome)
    print(f"activated new profile {name!r}. The previous one ({profile.name}) is still saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
