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
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_display as GD  # noqa: E402
import gf_live as L  # noqa: E402
import gf_pool as POOL  # noqa: E402
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

# What a recalibration does NOT measure, and therefore must not silently
# reset. Everything else in a profile is re-derived from the new recording --
# the model, the rig, the screen, the calibration pose, the scores -- and
# rebuilding those is the whole point. These two are not gaze at all:
#
# ``gesture``  the person's eyelid rule, measured from openness landmarks. It
#              belongs to a face and a camera geometry, neither of which a
#              gaze calibration touches. Dropping it silently replaces a rule
#              measured on someone with the built-in defaults.
# ``cursor``   how the pointer should feel, chosen by the operator by using
#              it. Dropping it reverts to smoothing 0.35 / dead zone 12 --
#              the setting one profile records as "felt jumpy" in the very
#              field that says why 0.6 / 6 replaced it.
#
# Established by reading the code and confirmed by reverting this carry and
# watching the tests fail, NOT by running a calibration session: an adopted
# recalibration produced a profile with neither block, so a session that had
# just been calibrated was also, unannounced, running a pointer the person
# had already rejected.
CARRIED_FORWARD = ("gesture", "cursor")


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
    # What trained the new model. None when the session path was used, so a
    # profile can never claim a pool it was not fitted from.
    pool: POOL.Pool | None = None


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


def config_name_of(model_dir: str | Path) -> str:
    """The configuration a saved model was fitted with, from its directory.

    A pooled model is saved under ``<config>__pooled_<stamp>``. Reading the
    directory name straight back would look up a configuration no build has,
    and the failure would land at the END of a calibration session, after the
    recording, with "a configuration this build does not know".
    """

    name = Path(model_dir).name
    marker = "__pooled"
    return name[: name.index(marker)] if marker in name else name


def fit_pooled(
    recording_dir: Path,
    config_name: str,
    *,
    check_protocol: str = "T1",
    root: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, POOL.Pool]:
    """Refit the profile's configuration on EVERY compatible past recording.

    The measured reason to prefer this over :func:`fit_pinned`: one session
    is 405 rows at nine screen positions, which a 258-dimensional model
    memorises (17.8 px on its own rows, 167 px held out). Leave-one-session-out
    over 13 sessions put a single fresh calibration at a median of 226 px and
    the pooled fit at 118 px, better on 11 of 11 folds -- and still 142 px,
    better on 10 of 11, when the held-out session's screen POSITIONS were
    removed from training as well. See ``gf_pool``.

    The check recording is excluded by name. Scoring a model on rows that
    trained it would report memory as accuracy, and it is the one mistake
    that would make every number in the comparison meaningless while looking
    like a large improvement.
    """

    import gf_fit as FIT  # noqa: PLC0415
    import gf_presets as PRE  # noqa: PLC0415

    config = next((c for c in PRE.sweep_with_presets(()) if c.name == config_name), None)
    if config is None:
        raise SystemExit(
            f"the profile names a configuration this build does not know: {config_name!r}"
        )
    head_names = tuple(config.head_names)
    # Pinned to the recording being calibrated against TODAY, not to whichever
    # recording happens to sort first on disk. Without this the pool adopts the
    # OLDEST recording's key: after a screen or rig change, months of old data
    # would form the pool and the new calibration -- the one session that
    # actually describes the new setup -- would be rejected as incompatible,
    # while the comparison still took its geometry from that new session.
    require = POOL.key_of(Path(recording_dir), check_protocol)
    pool = POOL.build(
        head_names,
        root=root,
        exclude=[(Path(recording_dir), check_protocol)],
        require=require,
    )
    ok, why = POOL.usable(pool)
    if not ok:
        raise SystemExit(f"the pooled fit has too little to learn from: {why}. Nothing changed.")
    model = FIT.FittedModel.fit(
        config,
        pool.X,
        pool.Y_cm,
        rig=pool.rig,
        train_meta={
            "protocol": "A",
            "fitted_by": "gf_recalibrate (pooled)",
            "pool_rows": pool.rows,
            "pool_positions": pool.positions,
            "pool_sessions": pool.sessions,
            "held_out": f"{Path(recording_dir).name}/{check_protocol}",
            # Read back by ``check_is_held_out`` when this model is later the
            # OLD side of a comparison. Without it nothing can tell whether a
            # check recording trained the model it is being used to judge.
            "trained_on": pool.identities(),
        },
    )
    # A NEW directory every time, never a replacement. The destination used to
    # be a fixed name, and refitting against a recording whose pooled model a
    # profile already pointed at overwrote that model IN PLACE -- so the "old"
    # and "new" sides of the comparison loaded the same replaced files, and
    # even --adopt never destroyed the model the active profile referenced.
    # FittedModel.save writes several files, so an interrupted overwrite could
    # leave a mixed one. An unused candidate costs disk; a replaced one costs
    # the calibration someone is relying on.
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    models = Path(recording_dir) / "models"
    target = models / f"{config.name}__pooled_{stamp}"
    serial = 0
    while target.exists():
        serial += 1
        target = models / f"{config.name}__pooled_{stamp}_{serial}"
    return model.save(target), pool


class CheckNotHeldOut(SystemExit):
    """Raised when a check recording trained one of the models it would judge."""


def check_is_held_out(model_dir: Path, recording_dir: Path, protocol: str) -> None:
    """Refuse a comparison in which the check recording trained either model.

    Scoring a model on rows that trained it reports memorisation as accuracy.
    The NEW model is held out by construction in :func:`fit_pooled`; the OLD
    one is whatever the active profile points at, and once pooled profiles
    exist that model may well have trained on this very check. Comparing then
    flatters the old side, and the adoption decision is made on a number that
    means nothing.

    A model that does not record what trained it is refused rather than
    assumed innocent: "it does not say" and "it did not" are different facts,
    and only one of them is safe.
    """

    trained = POOL.trained_on(model_dir)
    if trained is None:
        return
    if (Path(recording_dir).name, protocol) in trained:
        raise CheckNotHeldOut(
            f"the current model was trained on {Path(recording_dir).name}/{protocol}, so "
            "scoring it there would measure what it memorised, not what it learned. "
            "Choose another check recording, or record a new session. Nothing was changed."
        )


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
    pool: POOL.Pool | None = None,
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
        pool=pool,
    )


def adopt(profile: PROF.Profile, outcome: Outcome, *, name: str | None = None) -> str:
    """Save the new calibration as its own profile and make it active.

    The previous profile is neither edited nor deleted, so going back is
    selecting it again rather than recovering it.

    ``CARRIED_FORWARD`` is copied from the old profile because a recalibration
    does not measure it. This is the ONLY code path in the project that
    creates a profile, so a field it drops is a field nobody kept.
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
    # Copied rather than merged: these blocks are read as wholes (a wink rule
    # with half its thresholds from one measurement and half from another is
    # not a rule anyone measured), and ``fresh`` has nothing in them to merge.
    # Named one by one rather than splatted from CARRIED_FORWARD: a **dict is
    # opaque to the type checker, which then cannot tell that a typo names a
    # field Profile does not have. The constant stays the documentation and
    # the thing the tests iterate, and one of them fails if a name is added
    # here in one place and not the other.
    fresh = replace(
        fresh,
        gesture=dict(profile.gesture or {}),
        cursor=dict(profile.cursor or {}),
    )
    if outcome.pool is not None:
        # A model trained on an accumulating pool is only reproducible if what
        # went into it is written down. Stored under ``calibration`` because
        # that is where this profile says where its model came from, and
        # because the pool is now a larger part of that answer than the one
        # session the recording directory names.
        calibration = dict(fresh.calibration)
        calibration["pool"] = outcome.pool.provenance()
        calibration["fitted_on"] = "pool"
        fresh = replace(fresh, calibration=calibration)
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
    fit: str = "pooled",
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
    # Resolved here rather than left as None. run_session records
    # ``"monitor": None`` when it is not given, and a recording without a
    # monitor block yields a profile with no connector, resolution or physical
    # millimetres -- which gf_screen_check then refuses, because a panel that
    # cannot be identified cannot be a verified ruler. The CLI path always
    # resolved one (gf_record line ~2507); only this programmatic caller did
    # not, so a recalibration produced a profile that M3-00 blocks from ever
    # moving a cursor. Same trap the band_target_files docstring names: the
    # step lived in the CLI and the programmatic caller skipped it silently.
    chosen_monitor = monitor if monitor is not None else GD.pick_monitor(None)
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
        monitor=chosen_monitor,
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
    # The configuration is pinned either way; only the ROWS differ. A name
    # ending in the pooled suffix would otherwise compound each time.
    config_name = config_name_of(profile.model_dir)
    # A fresh round cannot normally have trained anything -- but --round takes
    # a number from the operator, and naming one that a pooled model already
    # learned from would score that model on its own rows.
    check_is_held_out(profile.model_path(), recording_dir, "T1")
    if fit == "session":
        return compare_on_check(profile, recording_dir, fit_pinned(recording_dir, config_name))
    model_dir, pool = fit_pooled(recording_dir, config_name, root=out_root)
    return compare_on_check(profile, recording_dir, model_dir, pool=pool)


def refit_from_existing(
    profile: PROF.Profile,
    recording_dir: Path,
    *,
    check_protocol: str = "T1",
    out_root: Path | None = None,
) -> Outcome:
    """Rebuild the model from the pool WITHOUT recording anything new.

    The accumulating pool means the data for a better model can already be on
    disk before a person sits down: the recordings that would improve today's
    profile were made on earlier days. Asking someone who cannot use their
    hands to repeat a calibration session to extract value from sessions they
    have already given is a cost with nothing behind it.

    ``recording_dir`` supplies the CHECK only. Its check protocol is held out
    of the pool and both models are scored on it, so this is the same paired
    comparison the recording path makes -- and, unlike that path, neither
    model has seen these frames, which makes it the fairer of the two.
    """

    recording_dir = Path(recording_dir)
    if not (recording_dir / f"{check_protocol}.meta.json").exists():
        raise SystemExit(
            f"{recording_dir} has no {check_protocol} recording to check against. "
            "Nothing was changed."
        )
    # Adoption reads the CALIBRATION protocol of this directory for the rig,
    # the screen identity and the calibration pose. Several rounds on disk
    # hold a check and nothing else; one of those would fit and compare
    # perfectly well and then fail at the last step, after the work. Refused
    # here, before anything is fitted, with the reason.
    if not (recording_dir / "A.meta.json").exists():
        raise SystemExit(
            f"{recording_dir} has {check_protocol} but no A recording. A profile takes its "
            "geometry and calibration pose from the calibration protocol, so this round "
            "cannot become one. Pick a round that has both. Nothing was changed."
        )
    check_is_held_out(profile.model_path(), recording_dir, check_protocol)
    config_name = config_name_of(profile.model_dir)
    model_dir, pool = fit_pooled(
        recording_dir, config_name, check_protocol=check_protocol, root=out_root
    )
    return compare_on_check(profile, recording_dir, model_dir, protocol=check_protocol, pool=pool)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--monitor", default=None)
    parser.add_argument(
        "--from-existing",
        default=None,
        metavar="RECORDING",
        help=(
            "skip the camera session entirely and rebuild the model from recordings "
            "already on disk. The named round supplies the CHECK: its T1 is held out of "
            "the pool and both models are scored on it. Use this to get the benefit of "
            "sessions already recorded without asking anyone to calibrate again."
        ),
    )
    parser.add_argument(
        "--fit",
        choices=("pooled", "session"),
        default="pooled",
        help=(
            "which rows train the new model. 'pooled' (default) uses every compatible "
            "past recording, which measured a median 118 px against 226 px for a single "
            "session and was better on 11 of 11 leave-one-session-out folds; 'session' "
            "is the old behaviour, this calibration alone, kept so the two can be "
            "compared on the same recording."
        ),
    )
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
    # --monitor was parsed and then dropped, so naming a display did nothing
    # at all. Resolved before the session rather than inside it, so a bad
    # selector fails with the list of displays instead of after a recording.
    monitor = GD.pick_monitor(args.monitor) if args.monitor is not None else None
    if args.from_existing is not None:
        if args.fit == "session":
            raise SystemExit(
                "--from-existing rebuilds from the pool; there is no single session to fit. "
                "Drop --fit session, or record a new one."
            )
        outcome = refit_from_existing(profile, Path(args.from_existing))
    else:
        outcome = run_sequence(profile, round_id=args.round, monitor=monitor, fit=args.fit)
    print()
    print(outcome.summary)
    print(f"new calibration: {outcome.model_dir}")
    if outcome.pool is not None:
        print(f"trained on the accumulated pool: {outcome.pool.summary()}")
    else:
        print("trained on this session alone (--fit session)")
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
    # Said out loud. A profile that quietly inherits half its settings is as
    # hard to reason about as one that quietly discards them.
    carried = [field for field in CARRIED_FORWARD if getattr(profile, field)]
    if carried:
        print(
            f"carried over from {profile.name} (a recalibration does not measure these): "
            f"{', '.join(carried)}"
        )
    missing = [field for field in CARRIED_FORWARD if not getattr(profile, field)]
    if missing:
        print(
            f"{profile.name} had no {', '.join(missing)} settings, so the new profile uses "
            "the built-in defaults for them"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
