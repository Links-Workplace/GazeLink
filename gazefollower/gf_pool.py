"""The accumulated training pool: every past recording, not the newest one (M2).

Why this module exists, measured rather than assumed.

A calibration session stores 45 accepted frames at each of **nine** target
positions -- 405 rows in a 258-dimensional feature space. The 45 frames at one
point are near-duplicates of each other, so the effective sample size is not
405, it is **nine**. Fitting 258 dimensions to nine points memorises them:
measured across three sessions, the model scored 17.8, 22.0 and 31.4 px on the
very rows it was trained on, and 167, 219 and 282 px on held-out check frames.
An order of magnitude apart. Every profile this project has shipped was a
draw from that lottery, and comparing two of them compared two draws.

Naive accumulation does NOT fix it, which is the finding that shaped this
module. Pooling several calibration sessions makes the error WORSE (measured:
307 px against 156 px for a single session), because another session of the
same nine points adds measurements, not coverage. What helps is pooling every
LABELLED recording -- calibration protocols and check protocols alike -- so
the pool spans many distinct screen positions:

    leave-one-session-out, filtered median error over 13 held-out sessions
      single fresh calibration (what recalibration built before)  226.2 px   (11 sessions had one)
      pooled, calibration+check protocols only, 35 positions      117.6 px   11/11 folds better
      the same but every poolable protocol, 63 positions          105.9 px   11/11 folds better
      calibration+check, and the held-out POSITIONS removed too   142.1 px   10/11 folds better

Three separate experiments, so the three numbers are not alternatives: the
middle two differ in which protocols were pooled, and 105.9 px is the one
measured through THIS module's ``build``. Eleven folds, not thirteen, carry a
comparison: rounds 3 and 33 have no single-session model of their own to be
compared against. The last row is the strict one -- the model is asked about
screen positions it was never trained on in any session -- and it still nearly
halves the error of a fresh calibration.

What this does NOT establish, measured and stated because the opposite is the
natural thing to assume: it is not monotonic. Training only on sessions
recorded EARLIER than the check (rather than on all other sessions) gave
144 px, 106 px, 209 px and 549 px over the first four checks before settling
near 80-120 px as the history grew. A pool can be large and still produce a
bad model. What makes the mechanism safe is not the pool but the gate around
it: a new model is adopted only when it beats the current one on a check
recording neither has trained on.

Two rules protect the measurement, and both are enforced here rather than
left to the caller:

* **The recording being scored is never in the pool.** Excluded by directory
  and protocol, not by round number, because one directory holds several
  protocols and only one of them is the check.
* **Only compatible recordings join.** A different feature dimension, screen
  or rig is a different measurement, and averaging it in would move error
  without recording why.

Reads recordings from disk and returns arrays. No camera, no model fitting,
no OS input, nothing written.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_fit as FIT  # noqa: E402
import gf_schema as S  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = PACKAGE_DIR / "recordings"

# Which protocols carry labelled rows worth training on. Calibration
# protocols contribute their controller-accepted rows; timed protocols
# contribute their COLLECTING rows -- the same mask the scorer predicts on,
# so a row is trained on exactly when it would have been scored.
#
# TUNE is deliberately absent. It exists to SELECT a configuration, and a
# configuration chosen on rows the model was then trained on is chosen on
# training error. Nothing else here is excluded: a check recording's rows are
# ordinary labelled observations once it is no longer being used as a check.
POOLABLE_PROTOCOLS = ("A", "B", "GRID16", "T1", "T2", "T3", "FULL")


@dataclass(frozen=True)
class Source:
    """One recording that joined the pool, and what it contributed."""

    directory: Path
    protocol: str
    round_id: int
    rows: int
    positions: int
    session: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "recording": str(self.directory),
            "protocol": self.protocol,
            "round_id": self.round_id,
            "rows": self.rows,
            "positions": self.positions,
            "session": self.session,
        }


@dataclass(frozen=True)
class Rejection:
    """A recording that did NOT join, and the reason.

    Kept and reported rather than dropped silently: a pool that quietly
    shrinks is a pool whose accuracy changes for reasons nobody can see.
    """

    directory: Path
    protocol: str
    reason: str


@dataclass
class Pool:
    """The training set assembled from every compatible past recording."""

    X: np.ndarray[Any, Any]
    Y_cm: np.ndarray[Any, Any]
    rig: C.RigGeometry
    sources: list[Source] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def positions(self) -> int:
        """Distinct labelled screen positions, the number that actually
        governs generalisation. Rounded to 0.1 cm: two targets nominally at
        the same place differ in the last float digit between sessions."""

        if self.Y_cm.size == 0:
            return 0
        return int(len(np.unique(np.round(self.Y_cm, 1), axis=0)))

    @property
    def sessions(self) -> int:
        return len({s.session for s in self.sources if s.session})

    def summary(self) -> str:
        return (
            f"{self.rows} rows from {len(self.sources)} recordings "
            f"({self.sessions} sessions), {self.positions} distinct screen positions"
        )

    def identities(self) -> list[tuple[str, str]]:
        """(round directory name, protocol) for every recording that trained.

        Stored in the fitted model so that a LATER comparison can tell whether
        a check recording is genuinely held out of the OLD model too. Names
        rather than absolute paths: a recordings folder that moves is the same
        measurement.
        """

        return [(s.directory.name, s.protocol) for s in self.sources]

    def provenance(self) -> dict[str, Any]:
        """What went in, small enough to store in a profile."""

        return {
            "rows": self.rows,
            "positions": self.positions,
            "sessions": self.sessions,
            "recordings": [s.as_dict() for s in self.sources],
            "rejected": [
                {"recording": str(r.directory), "protocol": r.protocol, "reason": r.reason}
                for r in self.rejected
            ],
        }


def compatibility_key(meta: dict[str, Any]) -> tuple[Any, ...]:
    """What must match for two recordings to describe the same measurement.

    The feature dimension is the model's input width and a mismatch cannot be
    trained at all. The rig and the screen are the ruler: the same eye
    geometry maps to a different centimetre on a different panel, so pooling
    across them would fold a geometry change into the gaze model, where it
    would read as scatter that no amount of data removes.

    Equal width is NOT equal meaning, which is the trap this key exists to
    close. ``builder_version`` and the embedding library identity can change
    while the column count and the column names stay the same: column 17 is
    then a different quantity in two recordings and averaging them trains on
    two measurements presented as one. That failure is silent -- it produces a
    model, and a worse one, with nothing recording why -- so the identity is
    pinned rather than trusted to be stable.
    """

    rig = meta.get("rig") or {}
    geometry = meta.get("target_geometry") or {}
    library = meta.get("library")
    if isinstance(library, dict):
        # Only the fields that change what a feature MEANS. Paths and
        # temporary files travel with a machine, not with a measurement.
        library = tuple(
            sorted(
                (k, str(v))
                for k, v in library.items()
                if k in ("name", "version", "model", "model_sha256", "weights_sha256")
            )
        )
    return (
        meta.get("feature_dim"),
        tuple(meta.get("head_names") or ()),
        meta.get("builder_version"),
        library,
        rig.get("camera_x_cm"),
        rig.get("camera_y_cm"),
        rig.get("screen_w_cm"),
        rig.get("screen_h_cm"),
        rig.get("device_w_px"),
        rig.get("device_h_px"),
        geometry.get("screen_id"),
        # How the frame reached the network. Recordings from before this
        # field existed were all taken through the library's own camera, so
        # the absence of the field means exactly that and they still pool.
        (meta.get("capture") or {}).get("pipeline", "640x480-library"),
    )


def key_of(recording_dir: Path, protocol: str) -> tuple[Any, ...]:
    """The compatibility key of one recording, read from its metadata alone.

    Exists so a caller can pin the pool to the setup being calibrated TODAY
    rather than to whichever recording happens to be oldest on disk.
    """

    meta = json.loads((Path(recording_dir) / f"{protocol}.meta.json").read_text(encoding="utf-8"))
    return compatibility_key(meta)


def trained_on(model_dir: Path) -> set[tuple[str, str]] | None:
    """Which (round, protocol) recordings trained a saved model.

    ``None`` means the model does not say -- an older model saved before the
    pool existed. That is not the same as "trained on nothing", and a caller
    deciding whether a check is held out must treat the two differently.
    """

    schema = Path(model_dir) / "schema.json"
    if not schema.exists():
        return None
    try:
        train = json.loads(schema.read_text(encoding="utf-8")).get("train") or {}
    except (OSError, ValueError):
        return None
    listed = train.get("trained_on")
    if listed is None:
        return None
    return {(str(entry[0]), str(entry[1])) for entry in listed}


def _labelled_rows(rec: S.Recording, head_names: Sequence[str]) -> np.ndarray[Any, Any]:
    """The rows of this recording that carry a usable label.

    Calibration and timed protocols name their collecting phase differently,
    and using the wrong mask is silent: ``training_rows`` on a check protocol
    returns an all-false mask, and that recording contributes NOTHING while
    every count still looks plausible. Measured while building this: a pool
    that was meant to add ten positions added zero and the run simply looked
    like a weaker result.
    """

    getter = FIT.training_rows if rec.protocol in S.CALIBRATION_PROTOCOLS else FIT.scoring_rows
    mask, _dropped = getter(rec, head_names)
    return mask


def discover(
    root: Path | None = None, *, protocols: Sequence[str] = POOLABLE_PROTOCOLS
) -> list[tuple[Path, str]]:
    """Every (directory, protocol) recording on disk, oldest round first.

    Ordered so that a pool built twice from the same disk is the same pool.
    """

    root = Path(root or RECORDINGS_DIR)
    if not root.exists():
        return []

    def order(directory: Path) -> tuple[int, str]:
        tail = directory.name[5:]
        return (int(tail) if tail.isdigit() else 1 << 30, directory.name)

    found: list[tuple[Path, str]] = []
    for directory in sorted(root.glob("round*"), key=order):
        if not directory.is_dir():
            continue
        for protocol in protocols:
            if (directory / f"{protocol}.meta.json").exists():
                found.append((directory, protocol))
    return found


def build(
    head_names: Sequence[str],
    *,
    root: Path | None = None,
    exclude: Iterable[tuple[Path, str]] = (),
    require: tuple[Any, ...] | None = None,
    protocols: Sequence[str] = POOLABLE_PROTOCOLS,
) -> Pool:
    """Assemble the pool.

    ``exclude`` names (directory, protocol) pairs that must not train -- the
    check recording the result will be scored on, above all. It is matched on
    the resolved directory, so a relative and an absolute path to the same
    recording exclude each other rather than one slipping through.

    ``require`` pins the compatibility key. When it is None the key of the
    FIRST accepted recording is adopted and everything after it must match,
    which makes the pool self-consistent even when nobody says what to expect.
    """

    excluded = {(Path(d).resolve(), p) for d, p in exclude}
    sources: list[Source] = []
    rejected: list[Rejection] = []
    Xs: list[np.ndarray[Any, Any]] = []
    Ys: list[np.ndarray[Any, Any]] = []
    rig: C.RigGeometry | None = None
    key = require

    for directory, protocol in discover(root, protocols=protocols):
        if (directory.resolve(), protocol) in excluded:
            rejected.append(Rejection(directory, protocol, "held out: this run is scored on it"))
            continue
        try:
            rec = S.Recording.load(directory, protocol)
        except Exception as exc:  # noqa: BLE001 - an unreadable recording is data, not a crash
            rejected.append(Rejection(directory, protocol, f"unreadable: {exc}"))
            continue
        got = compatibility_key(rec.meta)
        if key is None:
            key = got
        if got != key:
            rejected.append(
                Rejection(directory, protocol, "different feature space, rig or screen")
            )
            continue
        mask = _labelled_rows(rec, head_names)
        if not mask.any():
            rejected.append(Rejection(directory, protocol, "no labelled rows"))
            continue
        X = FIT.design_matrix(rec, mask, head_names)
        Y = rec.label_cm[mask]
        finite = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        if not finite.any():
            rejected.append(
                Rejection(directory, protocol, "every labelled row had a non-finite value")
            )
            continue
        if not finite.all():
            rejected.append(
                Rejection(directory, protocol, f"dropped {int((~finite).sum())} non-finite rows")
            )
        X, Y = X[finite], Y[finite]
        Xs.append(X)
        Ys.append(Y)
        rig = rig or C.RigGeometry.from_dict(rec.meta["rig"])
        sources.append(
            Source(
                directory=directory,
                protocol=protocol,
                round_id=int(rec.round_id),
                rows=int(X.shape[0]),
                positions=int(len(np.unique(np.round(Y, 1), axis=0))),
                session=str(rec.meta.get("session") or ""),
            )
        )

    if not Xs or rig is None:
        empty_x = np.zeros((0, 0), dtype=np.float64)
        return Pool(
            X=empty_x,
            Y_cm=np.zeros((0, 2), dtype=np.float64),
            rig=C.RigGeometry.from_dict(
                {
                    "camera_x_cm": 0.0,
                    "camera_y_cm": 0.0,
                    "screen_w_cm": 1.0,
                    "screen_h_cm": 1.0,
                    "device_w_px": 1,
                    "device_h_px": 1,
                }
            ),
            sources=sources,
            rejected=rejected,
        )
    return Pool(X=np.vstack(Xs), Y_cm=np.vstack(Ys), rig=rig, sources=sources, rejected=rejected)


# How small a pool stops being an improvement over one fresh calibration.
# A single calibration is 405 rows at 9 positions and measured 226 px; the
# pooled result that measured 142 px had 25 positions. These are floors, not
# targets: below them the pooled path has no evidence behind it and the
# caller is told so rather than being given a number that looks the same.
MIN_POOL_ROWS = 800
MIN_POOL_POSITIONS = 12


def usable(pool: Pool) -> tuple[bool, str]:
    """Is there enough here to be worth fitting? Reason returned either way."""

    if pool.rows < MIN_POOL_ROWS:
        return False, (
            f"only {pool.rows} rows in the pool, under the {MIN_POOL_ROWS} that the "
            "pooled fit was measured with; one calibration alone is 405"
        )
    if pool.positions < MIN_POOL_POSITIONS:
        return False, (
            f"only {pool.positions} distinct screen positions, under {MIN_POOL_POSITIONS}. "
            "More rows at the same few points is what was measured NOT to help"
        )
    return True, f"pool is usable: {pool.summary()}"
