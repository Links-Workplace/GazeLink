"""A saved profile: which model to load, how to filter it, and what it scored.

Every live run so far needed nine command-line flags naming the model, the
filter, the camera position and the screen size.  That is not a thing anyone
can be asked to retype to use their own computer, and a mistyped geometry does
not fail loudly -- it silently maps the prediction through the wrong screen and
reads as lost accuracy.

A profile stores that whole set once, together with the evidence for it: which
recording the model was fitted from, what it measured, and the head pose the
person held while calibrating.  The pose is not decoration.  Error tracks
drift away from the calibration pose more strongly than it tracks elapsed
time, so "how far are you sitting from where you calibrated" is the first
thing to report next to any accuracy number, and it cannot be computed without
knowing where that was.

Nothing here loads a model, opens a camera or emits OS input; it reads and
writes small JSON files describing where those things are.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402
import gf_gesture as GEST  # noqa: E402
import gf_gaze_filter as GF  # noqa: E402
import gf_schema as S  # noqa: E402

PROFILE_VERSION = "profile-1"
PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
ACTIVE_POINTER = "active.json"

# A profile names a model by path.  Storing the geometry it was fitted under
# lets a mismatch be reported rather than silently mapped through the wrong
# screen; these are the fields whose disagreement changes the prediction.
RIG_FIELDS = (
    "camera_x_cm",
    "camera_y_cm",
    "screen_w_cm",
    "screen_h_cm",
    "device_w_px",
    "device_h_px",
)


@dataclass(frozen=True)
class Profile:
    """One saved way of running: model + filter + geometry + its evidence."""

    name: str
    model_dir: str
    rig: dict[str, Any]
    filter: dict[str, Any]
    # Which physical display this was fitted on. Resolution alone does not
    # identify a monitor -- two can share one -- and a profile replayed on
    # a different panel maps confidently to the wrong place. This is the
    # identity M3-00 requires before a real cursor may be moved.
    screen: dict[str, Any] = field(default_factory=dict)
    # What counts as a deliberate eyelid gesture, for THIS person. Kept in
    # the profile rather than in the code because it is a property of a face
    # and a camera geometry, not of the algorithm: the same numbers that work
    # here are meaningless on another rig, and hard-coding them is how a
    # gesture ends up tuned for whoever last ran it.
    gesture: dict[str, Any] = field(default_factory=dict)
    # How the POINTER should feel. Deliberately separate from ``filter``:
    # that one belongs to the model and to the configuration verified on
    # the selection task, and must not be retuned to make a cursor
    # comfortable. These are presentation only and safe to change.
    cursor: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    pose: dict[str, float] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    x_range: list[float] | None = None
    created_utc: str = ""
    version: str = PROFILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Profile:
        got = str(value.get("version", ""))
        if got != PROFILE_VERSION:
            raise ValueError(f"profile version {got!r} is not {PROFILE_VERSION!r}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in value.items() if k in known})

    def rig_geometry(self) -> C.RigGeometry:
        return C.RigGeometry.from_dict(self.rig)

    def filter_settings(self) -> GF.FilterSettings:
        """Rebuild the filter, taking the pixel size from the rig.

        Width and height are deliberately NOT stored in ``filter``: they are a
        property of the screen, already in ``rig``, and keeping one copy means
        a profile cannot describe a filter sized for a screen it is not for.
        """

        rig = self.rig_geometry()
        values = {k: v for k, v in self.filter.items() if k not in ("width_px", "height_px")}
        # JSON has no enums, so the kind comes back as a plain string and
        # FilterSettings rejects it.  Coerce here rather than storing the enum
        # repr, so a profile stays readable and editable by hand.
        if "kind" in values:
            values["kind"] = GF.FilterKind(values["kind"])
        return GF.FilterSettings(width_px=rig.device_w_px, height_px=rig.device_h_px, **values)

    def wink_config(self) -> GEST.WinkConfig:
        """The person's wink rule, or the built-in default if none is saved."""

        saved = (self.gesture or {}).get("wink") or {}
        known = {f for f in GEST.WinkConfig.__dataclass_fields__}
        return GEST.WinkConfig(**{k: v for k, v in saved.items() if k in known})

    def gate_config(self) -> GEST.OpennessGateConfig:
        """How far an eye must close, relative to its own baseline, for them."""

        saved = (self.gesture or {}).get("gate") or {}
        known = {f for f in GEST.OpennessGateConfig.__dataclass_fields__}
        return GEST.OpennessGateConfig(**{k: v for k, v in saved.items() if k in known})

    def model_path(self) -> Path:
        path = Path(self.model_dir)
        return path if path.is_absolute() else (PROFILES_DIR.parent / path)


def filter_to_dict(settings: GF.FilterSettings) -> dict[str, Any]:
    """The filter minus its pixel size, with the kind as a plain string."""

    out = {k: v for k, v in asdict(settings).items() if k not in ("width_px", "height_px")}
    out["kind"] = GF.FilterKind(settings.kind).value
    return out


def calibration_pose(rec: S.Recording) -> dict[str, float]:
    """Median of each head feature over the rows the fit actually used.

    Taken over accepted rows rather than every row: the frames between targets
    are the person moving to the next point, and including them would describe
    a pose nobody held.
    """

    names = list(rec.meta.get("head_names") or [])
    if rec.head is None or not names:
        return {}
    rows = rec.rows_accepted() if rec.protocol in S.CALIBRATION_PROTOCOLS else rec.rows_collecting()
    if not np.any(rows):
        return {}
    head = rec.head[rows]
    return {name: float(np.median(head[:, i])) for i, name in enumerate(names)}


def pose_delta(profile: Profile, rec: S.Recording) -> dict[str, float]:
    """How far this recording's pose sits from the profile's calibration pose.

    Signed per feature, so the direction is visible and not only the size.
    Empty when either side has no head columns, rather than a fabricated zero.
    """

    here = calibration_pose(rec)
    there = profile.pose
    return {name: here[name] - there[name] for name in here if name in there}


def rig_mismatch(profile: Profile, rig: C.RigGeometry) -> list[str]:
    """Field-by-field disagreement between a profile and the current rig.

    Returned rather than raised: whether a mismatch is fatal is the caller's
    decision, and a 1 mm difference in a declared screen width should not
    refuse to start when a changed resolution must.
    """

    current = rig.to_dict()
    out: list[str] = []
    for name in RIG_FIELDS:
        want, got = profile.rig.get(name), current.get(name)
        if want is None or got is None:
            continue
        if isinstance(want, int) and isinstance(got, int):
            differs = want != got
        else:
            differs = abs(float(want) - float(got)) > 0.05
        if differs:
            out.append(f"{name}: profile {want}, current {got}")
    return out


def from_recording(
    name: str,
    *,
    model_dir: Path,
    recording_dir: Path,
    protocol: str = "A",
    settings: GF.FilterSettings,
    scores: dict[str, Any] | None = None,
    x_range: tuple[float, float] | None = None,
) -> Profile:
    """Build a profile from the recording its model was fitted on."""

    rec = S.Recording.load(recording_dir, protocol)
    return Profile(
        name=name,
        model_dir=str(model_dir),
        rig=dict(rec.meta["rig"]),
        screen=screen_identity(rec),
        filter=filter_to_dict(settings),
        calibration={
            "recording": str(recording_dir),
            "protocol": protocol,
            "round_id": rec.round_id,
            "session": rec.meta.get("session"),
        },
        pose=calibration_pose(rec),
        scores=dict(scores or {}),
        x_range=None if x_range is None else [float(x_range[0]), float(x_range[1])],
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# --- storage -----------------------------------------------------------------


def screen_identity(rec: S.Recording) -> dict[str, Any]:
    """What identifies the physical display this recording was made on.

    ``screen_id`` is the panel's own model string and is the strongest of
    these; the connector name (``\.\DISPLAY1``) is a slot, not a monitor, and
    moves between panels. Physical millimetres are kept as a second check
    because they survive a resolution change and differ between panels that
    share one.
    """

    geometry = rec.meta.get("target_geometry") or {}
    monitor = rec.meta.get("monitor") or {}
    return {
        "screen_id": geometry.get("screen_id"),
        "connector": monitor.get("name"),
        "width_px": monitor.get("width_px"),
        "height_px": monitor.get("height_px"),
        "width_mm": monitor.get("width_mm"),
        "height_mm": monitor.get("height_mm"),
    }


def profile_path(name: str, root: Path | None = None) -> Path:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"invalid profile name: {name!r}")
    return (root or PROFILES_DIR) / f"{name}.json"


def save(profile: Profile, *, root: Path | None = None, overwrite: bool = False) -> Path:
    """Write a profile, refusing to replace one unless asked.

    A profile is the pointer to the setup someone is relying on.  Silently
    replacing it is how a known-good configuration disappears with nothing left
    to go back to, so the default is to refuse.
    """

    directory = root or PROFILES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = profile_path(profile.name, directory)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"profile {profile.name!r} already exists at {path}; pass overwrite=True to replace it"
        )
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def load(name: str, *, root: Path | None = None) -> Profile:
    path = profile_path(name, root)
    if not path.exists():
        raise FileNotFoundError(f"no profile named {name!r} at {path}")
    return Profile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_profiles(*, root: Path | None = None) -> list[str]:
    directory = root or PROFILES_DIR
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json") if p.name != ACTIVE_POINTER)


def activate(name: str, *, root: Path | None = None) -> Path:
    """Point "the profile to use" at an existing profile.

    Loaded first: activating a name that does not resolve would leave the
    pointer describing something that cannot be started.
    """

    directory = root or PROFILES_DIR
    load(name, root=directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / ACTIVE_POINTER
    pointer.write_text(json.dumps({"active": name}, indent=2), encoding="utf-8")
    return pointer


def active_name(*, root: Path | None = None) -> str | None:
    pointer = (root or PROFILES_DIR) / ACTIVE_POINTER
    if not pointer.exists():
        return None
    value = json.loads(pointer.read_text(encoding="utf-8")).get("active")
    return None if value is None else str(value)


def load_active(*, root: Path | None = None) -> Profile | None:
    name = active_name(root=root)
    return None if name is None else load(name, root=root)


def describe(profile: Profile) -> str:
    lines = [
        f"profile {profile.name}  (created {profile.created_utc or 'unknown'})",
        f"  model   {profile.model_dir}",
        f"  screen  {profile.rig.get('device_w_px')}x{profile.rig.get('device_h_px')} px, "
        f"{profile.rig.get('screen_w_cm')}x{profile.rig.get('screen_h_cm')} cm",
        f"  filter  {profile.filter.get('kind')} "
        f"fc={profile.filter.get('one_euro_min_cutoff_hz')} "
        f"beta={profile.filter.get('one_euro_beta_hz_per_px_s')}",
    ]
    cal = profile.calibration
    if cal:
        lines.append(
            f"  fitted from {cal.get('recording')} / {cal.get('protocol')} "
            f"(round {cal.get('round_id')})"
        )
    if profile.scores:
        lines.append(f"  scored  {json.dumps(profile.scores, sort_keys=True)}")
    if profile.pose:
        pose = "  ".join(f"{k}={v:+.3f}" for k, v in profile.pose.items())
        lines.append("  calibration pose  " + pose)
    return "\n".join(lines)
