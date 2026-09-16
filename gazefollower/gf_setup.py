"""The setup manifest: what the rig was when a recording was made (M0).

The measurement contract requires the camera, screen, geometry, machine and
code version to be recorded with every session and NOT changed quietly
between the sessions that decide anything. This module builds that manifest,
compares it against the previous one, and refuses to let a silent change pass
as a continuation.

Three kinds of field, kept apart because they carry different weight:

``measured``  read from the hardware or the OS (screen resolution and physical
              size, camera backend and frame size, CPU, actual frame rate when
              probed). Trustworthy without a person vouching for it.
``declared``  entered by the operator (camera position, eye distance, glasses,
              lighting). No sensor confirms these; treat every number derived
              from them as depending on them.
``derived``   hashes of the code and model weights, library versions. Exact,
              and what makes a run reproducible.

A declared physical screen size is cross-checked against the size the OS
reports. That is the one place a wrong number would silently rescale every
label in the experiment, so a disagreement is surfaced rather than averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_common as C  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_VERSION = "setup-1"

# Modules whose content defines the experiment's behaviour. A change to any of
# them makes a later recording a different experiment, so all are hashed.
CODE_MODULES = (
    "gazelink_core/domain/common.py",
    "gazelink_core/gaze/head_features.py",
    "gazelink_core/calibration/schema.py",
    "gf_fit.py",
    "gf_record.py",
    "gazelink_core/calibration/targets.py",
    "gf_geometry.py",
    "gf_report.py",
    "gf_setup.py",
    "gf_diagnose_bias.py",
    # Extracted from gf_record/gf_fit into the core (ARCH-01 stage D). They were
    # hashed before as part of those files, so they stay hashed now. The combined
    # code hash therefore differs from recordings made before 2026-09-15 by design.
    "gazelink_core/calibration/model.py",
    "gazelink_core/gaze/prediction.py",
    "gazelink_core/gaze/preflight.py",
    "gazelink_core/gaze/visibility.py",
    "gazelink_core/tracking/gazefollower_library.py",
    "gazelink_core/ui/prompt_layout.py",
    "gazelink_core/ui/pygame_display.py",
)

# Screen size the OS reports vs the operator's measurement: more than this and
# somebody measured the bezel, or the wrong monitor.
SCREEN_SIZE_TOLERANCE_MM = 20.0

# How long the camera probe may take, and how many frames it wants. Kept in
# one place so the value in the signature and the value the manifest passes
# cannot drift apart -- they had, and the docstring described the wrong bound.
CAMERA_PROBE_FRAMES = 30
CAMERA_PROBE_TIMEOUT_S = 8.0

# Fields that must not change between the sessions of one comparison. Camera
# position is here because moving the camera changes the geometry the model
# was calibrated in -- a separate experiment, not a continuation.
CRITICAL_FIELDS = (
    "measured.screen.width_px",
    "measured.screen.height_px",
    "measured.camera.frame_width",
    "measured.camera.frame_height",
    "declared.camera_x_cm",
    "declared.camera_y_cm",
    "declared.screen_w_cm",
    "declared.screen_h_cm",
    "derived.model_sha256",
    "derived.code_sha256",
)


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def code_fingerprint(modules: Sequence[str] = CODE_MODULES) -> dict[str, Any]:
    """Per-module hashes plus one hash over all of them, in a fixed order."""

    per_file: dict[str, str | None] = {}
    combined = hashlib.sha256()
    for name in sorted(modules):
        digest = sha256_file(PACKAGE_DIR / name)
        per_file[name] = digest
        combined.update(f"{name}:{digest}".encode())
    return {"files": per_file, "combined": combined.hexdigest()}


# GazeFollower is CC BY-NC-SA and stays installed in its own environment
# OUTSIDE this repository; only our code and our results live here. These are
# the places we look for that environment, in order.
LIBRARY_VENVS = (
    PACKAGE_DIR.parent.parent / "gazefollower_eval" / ".venv",
    PACKAGE_DIR / ".venv",
    PACKAGE_DIR.parent / ".venv",
)


def find_library_venv() -> Path | None:
    for candidate in LIBRARY_VENVS:
        if (candidate / "Lib" / "site-packages" / "gazefollower").is_dir():
            return candidate
    return None


def model_weights_hash() -> str | None:
    """Hash of the frozen MGazeNet weights, without importing the library."""

    venv = find_library_venv()
    if venv is None:
        return None
    candidates = list((venv / "Lib" / "site-packages" / "gazefollower" / "res" / "model_weights").glob("*.mnn"))
    if not candidates:
        return None
    return sha256_file(sorted(candidates)[0])


def probe_screen() -> dict[str, Any]:
    try:
        from screeninfo import get_monitors  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    monitors = []
    for m in get_monitors():
        monitors.append(
            {
                "name": m.name,
                "width_px": m.width,
                "height_px": m.height,
                "width_mm": getattr(m, "width_mm", None),
                "height_mm": getattr(m, "height_mm", None),
                "is_primary": bool(getattr(m, "is_primary", False)),
                "x": m.x,
                "y": m.y,
            }
        )
    primary = next((m for m in monitors if m["is_primary"]), monitors[0] if monitors else None)
    return {"monitors": monitors, **(primary or {})}


def _probe_camera_blocking(index: int, frames: int, deadline: float) -> dict[str, Any]:
    """The part that can block. Always called through a bounded wrapper."""

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return {"opened": False}
    # Ask for the same mode WebCamCamera uses, so the rate measured here is
    # the rate the recording will actually get rather than some other mode's.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    info: dict[str, Any] = {
        "opened": True,
        "index": index,
        "backend": cap.getBackendName(),
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "advertised_fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    stamps: list[float] = []
    try:
        for _ in range(frames):
            ok, _frame = cap.read()
            if not ok:
                break
            stamps.append(time.monotonic())
            if stamps[-1] > deadline:
                info["stopped_early"] = "deadline reached between frames"
                break
    finally:
        cap.release()
    if len(stamps) > 2:
        gaps = [b - a for a, b in zip(stamps[1:], stamps[2:]) if b > a]  # drop the first, warm-up
        if gaps:
            info["measured_fps"] = 1.0 / statistics.median(gaps)
            info["measured_frames"] = len(stamps)
    return info


def probe_camera(index: int = 0, frames: int = CAMERA_PROBE_FRAMES, timeout_s: float = CAMERA_PROBE_TIMEOUT_S) -> dict[str, Any]:
    """Open the camera briefly to record what it actually delivers.

    The driver's advertised FPS is routinely wrong or absent (this rig reports
    0.0), so the rate is MEASURED by timing real frames. Nothing is stored: the
    frames are counted and discarded.

    Bounded twice, because one bound is not enough. A deadline BETWEEN frames
    handles a camera that has merely slowed down -- low light makes many
    webcams stretch the exposure until they deliver about one frame a second.
    But ``VideoCapture()`` and ``read()`` block with no timeout of their own,
    so a camera another process is holding can park a single call forever and
    the deadline is never reached to be tested. The whole probe therefore runs
    through :func:`gf_common.call_with_timeout`.

    A timeout is reported as a failure with ``timed_out``, never as a missing
    measurement: the camera is stuck, the leaked thread is still holding it,
    and recording next would not work either.
    """

    deadline = time.monotonic() + timeout_s
    finished, result = C.call_with_timeout(
        lambda: _probe_camera_blocking(index, frames, deadline), timeout_s + 2.0
    )
    if not finished:
        return {
            "opened": None,
            "timed_out": True,
            "error": str(result),
            "note": (
                "The camera did not answer. Usually another program is holding it "
                "(video call, browser tab, or an earlier run that never exited). "
                "The probe thread is still blocked on it and will not release it "
                "until this process exits."
            ),
        }
    if isinstance(result, BaseException):
        return {"opened": False, "error": repr(result)}
    return result


def probe_machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import os  # noqa: PLC0415

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        info["cpu_count"] = None
    return info


@dataclass
class SetupManifest:
    measured: dict[str, Any]
    declared: dict[str, Any]
    derived: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    manifest_version: str = MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "measured": self.measured,
            "declared": self.declared,
            "derived": self.derived,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SetupManifest":
        if value.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(f"manifest version {value.get('manifest_version')!r} != {MANIFEST_VERSION!r}")
        return cls(
            measured=dict(value["measured"]),
            declared=dict(value["declared"]),
            derived=dict(value["derived"]),
            warnings=list(value.get("warnings", [])),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SetupManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, dotted: str) -> Any:
        node: Any = self.to_dict()
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        return node


def build_manifest(
    rig: C.RigGeometry,
    *,
    declared_extra: Mapping[str, Any] | None = None,
    probe_camera_frames: int = CAMERA_PROBE_FRAMES,
    camera_index: int = 0,
    skip_camera: bool = False,
) -> SetupManifest:
    screen = probe_screen()
    camera = {"opened": None, "skipped": True} if skip_camera else probe_camera(camera_index, probe_camera_frames)
    measured = {"screen": screen, "camera": camera, "machine": probe_machine()}
    declared = dict(rig.to_dict())
    declared.update(
        {
            "eye_distance_cm": None,
            "glasses": None,
            "lighting": None,
            "operator_note": None,
        }
    )
    declared.update(dict(declared_extra or {}))
    derived = {
        "code_sha256": code_fingerprint()["combined"],
        "code_files": code_fingerprint()["files"],
        "model_sha256": model_weights_hash(),
        "libraries": _library_versions(),
    }
    manifest = SetupManifest(measured=measured, declared=declared, derived=derived)
    manifest.warnings = check_consistency(manifest, rig)
    return manifest


# Distribution name -> the name used in the manifest. Versions come from the
# package metadata rather than by importing the module: importing pygame
# prints a banner and importing mediapipe loads native code, neither of which
# a version string is worth.
_LIBRARY_DISTRIBUTIONS = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "mediapipe": "mediapipe",
    "screeninfo": "screeninfo",
    "pygame": "pygame",
    "MNN": "mnn",
}


def _library_versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    versions: dict[str, Any] = {"python": platform.python_version()}
    for dist, label in _LIBRARY_DISTRIBUTIONS.items():
        try:
            versions[label] = version(dist)
        except PackageNotFoundError:
            versions[label] = None
        except Exception:  # noqa: BLE001 - absence is itself the information
            versions[label] = None
    return versions


def check_consistency(manifest: SetupManifest, rig: C.RigGeometry) -> list[str]:
    """Cross-checks that catch a wrong declared number before it rescales everything."""

    warnings: list[str] = []
    screen = manifest.measured.get("screen", {})
    px_w, px_h = screen.get("width_px"), screen.get("height_px")
    if px_w and px_h and (px_w, px_h) != (rig.device_w_px, rig.device_h_px):
        warnings.append(
            f"declared device resolution {rig.device_w_px}x{rig.device_h_px} != OS-reported {px_w}x{px_h}; "
            "labels would be computed in a different pixel space than the library uses"
        )
    mm_w, mm_h = screen.get("width_mm"), screen.get("height_mm")
    if mm_w:
        delta = abs(mm_w - rig.screen_w_cm * 10.0)
        if delta > SCREEN_SIZE_TOLERANCE_MM:
            warnings.append(
                f"declared screen width {rig.screen_w_cm} cm vs OS-reported {mm_w / 10:.1f} cm "
                f"(differs by {delta / 10:.1f} cm)"
            )
    if mm_h:
        delta = abs(mm_h - rig.screen_h_cm * 10.0)
        if delta > SCREEN_SIZE_TOLERANCE_MM:
            warnings.append(
                f"declared screen height {rig.screen_h_cm} cm vs OS-reported {mm_h / 10:.1f} cm "
                f"(differs by {delta / 10:.1f} cm)"
            )
    if not rig.camera_below_screen and rig.camera_y_cm >= 0:
        warnings.append(
            f"camera_y_cm {rig.camera_y_cm} places the camera within the screen area; "
            "this rig's camera is below the bottom edge (y greater than the screen height)"
        )
    camera = manifest.measured.get("camera", {})
    fps = camera.get("measured_fps")
    if fps is not None and fps < 25:
        warnings.append(f"camera delivered only {fps:.1f} fps; the sampling-rate goal needs at least 30")
    if manifest.derived.get("model_sha256") is None:
        warnings.append("model weights not found: the run cannot be tied to a specific frozen model")
    return warnings


def compare_manifests(previous: SetupManifest, current: SetupManifest) -> dict[str, Any]:
    """What changed between two sessions, and does it break comparability?

    A change to a critical field means the two recordings were made on
    different rigs or different code; treating them as one series would
    attribute the difference to the thing under test.
    """

    changes: list[dict[str, Any]] = []
    for dotted in CRITICAL_FIELDS:
        before, after = previous.get(dotted), current.get(dotted)
        if before != after:
            changes.append({"field": dotted, "before": before, "after": after})
    result: dict[str, Any] = {"critical_changes": changes, "comparable": not changes}
    before_files = set((previous.get("derived.code_files") or {}).keys())
    after_files = set((current.get("derived.code_files") or {}).keys())
    if before_files and after_files and before_files != after_files:
        # Not a silent "code changed": the module LIST itself differs, which
        # is what ARCH-01 did on 2026-09-15 when library code moved into
        # gazelink_core. Hashes of different file sets cannot be compared.
        result["code_layout_changed"] = {
            "only_before": sorted(before_files - after_files),
            "only_after": sorted(after_files - before_files),
            "note": (
                "the set of hashed code files changed (ARCH-01 moved modules into "
                "gazelink_core on 2026-09-15); sessions on either side are different "
                "experiments by code fingerprint"
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-x-cm", type=float, required=True)
    parser.add_argument("--camera-y-cm", type=float, required=True)
    parser.add_argument("--screen-width-cm", type=float, required=True)
    parser.add_argument("--screen-height-cm", type=float, required=True)
    parser.add_argument("--device-w", type=int, default=5120)
    parser.add_argument("--device-h", type=int, default=1440)
    parser.add_argument("--eye-distance-cm", type=float, default=None, help="operator-measured eye-to-screen distance")
    parser.add_argument("--glasses", default=None, help="none | glasses | contacts")
    parser.add_argument("--lighting", default=None, help="short description, kept constant across sessions")
    parser.add_argument("--note", default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the manifest here")
    parser.add_argument("--compare", type=Path, default=None, help="previous manifest to compare against")
    parser.add_argument("--skip-camera", action="store_true", help="do not open the camera")
    args = parser.parse_args(argv)

    rig = C.RigGeometry(args.camera_x_cm, args.camera_y_cm, args.screen_width_cm, args.screen_height_cm, args.device_w, args.device_h)
    manifest = build_manifest(
        rig,
        declared_extra={
            "eye_distance_cm": args.eye_distance_cm,
            "glasses": args.glasses,
            "lighting": args.lighting,
            "operator_note": args.note,
        },
        skip_camera=args.skip_camera,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    for warning in manifest.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if args.compare:
        comparison = compare_manifests(SetupManifest.load(args.compare), manifest)
        print(json.dumps(comparison, indent=2))
        if not comparison["comparable"]:
            print("SETUP CHANGED: these recordings are not one series", file=sys.stderr)
    if args.out:
        manifest.save(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
