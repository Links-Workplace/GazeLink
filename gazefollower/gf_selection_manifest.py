"""Manifest of the models behind a selection comparison. Metadata only.

For every model a comparison log names, this records what can be known
without a camera: file hashes, feature schema and scaling, SVR parameters,
the rig, the capture pipeline, the recordings it was trained on (with each
recording's monitor geometry and capture block), the software versions, and
whether any recording used for evaluation in the log was also in training.

Nothing biometric is copied: no feature arrays, no scaler vectors, no rows.
Scaler statistics are summarised as their length only.

Each field is tagged with where it stands:
  available            -- read from a file that exists
  missing_field        -- the file exists but does not carry it
  needs_new_image      -- only a new in-memory frame could tell (never
                          recoverable from stored features)

Usage (from gazefollower/):
    python gf_selection_manifest.py results/select_compare/20260915_073919.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RECORDINGS = HERE / "recordings"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tag(value: Any, status: str | None = None) -> dict[str, Any]:
    return {
        "value": value,
        "status": status or ("available" if value is not None else "missing_field"),
    }


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=HERE, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain", "--", ".")
    return {"commit": run("rev-parse", "HEAD"), "gazefollower_dir_dirty": bool(status)}


def recording_dir(model_dir: Path, round_name: str) -> Path | None:
    """Where a training round lives: a sibling tree for the hi pipeline, else the main one."""

    candidates = []
    parts = model_dir.parts
    if "hi" in parts:
        candidates.append(RECORDINGS / "hi" / round_name)
    if "resolution" in parts:
        candidates.append(RECORDINGS / "resolution" / round_name)
    candidates.append(RECORDINGS / round_name)
    for c in candidates:
        if c.exists():
            return c
    return None


def recording_entry(model_dir: Path, round_name: str, protocol: str) -> dict[str, Any]:
    folder = recording_dir(model_dir, round_name)
    meta_path = None if folder is None else folder / f"{protocol}.meta.json"
    if meta_path is None or not meta_path.exists():
        return {"round": round_name, "protocol": protocol, "meta": tag(None)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    monitor = meta.get("monitor") or {}
    capture = meta.get("capture")
    width = monitor.get("width_px")
    height = monitor.get("height_px")
    return {
        "round": round_name,
        "protocol": protocol,
        "path": str(meta_path.relative_to(HERE)),
        "session": tag(meta.get("session")),
        "n_rows": tag(meta.get("n_rows")),
        "feature_dim": tag(meta.get("feature_dim")),
        "builder_version": tag(meta.get("builder_version")),
        "monitor_px": tag([width, height] if width and height else None),
        # 4096x1152 is the 5120x1440 panel reported through 125 % scaling;
        # both are normalised before use, but the pair is recorded as found.
        "monitor_px_kind": tag(
            None
            if not width
            else ("logical_125pct" if width == 4096 else "device" if width == 5120 else "other")
        ),
        # Early recordings carry no monitor block, only the rig's configured
        # size: that is what the recorder was TOLD, not what Windows reported.
        "rig_device_px": tag(
            [meta["rig"].get("device_w_px"), meta["rig"].get("device_h_px")]
            if isinstance(meta.get("rig"), dict)
            else None
        ),
        "capture": tag(capture),
        "overlay_filter": tag(meta.get("overlay_filter")),
        "library_versions": tag(meta.get("versions")),
        "npz_sha256": tag(sha256(meta_path.with_name(f"{protocol}.npz"))),
    }


def model_entry(model_dir: Path) -> dict[str, Any]:
    schema_path = model_dir / "schema.json"
    if not schema_path.exists():
        return {"dir": str(model_dir), "schema": tag(None)}
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    train = schema.get("train") or {}
    feats = schema.get("features") or {}
    trained_on = [tuple(x) for x in train.get("trained_on") or []]
    return {
        "dir": str(model_dir.relative_to(HERE))
        if model_dir.is_relative_to(HERE)
        else str(model_dir),
        "files_sha256": {p.name: sha256(p) for p in sorted(model_dir.iterdir()) if p.is_file()},
        "schema_version": tag(schema.get("schema_version")),
        "feature_dim": tag(schema.get("base_dim")),
        "feature_order": tag(
            None
            if not schema.get("columns")
            else f"{schema['columns'][0]}..{schema['columns'][-1]} ({len(schema['columns'])})"
        ),
        "head_names": tag(schema.get("head_names")),
        "feature_scaling": tag(schema.get("feature_scaling")),
        "scaler_lengths": tag(
            {"mean": len(feats.get("mean") or []), "std": len(feats.get("std") or [])}
            if feats
            else None
        ),
        "label_scaling": tag(schema.get("label_scaling")),
        "svr": tag(schema.get("svr")),
        "svr_backend": tag(
            "OpenCV ml.SVM EPS_SVR RBF" if (model_dir / "svr_x.xml").exists() else None
        ),
        "rig": tag(schema.get("rig")),
        "builder_version": tag(schema.get("builder_version")),
        "fitted_by": tag(train.get("fitted_by")),
        "train_protocol": tag(train.get("protocol")),
        "capture_pipeline": tag(train.get("capture_pipeline")),
        "n_rows": tag(train.get("n_rows")),
        "versions": tag(schema.get("versions")),
        "extractor_weights_hash": tag(None, "missing_field"),
        "sample_weighting": tag(None, "missing_field"),
        "eye_crop_detail_quality": tag(None, "needs_new_image"),
        "trained_on": [recording_entry(model_dir, r, p) for r, p in trained_on],
        "_trained_on_keys": [f"{r}/{p}" for r, p in trained_on],
    }


def leakage(entry: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Recordings a block used for evaluation that are also in this model's training."""

    keys = set(entry.get("_trained_on_keys") or [])
    out = []
    source = str((report.get("bias_used") or {}).get("source") or "")
    if source.startswith("measured on "):
        used = source.removeprefix("measured on ").strip()
        if used in keys:
            out.append(
                f"bias_used for layout warnings was measured on {used}, which this model trained on"
            )
    t1 = sorted(k for k in keys if k.endswith("/T1"))
    if t1:
        out.append(
            f"{len(t1)} T1 (held-out protocol) recording(s) are in training, e.g. {t1[:3]}; "
            "no T1 of those rounds is a held-out test for this model"
        )
    return out


def build(log: Path) -> dict[str, Any]:
    doc = json.loads(log.read_text(encoding="utf-8"))
    blocks = doc.get("blocks") or [{"block": 0, "arm": "-", "report": doc}]
    models: dict[str, dict[str, Any]] = {}
    for name, path in (doc.get("models") or {}).items():
        models[name] = model_entry(Path(path))
    if not models:
        models[str(doc.get("profile"))] = model_entry(Path(doc["model"]))
    findings = []
    for b in blocks:
        rep = b["report"]
        entry = models.get(rep.get("profile"))
        if entry is None:
            continue
        for f in leakage(entry, rep):
            line = f"{rep.get('profile')}: {f}"
            if line not in findings:
                findings.append(line)
    for entry in models.values():
        entry.pop("_trained_on_keys", None)
    screens = sorted(
        {
            str(r.get("monitor_px", {}).get("value"))
            for e in models.values()
            for r in e.get("trained_on", [])
        }
    )
    return {
        "log": str(log),
        "git": git_state(),
        "settings": doc.get("settings")
        or {k: doc.get(k) for k in ("dwell_ms", "trial_timeout_s", "neutral_timeout_s")},
        "models": models,
        "training_monitor_geometries": screens,
        "findings": findings,
        "not_recoverable_offline": [
            "capture timestamps (logs carry read-return / display-loop time only)",
            "eye crop detail quality (needs a new in-memory image)",
            "extractor weights hash (not recorded in model or recording metadata)",
        ],
        "contains_biometric_arrays": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Metadata-only manifest of models in a comparison log."
    )
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "selection_diag")
    args = ap.parse_args(argv)
    manifest = build(args.log)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.log.stem}.manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, m in manifest["models"].items():
        print(
            f"{name}: {m.get('fitted_by', {}).get('value')} | "
            f"rows {m.get('n_rows', {}).get('value')} | "
            f"capture {m.get('capture_pipeline', {}).get('value')} | trained on "
            f"{len(m.get('trained_on', []))} recordings"
        )
    print("training monitor geometries:", manifest["training_monitor_geometries"])
    for f in manifest["findings"]:
        print("FINDING:", f)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
