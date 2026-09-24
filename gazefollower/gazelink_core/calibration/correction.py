"""A linear correction applied AFTER a fitted model, in screen fractions.

WHY THIS EXISTS AND WHAT IT IS NOT

Measured on the multi-pose candidate (2026-09-16): its predictions reproduce
the calibration points almost exactly -- x slope 0.985 to 0.988 on the rows it
was fitted on -- but on held-out targets BETWEEN those points the slope reads
1.23 to 1.25. The calibration grid has only three distinct x values and three
distinct y values, so every held-out target sits between columns the model
never saw, and the error there grows with distance from the nearest one
(r = 0.64).

A correction learned on three sittings and applied to a fourth, recorded
afterwards and used for nothing else, took its per-presentation error from
127 to 100 logical px filtered (median 132 to 92).

This is a SYMPTOM layer, and it is written as one on purpose:

  * it does not touch the model, which stays byte-identical on disk
  * it is one file next to the model and deleting the file removes it
  * the gain it applies is NOT a constant of the system. Measured per
    sitting it ranged 1.16 to 1.25, so a correction learned on one day may
    not be the right one on another. Every correction records which
    recordings it came from, and comparing that list against the run being
    corrected is the caller's job.

A previous attempt at a per-session OFFSET correction failed to transfer
between sittings. This is a different shape -- a gain, not a shift -- and
removing the best possible shift from the same data made the error WORSE
(134 to 137 px), so the two are not the same idea retried.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CORRECTION_FILE = "correction.json"
CORRECTION_VERSION = "axis-linear-1"


@dataclass(frozen=True)
class AxisCorrection:
    """``corrected = (predicted - offset) / gain``, independently per axis.

    Fitted the other way round -- predicted as a function of the target --
    so the numbers read as "what the model does" rather than "what undoes
    it": a gain of 1.24 means the model spreads the screen 24 % too wide.
    """

    gain_x: float
    offset_x: float
    gain_y: float
    offset_y: float
    source: tuple[str, ...] = ()
    note: str = ""
    version: str = CORRECTION_VERSION

    def __post_init__(self) -> None:
        for axis, gain in (("x", self.gain_x), ("y", self.gain_y)):
            if not np.isfinite(gain) or abs(gain) < 1e-6:
                raise ValueError(f"{axis} gain must be finite and non-zero, got {gain!r}")

    @classmethod
    def fit(
        cls,
        targets_norm: np.ndarray,
        predictions_norm: np.ndarray,
        *,
        source: Sequence[str] = (),
        note: str = "",
    ) -> "AxisCorrection":
        """Least squares, one axis at a time, on screen fractions.

        No cross terms: measured, they contributed nothing and cost accuracy
        on the moving check (126 px against 103 px for the per-axis form).
        Fewer parameters also means less of the held-out data spent on fitting
        the correction rather than testing it.
        """

        targets = np.asarray(targets_norm, dtype=np.float64)
        predictions = np.asarray(predictions_norm, dtype=np.float64)
        if targets.shape != predictions.shape or targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("targets and predictions must both be (n, 2)")
        ok = np.isfinite(targets).all(axis=1) & np.isfinite(predictions).all(axis=1)
        if int(ok.sum()) < 3:
            raise ValueError("need at least three finite pairs to fit a correction")
        gain_x, offset_x = np.polyfit(targets[ok, 0], predictions[ok, 0], 1)
        gain_y, offset_y = np.polyfit(targets[ok, 1], predictions[ok, 1], 1)
        return cls(
            gain_x=float(gain_x),
            offset_x=float(offset_x),
            gain_y=float(gain_y),
            offset_y=float(offset_y),
            source=tuple(source),
            note=note,
        )

    def apply(self, norm: np.ndarray) -> np.ndarray:
        """Correct an (n, 2) block of screen fractions. NaN rows stay NaN."""

        norm = np.asarray(norm, dtype=np.float64)
        out = np.empty_like(norm)
        out[..., 0] = (norm[..., 0] - self.offset_x) / self.gain_x
        out[..., 1] = (norm[..., 1] - self.offset_y) / self.gain_y
        return out

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source"] = list(self.source)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AxisCorrection":
        version = str(value.get("version", ""))
        if version != CORRECTION_VERSION:
            raise ValueError(
                f"correction version {version!r} is not {CORRECTION_VERSION!r}; refusing to "
                "guess what an unknown correction meant"
            )
        return cls(
            gain_x=float(value["gain_x"]),
            offset_x=float(value["offset_x"]),
            gain_y=float(value["gain_y"]),
            offset_y=float(value["offset_y"]),
            source=tuple(value.get("source") or ()),
            note=str(value.get("note", "")),
        )

    def save(self, directory: Path) -> Path:
        path = Path(directory) / CORRECTION_FILE
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> "AxisCorrection | None":
        path = Path(directory) / CORRECTION_FILE
        if not path.exists():
            return None
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def describe(self) -> str:
        return (
            f"x gain {self.gain_x:.4f} offset {self.offset_x:+.4f}, "
            f"y gain {self.gain_y:.4f} offset {self.offset_y:+.4f}"
            + (f"  [from {', '.join(self.source)}]" if self.source else "")
        )


class CorrectedModel:
    """A fitted model plus a correction, with the model's own interface.

    Presents ``schema`` and ``predict_norm`` so anything that predicts with a
    model -- the live view, the overlay, the scoring code -- works unchanged
    and cannot accidentally use the uncorrected point for one and the
    corrected point for another.

    The wrapped model is exposed as ``base`` so a caller can score both and
    compare, which is the only way the correction is allowed to be judged.
    """

    def __init__(self, base: Any, correction: AxisCorrection):
        self.base = base
        self.correction = correction

    @property
    def schema(self) -> Any:
        return self.base.schema

    @property
    def config(self) -> Any:
        return self.base.config

    def predict_norm(self, X_raw: np.ndarray, rig: Any) -> np.ndarray:
        return self.correction.apply(self.base.predict_norm(X_raw, rig))

    def predict_cm(self, X_raw: np.ndarray) -> np.ndarray:
        """Deliberately absent as a corrected quantity.

        The correction is defined on screen fractions. Returning a "corrected"
        centimetre would invite a caller to convert it a second time, so the
        raw model's centimetres are returned unchanged and anything that needs
        the correction must go through ``predict_norm``.
        """

        return self.base.predict_cm(X_raw)

    def __getattr__(self, name: str) -> Any:
        """Everything else is the wrapped model's, unchanged.

        Without this the wrapper silently REMOVES capabilities. It removed one:
        ``support_activation`` is how the recorder decides a model has frozen
        against today's conditions, the caller guards it with a broad except so
        a check can never abort a good session, and the preflight answers "ok"
        for anything it cannot measure. An AttributeError here therefore did
        not fail loudly -- it turned the frozen-model abort off, on exactly the
        run meant to validate the correction.

        Only reached for names this class does not define, so ``predict_norm``
        stays corrected and cannot be bypassed.

        ``base`` itself is refused rather than delegated. ``copy.deepcopy`` and
        ``pickle`` build an instance WITHOUT calling ``__init__`` and then look
        up ``__deepcopy__`` / ``__reduce_ex__``; with a plain delegation the
        lookup of ``self.base`` re-enters this method and recurses until the
        stack ends. Nothing in the project copies a model today, so this is a
        trap rather than a live bug -- but a RecursionError from a copy is a
        miserable thing to diagnose later.
        """

        if name == "base" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __repr__(self) -> str:
        return f"CorrectedModel({self.base!r}, {self.correction.describe()})"


def load_with_correction(directory: Path, loader: Any) -> Any:
    """Load a model and wrap it if a correction sits beside it.

    ``loader`` is passed in rather than imported so this module stays free of
    the fitting stack, which pulls in OpenCV.
    """

    model = loader(directory)
    correction = AxisCorrection.load(Path(directory))
    return model if correction is None else CorrectedModel(model, correction)
