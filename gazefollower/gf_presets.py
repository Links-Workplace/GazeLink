"""Named, reproducible calibration configurations.

A preset freezes the METHOD -- features, preprocessing, hyperparameters and
the library versions they were characterised against. It does not freeze a
fitted model. Every screen and every person needs a fresh personal
calibration; reusing another display's fitted SVR would be reusing a mapping
built for a geometry that no longer applies.

Each entry records where its numbers came from and on which data split, so a
tuning-set result can never be mistaken for independent validation.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_fit as F  # noqa: E402
import gf_schema as S  # noqa: E402

PRESET_VERSION = "preset-1"


@dataclass(frozen=True)
class Preset:
    """A calibration configuration plus its provenance."""

    name: str
    config: F.FitConfig
    summary: str
    evidence: str
    evidence_split: str  # "TUNE" | "T1" | "none"
    caveat: str = ""
    characterised_versions: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_version": PRESET_VERSION,
            "name": self.name,
            "config": self.config.to_dict(),
            "summary": self.summary,
            "evidence": self.evidence,
            "evidence_split": self.evidence_split,
            "caveat": self.caveat,
            "characterised_versions": dict(self.characterised_versions),
            "note": (
                "A preset freezes the method and settings, never a fitted model. "
                "A new screen or a new person requires fresh personal calibration."
            ),
        }


# Versions the presets below were characterised against. A different version
# does not invalidate a preset, but it does mean the numbers were not measured
# on that stack.
_CHARACTERISED = {"python": "3.11.9", "numpy": "2.4.6", "cv2": "4.14.0", "gazefollower": "1.0.2"}

PRESETS: dict[str, Preset] = {
    "library-default": Preset(
        name="library-default",
        config=F.library_default_config(),
        summary="GazeFollower's own SVR settings: RBF, C=1, gamma=0.005, no feature or label scaling.",
        evidence=(
            "The only configuration scored on the independent held-out set in every round. "
            "T1 median Euclidean error: round0 332 px, round1 566 px, round2 (central band) 285 px."
        ),
        evidence_split="T1",
        caveat="Fails the project's 120 px reference on X in every round measured so far.",
        characterised_versions=_CHARACTERISED,
    ),
    "central-band-svr": Preset(
        name="central-band-svr",
        config=F.FitConfig(
            name="central-band-svr",
            feature_scaling="zscore",
            label_scaling="zscore",
            C=100.0,
            gamma=0.0005,
            P=0.001,
        ),
        summary=(
            "z-scored 258-d features and z-scored centimetre labels, RBF SVR with C=100 and "
            "gamma=0.0005. Standardising both sides is what lets a single C mean the same thing "
            "on an axis spanning 114 cm and one spanning 31 cm."
        ),
        evidence=(
            "Best-behaved candidate in the round2 central-band sweep ON THE TUNING SET: "
            "Euclidean median 141 px, Euclidean P90 239 px, |dx| median 52 px, |dy| median 107 px, "
            "no off-screen predictions."
        ),
        evidence_split="TUNE",
        caveat=(
            "TUNING-SET NUMBERS ONLY. This configuration has never been scored on an independent "
            "test set; it was not among the models T1 was scored with in any round. Its numbers "
            "are a reason to carry it forward as a candidate, not evidence of accuracy."
        ),
        characterised_versions=_CHARACTERISED,
    ),
    "central-band-ridge": Preset(
        name="central-band-ridge",
        config=F.FitConfig(
            name="central-band-ridge",
            feature_scaling="zscore",
            label_scaling="zscore",
            kind="ridge",
            alpha=100.0,
        ),
        summary="Centred ridge on z-scored features and labels, alpha=100.",
        evidence=(
            "Chosen by the old median-only selector in round2 on a TUNE median of 138 px, then "
            "measured on T1 at Euclidean P90 142,863 px, maximum 170,788 px, and 102 off-screen "
            "predictions."
        ),
        evidence_split="T1",
        caveat=(
            "RETAINED AS A NEGATIVE CONTROL, not for use. It is the candidate whose tail the "
            "screening in gf_select.py exists to catch, and the regression tests use it as such."
        ),
        characterised_versions=_CHARACTERISED,
    ),
}

DEFAULT_PRESET = "central-band-svr"


def get(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {', '.join(sorted(PRESETS))}")
    return PRESETS[name]


def config_for(name: str) -> F.FitConfig:
    return get(name).config


def sweep_with_presets(head_names: tuple[str, ...] = ()) -> list[F.FitConfig]:
    """The standard sweep plus every preset, de-duplicated by name.

    Presets ride along in the sweep so a run can compare them against the grid
    and so ``--preset`` always has its candidate present to screen.
    """

    grid = list(F.phase0_grid(head_names))
    seen = {c.name for c in grid}
    for preset in PRESETS.values():
        cfg = preset.config
        if head_names and not cfg.head_names:
            cfg = F.FitConfig(**{**cfg.to_dict(), "head_names": head_names})
        if cfg.name not in seen:
            grid.append(cfg)
            seen.add(cfg.name)
    return grid


def describe(name: str) -> str:
    p = get(name)
    lines = [f"### Preset `{p.name}`", "", p.summary, ""]
    lines.append(f"- **Evidence** ({p.evidence_split}): {p.evidence}")
    if p.caveat:
        lines.append(f"- **Caveat**: {p.caveat}")
    lines.append(f"- **Characterised against**: {', '.join(f'{k} {v}' for k, v in p.characterised_versions.items())}")
    lines.append("- A preset freezes the method, not a fitted model: a new screen needs fresh calibration.")
    return "\n".join(lines) + "\n"


def describe_all() -> str:
    return "\n".join(describe(n) for n in sorted(PRESETS))


if __name__ == "__main__":
    print(describe_all())
