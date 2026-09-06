"""The same identity, but on a real Qt widget instead of pure arithmetic.

``tests/test_screen_mapping.py`` proves the mapping is self-consistent. It
cannot prove that Qt agrees: that a fixed-size label really is that size once
font metrics have had their say, or that ``move()`` round-trips a negative
coordinate rather than clamping it to the parent. Those are the two places
where the arithmetic could be right and the screen still wrong.

Runs in a subprocess with the offscreen platform plugin, for the same reason
the import probes in ``test_smoke.py`` do: ``QApplication`` is a process-wide
singleton whose platform plugin cannot be changed once chosen, and other tests
in this session may already have created one.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

_PROBE = r"""
import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from gazelink.domain import GazePoint, ScreenGeometry
from gazelink.gaze_engine import normalized_to_pixel
from gazelink.screen_mapping import centered_top_left, drawn_center
from gazelink.test_window import TARGET_GLYPH_PX

GEOMETRY = ScreenGeometry("ultrawide", 4096, 1152, 1.25)
COORDS = [0.0, 0.08, 0.12, 0.25, 0.5, 0.75, 0.88, 0.92, 1.0]

app = QApplication([])
surface = QWidget()
surface.resize(GEOMETRY.width_px, GEOMETRY.height_px)

# Built exactly as _TestPointWindow builds its target.
label = QLabel("\u25cf", surface)
font = QFont()
font.setPointSize(44)
font.setBold(True)
label.setFont(font)
label.setFixedSize(TARGET_GLYPH_PX, TARGET_GLYPH_PX)
label.setAlignment(Qt.AlignmentFlag.AlignCenter)

failures = []

# 1. The box is the size we asked for, not the size the font wanted.
if (label.width(), label.height()) != (TARGET_GLYPH_PX, TARGET_GLYPH_PX):
    failures.append(
        "fixed size ignored: got %dx%d, wanted %dx%d"
        % (label.width(), label.height(), TARGET_GLYPH_PX, TARGET_GLYPH_PX)
    )

# 2. The drawn centre is the scored pixel, for every known point.
for nx in COORDS:
    for ny in COORDS:
        point = GazePoint(nx, ny)
        left, top = centered_top_left(
            point, GEOMETRY,
            glyph_width_px=label.width(), glyph_height_px=label.height(),
        )
        label.move(left, top)
        position = label.pos()
        if (position.x(), position.y()) != (left, top):
            failures.append(
                "Qt moved the label somewhere else at (%.2f, %.2f): asked (%d, %d), got (%d, %d)"
                % (nx, ny, left, top, position.x(), position.y())
            )
            continue
        centre = drawn_center(
            (position.x(), position.y()),
            glyph_width_px=label.width(), glyph_height_px=label.height(),
        )
        scored = normalized_to_pixel(point, GEOMETRY)
        if centre != scored:
            failures.append(
                "drawn centre != scored pixel at (%.2f, %.2f): drawn %s, scored %s"
                % (nx, ny, centre, scored)
            )

# 3. A corner target's glyph must be allowed to hang off the edge.
left, top = centered_top_left(
    GazePoint(0.0, 0.0), GEOMETRY,
    glyph_width_px=label.width(), glyph_height_px=label.height(),
)
label.move(left, top)
if (label.pos().x(), label.pos().y()) != (left, top):
    failures.append(
        "a negative position was clamped by Qt: asked (%d, %d), got (%d, %d)"
        % (left, top, label.pos().x(), label.pos().y())
    )

sys.exit("; ".join(failures[:5]) if failures else 0)
"""


def test_a_real_qt_label_lands_on_the_scored_pixel() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )

    combined = completed.stdout + completed.stderr
    if "could not be found" in combined or "platform plugin" in combined.lower():
        pytest.skip(f"no usable Qt platform plugin in this environment: {combined.strip()[:200]}")
    assert completed.returncode == 0, combined
