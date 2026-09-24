# design/

Static interface mockups: HTML drawings, not running code. Copied into the repo on
2026-09-24 from a Claude Code session's temporary folder, so they are not lost.

**The real interface is in code:** `gazefollower/gazelink_core/ui/` and
`gazefollower/gazelink_core/interaction/` (the `--desk` bar, the menu), launched via
`gf_live.py`. When a mockup and the code disagree, **the code is what actually works.**

| File | What it shows |
|---|---|
| `mockups/Main.dc.html` | Work screen with the control bar |
| `mockups/Panel.dc.html` | More actions (the panel) |
| `mockups/Keyboard.dc.html` | On-screen keyboard |
| `mockups/Scroll.dc.html` | Scrolling |
| `mockups/Zoom.dc.html` | Zoom to aim precisely |
| `mockups/Drag.dc.html` | Drag |
| `mockups/States.dc.html` | States and feedback |
| `mockups/Density.dc.html` | The spacing behind the design |
| `mockups/canvas.json` | Page layout on the design canvas |
| `gazelink-runbook.html` | GAZELINK control board (run guide) from an earlier session. **The up-to-date commands are in `COMMANDS.MD`** |

The mockups reference `./support.js`, which comes from the design environment and is not
in the repo. They open in a browser without it (inline styles), but repeated elements the
script generates may be missing.
