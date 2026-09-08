# Live display test — M3-00

One session, about ten minutes, at a desk with **two monitors**. This is the
first time any of this code meets a real screen; every check below has only
ever been verified against fakes.

Nothing here moves the mouse or emits OS input. If anything behaves worse than
described, stop and record what you saw — a wrong answer here matters more
than finishing the list.

## Start

```
python -m gazelink --gaze-check
```

You need a calibration for your current display first. If it prints
`No compatible gaze model`, run `python -m gazelink --guided-calibration`
and then start again.

You should see the gaze dot following your eyes. That is the baseline.

---

## 1. Resolution change

Keep the window open. Change the display resolution in Windows Settings.

- [ ] The dot **stops immediately** — no drifting, no last position left behind
- [ ] A message names both the old and new size
- [ ] The window does not crash or go blank

## 2. Scaling change

Set Windows scaling from 100% to 125% (or back).

- [ ] Freezes, same as above
- [ ] The message mentions scaling, not just resolution

**If it freezes when you did *not* change anything**, that is a false trip —
note it. It means DPI rounding is still leaking through.

## 3. Drag to the other monitor

Drag the window onto your second screen.

- [ ] Freezes on arrival

## 4. Unplug

Unplug the second monitor while the window is on it.

- [ ] A clean message, no crash, no stack trace

## 5. The latch — the one that matters most

Put the resolution **back** to what it was in step 1.

- [ ] It stays frozen. It must **not** start tracking again.

Silently resuming is the failure this whole feature exists to prevent: the
session already spanned a change, and you would have no way to know the gaze
had quietly become wrong again.

---

## 6. Recovery by eye gesture

While frozen, you should see two options with one highlighted:

```
[ Calibrate for this screen ]    Close GAZELINK
```

- [ ] **Short close** (about half a second) — the highlight moves to the other option
- [ ] **Long close** (about a second and a half) — the highlighted option is taken
- [ ] **Normal blinking does nothing at all** — blink freely for ten seconds and confirm the highlight never moves
- [ ] Turning your head away mid-hold cancels instead of choosing
- [ ] Choosing *Calibrate for this screen* closes the window and calibration
      opens **on the monitor you were actually looking at** — do this test
      after dragging the window to the second screen, because opening on the
      primary one is the failure it is checking for

### Tuning the durations

The two thresholds are **guesses, not measurements** — nobody has held their
eyes shut in front of this code. They live in one place:

`src/gazelink/config.py` → `GestureTimingConfig`

| Setting | Default | Raise it if... | Lower it if... |
|---|---|---|---|
| `natural_blink_max_ms` | 400 | your normal blinks trigger the menu | a deliberate close is ignored |
| `recovery_confirm_ms` | 1500 | you choose things by accident | holding that long is tiring |

`recovery_confirm_ms` must stay above `intentional_hold_min_ms` (700); the
config refuses otherwise.

**Tell me which numbers felt right.** That is the measurement this feature has
been missing, and it is the only part of this I cannot do without you.

---

## What to report back

For each of the six sections: did it do what it says, or something else?

Also worth noting:
- Anything that froze when nothing had changed (false trip)
- Anything that kept running when it should have frozen (missed trip)
- Whether the eye gestures felt usable or fought you
