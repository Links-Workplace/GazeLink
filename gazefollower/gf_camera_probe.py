"""Is the camera actually delivering a usable picture right now?

A recording that captures no face looks the same from the outside whether the
lens is covered, the room is dark, another application holds the device, or
the driver is handing back blank buffers. This grabs a few frames and reports
what is in them, so that question is answered in seconds instead of after a
six-minute protocol.

Deliberately tiny: OpenCV only, no GazeFollower, no MediaPipe, no window. It
reports brightness and frame-to-frame change; it does not detect faces, and a
bright, varying picture still proves only that the camera works.

    python gf_camera_probe.py            # default device 0
    python gf_camera_probe.py --index 1  # a second camera
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

# Below this mean pixel value an 8-bit frame is effectively a black image.
DARK_MEAN = 12.0
# Below this standard deviation across a frame there is no scene in it.
FLAT_STD = 3.0
# Below this mean absolute difference between consecutive frames the stream is
# repeating one buffer rather than delivering live video.
FROZEN_DIFF = 0.5


def probe(index: int = 0, frames: int = 30) -> int:
    capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not capture.isOpened():
        print(f"camera {index}: WOULD NOT OPEN -- another application is probably holding it")
        return 2
    try:
        grabbed = []
        for _ in range(frames):
            ok, frame = capture.read()
            if ok and frame is not None:
                grabbed.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64))
    finally:
        capture.release()

    if not grabbed:
        print(f"camera {index}: opened but returned NO frames")
        return 2

    means = np.array([f.mean() for f in grabbed])
    stds = np.array([f.std() for f in grabbed])
    diffs = np.array([np.abs(b - a).mean() for a, b in zip(grabbed, grabbed[1:], strict=False)])
    height, width = grabbed[0].shape
    print(f"camera {index}: {len(grabbed)} frames at {width}x{height}")
    print(f"  brightness mean {means.mean():6.1f}  (0 = black, 255 = white)")
    print(f"  detail    std   {stds.mean():6.1f}  (0 = flat, featureless image)")
    motion = diffs.mean() if diffs.size else float("nan")
    print(f"  motion    diff  {motion:6.2f}  (0 = identical frames)")

    problems = []
    if means.mean() < DARK_MEAN:
        problems.append("the picture is essentially black -- lens covered, or the room is dark")
    if stds.mean() < FLAT_STD:
        problems.append("the picture has no detail -- a blank buffer rather than a scene")
    if diffs.size and diffs.mean() < FROZEN_DIFF:
        problems.append("consecutive frames are identical -- the stream is frozen")
    if problems:
        print("\nPROBLEM:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("\nThe camera is delivering a live, lit picture.")
    print("If recordings still find no face, the issue is framing or the detector,")
    print("not the camera: check that your face is inside the frame and evenly lit.")
    return 0


def probe_face(index: int = 0, frames: int = 30, white_screen: bool = False) -> int:
    """Does the detector find a face right now, under recording conditions?

    The protocols fill a 49-inch display with white a few tens of centimetres
    from the operator. That can drive the camera's automatic exposure down
    until the face is a dark silhouette, so the probe can raise the same white
    field: a face found on a grey desktop but lost against white is an
    exposure problem, not a framing one.
    """

    screen = None
    if white_screen:
        import pygame  # noqa: PLC0415  (only needed for this mode)

        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        screen.fill((255, 255, 255))
        pygame.display.flip()
        for _ in range(30):
            pygame.event.pump()
            pygame.time.wait(10)

    try:
        import mediapipe as mp  # noqa: PLC0415

        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
        capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not capture.isOpened():
            print(f"camera {index}: WOULD NOT OPEN")
            return 2
        found = 0
        seen = 0
        face_means: list[float] = []
        frame_means: list[float] = []
        try:
            for _ in range(frames):
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                seen += 1
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_means.append(float(grey.mean()))
                result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.multi_face_landmarks:
                    found += 1
                    height, width = grey.shape
                    xs = [lm.x * width for lm in result.multi_face_landmarks[0].landmark]
                    ys = [lm.y * height for lm in result.multi_face_landmarks[0].landmark]
                    x0, x1 = max(0, int(min(xs))), min(width, int(max(xs)))
                    y0, y1 = max(0, int(min(ys))), min(height, int(max(ys)))
                    if x1 > x0 and y1 > y0:
                        face_means.append(float(grey[y0:y1, x0:x1].mean()))
        finally:
            capture.release()
            mesh.close()
    finally:
        if screen is not None:
            import pygame  # noqa: PLC0415

            pygame.quit()

    if not seen:
        print("no frames read")
        return 2
    label = "WHITE FULLSCREEN" if white_screen else "normal desktop"
    print(f"camera {index}, {label}: face found in {found}/{seen} frames")
    print(f"  whole-frame brightness {np.mean(frame_means):6.1f}")
    if face_means:
        print(f"  face-region brightness {np.mean(face_means):6.1f}")
    if found == 0:
        print("\nThe detector finds NO face in these conditions.")
        return 1
    if found < seen * 0.8:
        print("\nThe face is found only intermittently.")
        return 1
    print("\nFace detection is healthy here.")
    return 0


def probe_pose(index: int = 0, seconds: float = 60.0, target_pitch: float = 2.07) -> int:
    """Live head pitch against the pitch the calibration was recorded at.

    Every recording whose features carried a strong gaze signal was made with
    the head within a couple of degrees of the calibration pose; every
    recording that lost the signal sat about eleven degrees below it. Eleven
    degrees is easy not to notice and hard to restore by feel, so this prints
    the number live and says which way to move.
    """

    print(
        "DISABLED: this mode is not trustworthy and must not be used to aim the camera.\n"
        "\n"
        "It reads landmarks straight from MediaPipe, while a recording reads them\n"
        "from GazeFollower's FaceInfo. The two do not agree: with the camera\n"
        "untouched, this mode reported -11 deg while round17 recorded +2.08 deg for\n"
        "the same pose. Acting on that difference moved a correctly-aimed camera.\n"
        "\n"
        "Read the pitch a recording actually stored (pnp_deg) instead, and calibrate\n"
        "in whatever pose is comfortable rather than aiming for a target angle."
    )
    return 2


def _probe_pose_unverified(index: int, seconds: float, target_pitch: float) -> int:
    import time  # noqa: PLC0415

    import mediapipe as mp  # noqa: PLC0415

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import gf_head_features as HF  # noqa: PLC0415

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )
    capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not capture.isOpened():
        print(f"camera {index}: WOULD NOT OPEN")
        return 2
    print(f"calibration pose was pitch {target_pitch:+.2f} deg. Ctrl-C to stop.\n")
    deadline = time.monotonic() + seconds
    last = 0.0
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                continue
            height, width = frame.shape[:2]
            xy = np.array(
                [[lm.x * width, lm.y * height] for lm in result.multi_face_landmarks[0].landmark]
            )
            angles = HF.pnp_degrees_from_landmarks(xy, width, height)
            if angles is None:
                continue
            pitch = float(angles[1])
            now = time.monotonic()
            if now - last < 0.25:
                continue
            last = now
            delta = pitch - target_pitch
            if abs(delta) <= 2.0:
                advice = "MATCHED -- hold this and start recording"
            elif delta < 0:
                advice = "BELOW calibration: raise your seat, or tilt the camera down"
            else:
                advice = "ABOVE calibration: lower your seat, or tilt the camera up"
            print(f"  pitch {pitch:+7.2f} deg   offset {delta:+7.2f}   {advice}   ", end="\r")
    except KeyboardInterrupt:
        print()
    finally:
        capture.release()
        mesh.close()
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="camera device index")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--face", action="store_true", help="also run the face detector")
    parser.add_argument(
        "--pose",
        action="store_true",
        help="live head pitch against the calibration pose, to restore the geometry",
    )
    parser.add_argument(
        "--target-pitch",
        type=float,
        default=2.07,
        help="pitch the calibration was recorded at (round5/A: 2.07)",
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument(
        "--white",
        action="store_true",
        help="raise a white fullscreen first, reproducing the protocol's lighting",
    )
    args = parser.parse_args(argv)
    if args.pose:
        return probe_pose(args.index, args.seconds, args.target_pitch)
    if args.face or args.white:
        return probe_face(args.index, args.frames, white_screen=args.white)
    return probe(args.index, args.frames)


if __name__ == "__main__":
    sys.exit(main())
