"""What resolution does the camera ACTUALLY deliver on the library's path?

Read-only probe. No frame is saved, logged or displayed: only shapes, the
driver's reported properties and the measured frame rate are printed.

Why it exists: ``gazefollower.camera.WebCamCamera`` calls ``cap.set(width,
height)`` on a VideoCapture that is not open yet (OpenCV 4.14 returns False
and ignores it), then opens the device and resizes every frame to 640x480.
``gf_setup``'s probe opens first and sets afterwards, so the 640x480 in
setup.json describes the probe's path, not the library's.

Stages, each printing frame.shape as read (before any resize):
1. exactly the library's order: VideoCapture() -> set 640x480 -> open(index)
2. per mode and backend, a FRESH capture: open -> request the mode -> read.
   Switching modes on an already-streaming MSMF capture crashed on this rig
   (first run: "Failed to select stream 0", then a cv::Mat assertion), so
   each mode gets its own capture and any failure is reported, not raised.
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from gazelink_core.tracking import face_landmarks as FL

MODES = ((640, 480), (1280, 720), (1920, 1080))
BACKENDS = (("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW))


def read_stats(cap: cv2.VideoCapture, frames: int) -> dict:
    shapes: set[tuple[int, ...]] = set()
    stamps: list[float] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        shapes.add(tuple(frame.shape))
        stamps.append(time.monotonic())
        del frame  # never kept
    gaps = [b - a for a, b in zip(stamps[1:], stamps[2:]) if b > a]
    return {
        "frames_read": len(stamps),
        "shapes_as_read": sorted(shapes),
        "measured_fps": round(1.0 / statistics.median(gaps), 1) if gaps else None,
        "reported": (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        ),
        "backend": cap.getBackendName(),
    }


def field_of_view(args: argparse.Namespace) -> int:
    """Outer inter-ocular distance and eye position per mode, LO-HI-HI-LO.

    MediaPipe runs IN MEMORY through the library's own face alignment; only
    medians of a distance and two normalised positions are printed. A fresh
    FaceMesh per block, because it tracks across frames. Keep the head still.

    Reading: IOD(720)/IOD(480) ~2.0 means the 16:9 mode keeps the horizontal
    field of view (it crops top/bottom); ~1.5 means 4:3 was a side crop.
    """

    from gazefollower.face_alignment import MediaPipeFaceAlignment  # noqa: PLC0415

    blocks = ((640, 480), (1280, 720), (1920, 1080), (640, 480))
    rows = []
    for w, h in blocks:
        aligner = MediaPipeFaceAlignment()
        cap = cv2.VideoCapture(args.index, cv2.CAP_MSMF)
        iod, mid_x, mid_y, shapes = [], [], [], set()
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            for i in range(args.frames + 10):
                ok, frame = cap.read()
                if not ok:
                    break
                if i < 10 or frame.shape[:2] != (h, w):
                    continue  # warm-up or a frame still in the old mode
                shapes.add(frame.shape)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                info = aligner.detect(time.time_ns(), rgb)
                del frame, rgb  # never kept
                marks = FL.raw_landmarks(info)
                if marks is None:
                    continue
                left, right = marks[33, :2].astype(float), marks[263, :2].astype(float)
                iod.append(float(np.hypot(*(right - left))))
                mid_x.append(float((left[0] + right[0]) / 2 / w))
                mid_y.append(float((left[1] + right[1]) / 2 / h))
        finally:
            cap.release()
        if not iod:
            print(f"  {w}x{h}: no face found in {sorted(shapes)}")
            continue
        med = statistics.median(iod)
        rows.append((w, h, med))
        print(
            f"  {w}x{h}: {len(iod)} frames, outer IOD median {med:.1f}px, "
            f"eye-mid x {statistics.median(mid_x):.3f} y {statistics.median(mid_y):.3f} "
            f"(eye box ~{0.47 * med:.0f}x{0.35 * med:.0f}px, ESTIMATE)"
        )
    base = [r[2] for r in rows if r[:2] == (640, 480)]
    if base:
        lo = statistics.mean(base)
        for w, h, med in rows:
            if (w, h) != (640, 480):
                print(f"  IOD ratio {w}x{h} / 640x480 = {med / lo:.2f}  (width ratio {w / 640:.2f})")
    return 0


def pipeline_rate(args: argparse.Namespace) -> int:
    """Delivered frame rate with ONE full gaze pipeline, native-lo vs hi crop.

    Runs face alignment + MGazeNet on every frame in the capture loop, the
    way the library does, for ``--seconds`` per arm: the library's own
    640x480 path (set before open, resize to 640x480) and the hi path
    (1920x1080 after open, centre crop 1440x1080). No model, no saving; only
    timing numbers and a face-found fraction are printed. Order LO-HI-LO so a
    drift over the run shows up.

    Pass marks fixed before the run: delivered fps >= 28 and alignment +
    estimator p95 <= 25 ms.
    """

    import gf_capture as CAP  # noqa: PLC0415
    from gazelink_core.tracking import gazefollower_source as SRC  # noqa: PLC0415
    from gazefollower.face_alignment import MediaPipeFaceAlignment  # noqa: PLC0415
    from gazefollower.gaze_estimator import MGazeNetGazeEstimator  # noqa: PLC0415

    estimator = MGazeNetGazeEstimator()
    arms = (("lo-library-640x480", None), ("hi-1440x1080-crop", CAP.SOURCE_MODE), ("lo-library-640x480", None))
    for name, mode in arms:
        aligner = MediaPipeFaceAlignment()
        if mode is None:
            cap = cv2.VideoCapture()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.open(args.index)
        else:
            cap = cv2.VideoCapture(args.index, cv2.CAP_MSMF)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode[1])
        stamps, work, faces, wrong = [], [], 0, 0
        try:
            if not cap.isOpened():
                print(f"  {name}: camera did not open")
                continue
            for _ in range(10):
                cap.grab()
            until = time.monotonic() + args.seconds
            while time.monotonic() < until:
                ok, frame = cap.read()
                if not ok:
                    continue
                stamps.append(time.monotonic())
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if mode is None:
                    image = cv2.resize(rgb, (640, 480))
                elif frame.shape[:2] != (mode[1], mode[0]):
                    wrong += 1
                    continue
                else:
                    image = np.ascontiguousarray(CAP.centre_crop_4x3(rgb))
                started = time.perf_counter()
                face = aligner.detect(time.time_ns(), image)
                gaze = estimator.detect(image, face)
                work.append((time.perf_counter() - started) * 1000.0)
                faces += SRC.gaze_status_of(gaze)
                del frame, rgb, image  # never kept
        finally:
            cap.release()
        gaps = np.diff(stamps) * 1000.0 if len(stamps) > 2 else np.array([])
        fps = 1000.0 / float(np.median(gaps)) if gaps.size else None
        p95 = float(np.percentile(work, 95)) if work else None
        verdict = (
            "PASS" if fps is not None and p95 is not None and fps >= 28 and p95 <= 25 else "FAIL"
        )
        print(
            f"  {name}: frames {len(stamps)}, delivered fps "
            f"{'--' if fps is None else f'{fps:.1f}'}, frame interval p95 "
            f"{'--' if not gaps.size else f'{np.percentile(gaps, 95):.0f}'} ms, "
            f"pipeline p50 {'--' if not work else f'{np.median(work):.1f}'} / p95 "
            f"{'--' if p95 is None else f'{p95:.1f}'} ms, gaze found "
            f"{faces}/{len(work)}, wrong shape {wrong}  -> {verdict}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument(
        "--fov",
        action="store_true",
        help="instead of stage 2: measure, per mode, the eye size in pixels and where the eyes "
        "sit in the frame (head still), to learn whether 16:9 keeps the 4:3 field of view",
    )
    parser.add_argument(
        "--pipeline-rate",
        action="store_true",
        help="measure delivered fps and processing time with one full gaze pipeline, "
        "native 640x480 vs the 1440x1080 crop (LO-HI-LO)",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.pipeline_rate:
        return pipeline_rate(args)

    print("stage 1: library order (set before open)")
    cap = cv2.VideoCapture()
    print("  set before open returned:", cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
          cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480))
    try:
        if cap.open(args.index):
            print("  ", read_stats(cap, args.frames))
        else:
            print("  camera did not open")
    finally:
        cap.release()

    if args.fov:
        return field_of_view(args)

    print("stage 2: fresh capture per mode, mode requested right after open")
    for name, api in BACKENDS:
        for w, h in MODES:
            cap = cv2.VideoCapture(args.index, api)
            try:
                if not cap.isOpened():
                    print(f"  {name} {w}x{h}: did not open")
                    continue
                accepted = (cap.set(cv2.CAP_PROP_FRAME_WIDTH, w), cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h))
                print(f"  {name} request {w}x{h} set returned {accepted}:", read_stats(cap, args.frames))
            except cv2.error as exc:
                print(f"  {name} {w}x{h}: failed ({str(exc).splitlines()[-1][:120]})")
            finally:
                cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
