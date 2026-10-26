"""
Circular-motion shape detector — improved foreground segmentation
-----------------------------------------------------------------
Detects the silhouette of one moving object of ANY color/shape against a
static background. It does not need a visible circular track and never draws
the original frame or the track in the output window.

Install:
    python -m pip install opencv-python numpy

Run:
    python circular_motion_shape_detector_optimized.py

Controls:
    b  capture/rebuild background reference (keep camera/background static;
       if the object keeps moving, the temporal median can suppress it)
    r  clear tracking history
    d  toggle debug view
    q / ESC quit
"""

import cv2
import numpy as np
import math
import time
from collections import deque

# -------------------------- CONFIG --------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
REQUESTED_FPS = 60

BACKGROUND_SAMPLES = 45       # temporal median samples
DIFF_THRESHOLD = 22           # increase if noise is detected; lower if shape is missed
MIN_OBJECT_AREA = 250
MAX_OBJECT_AREA = 0            # 0 = no maximum
MORPH_KERNEL = 5
CLOSE_ITERATIONS = 2
OPEN_ITERATIONS = 1

# Track centroid motion to validate a circular trajectory.
TRACK_HISTORY = 90
MIN_POINTS_FOR_CIRCLE = 20
MIN_CIRCLE_RADIUS = 15
MAX_RADIAL_ERROR_RATIO = 0.16
MIN_ANGLE_PROGRESS_DEG = 50

DRAW_THICKNESS = 2
SHOW_STATUS = True
SHOW_DEBUG = False
# ------------------------------------------------------------


def fit_circle(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return None

    x, y = pts[:, 0], pts[:, 1]
    matrix = np.column_stack((x, y, np.ones_like(x)))
    rhs = -(x * x + y * y)

    try:
        a, b, c = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    cx, cy = -a / 2.0, -b / 2.0
    r2 = cx * cx + cy * cy - c
    if r2 <= 0 or not np.isfinite(r2):
        return None

    radius = math.sqrt(r2)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mean_error = float(np.mean(np.abs(distances - radius)))
    return float(cx), float(cy), float(radius), mean_error


def angular_progress(points, cx, cy):
    pts = np.asarray(points, dtype=np.float64)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    delta = np.diff(angles)
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sum(np.abs(delta)) * 180.0 / np.pi)


def build_background(cap, sample_count):
    """Temporal median background; best when the scene is static and target moves."""
    samples = []
    print(f"Capturing background reference ({sample_count} frames)...")
    for i in range(sample_count):
        ok, frame = cap.read()
        if not ok:
            continue
        # Downsample for lower memory use, then keep full-size reference at capture size.
        samples.append(frame)
        cv2.waitKey(1)

    if len(samples) < max(8, sample_count // 3):
        raise RuntimeError("Not enough frames to build background reference.")
    background = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
    print("Background reference ready.")
    return background


def make_kernel(size):
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def main():
    global SHOW_DEBUG

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {CAMERA_INDEX}. Try index 1 or close other camera apps."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    kernel = make_kernel(MORPH_KERNEL)
    background = build_background(cap, BACKGROUND_SAMPLES)
    centers = deque(maxlen=TRACK_HISTORY)
    last_center = None
    circle_confirmed = False
    last_time = time.perf_counter()
    fps = 0.0

    print("Output shows only the selected object's contour on black.")
    print("Keys: b=rebuild background, r=reset tracking, d=debug, q/ESC=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            # Compare in LAB, which handles luminance and chromatic differences
            # better than a single fixed bright/dark threshold.
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
            diff = cv2.absdiff(frame_lab, bg_lab)
            # Max channel difference gives a color-independent foreground score.
            score = np.max(diff, axis=2).astype(np.uint8)
            _, mask = cv2.threshold(score, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=OPEN_ITERATIONS)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=CLOSE_ITERATIONS)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            h, w = frame.shape[:2]

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_OBJECT_AREA:
                    continue
                if MAX_OBJECT_AREA > 0 and area > MAX_OBJECT_AREA:
                    continue
                moments = cv2.moments(contour)
                if abs(moments["m00"]) < 1e-6:
                    continue
                center = (moments["m10"] / moments["m00"],
                          moments["m01"] / moments["m00"])
                candidates.append((contour, area, center))

            selected = None
            if candidates:
                if last_center is None:
                    selected = max(candidates, key=lambda item: item[1])
                else:
                    # Prefer the closest candidate, with area as a tie-breaker.
                    max_jump = max(w, h) * 0.30
                    near = [item for item in candidates
                            if math.dist(item[2], last_center) <= max_jump]
                    if near:
                        selected = min(
                            near,
                            key=lambda item: (
                                math.dist(item[2], last_center),
                                -item[1]
                            )
                        )
                    else:
                        # Allow reacquisition after a temporary miss.
                        selected = max(candidates, key=lambda item: item[1])
                        centers.clear()
                        circle_confirmed = False

            # Output canvas is always black; only target contour is drawn.
            output = np.zeros_like(frame)
            status = "Searching for target"
            if selected is not None:
                contour, area, center = selected
                last_center = center
                centers.append(center)

                if len(centers) >= MIN_POINTS_FOR_CIRCLE:
                    model = fit_circle(list(centers))
                    if model is not None:
                        cx, cy, radius, error = model
                        progress = angular_progress(list(centers), cx, cy)
                        if (radius >= MIN_CIRCLE_RADIUS
                                and error / max(radius, 1e-6) <= MAX_RADIAL_ERROR_RATIO
                                and progress >= MIN_ANGLE_PROGRESS_DEG):
                            circle_confirmed = True

                # Draw object boundary while tracking. Status reports whether
                # the centroid trajectory has passed the circular-motion check.
                cv2.drawContours(output, [contour], -1, (0, 255, 0), DRAW_THICKNESS)
                status = "Circular motion confirmed" if circle_confirmed else "Tracking shape"

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            last_time = now

            if SHOW_STATUS:
                cv2.putText(output, f"{status} | FPS: {fps:.1f}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (0, 180, 255), 1, cv2.LINE_AA)

            cv2.namedWindow("Moving Shape Only", cv2.WINDOW_NORMAL)
            cv2.imshow("Moving Shape Only", output)

            if SHOW_DEBUG:
                debug = np.hstack((
                    cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                    cv2.resize(background, (w // 2, h // 2)),
                    cv2.resize(frame, (w // 2, h // 2))
                ))
                cv2.imshow("Debug: Foreground Mask | Background | Input", debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                SHOW_DEBUG = not SHOW_DEBUG
                if not SHOW_DEBUG:
                    cv2.destroyWindow("Debug: Foreground Mask | Background | Input")
            elif key == ord("r"):
                centers.clear()
                last_center = None
                circle_confirmed = False
            elif key == ord("b"):
                background = build_background(cap, BACKGROUND_SAMPLES)
                centers.clear()
                last_center = None
                circle_confirmed = False

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
