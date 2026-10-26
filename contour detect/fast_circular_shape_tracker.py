"""
Fast circular-motion object boundary tracker (single file)
=========================================================
Designed for one moving object of any color on a mostly static background.
Output: black canvas + only the selected object's green outline.

Install:
    python -m pip install opencv-python numpy

Run:
    python fast_circular_shape_tracker.py

Keys:
    q / ESC : quit
    d       : toggle foreground-mask debug window
    r       : reset tracking and background model
    + / =   : increase minimum contour area
    -       : decrease minimum contour area
"""

import cv2
import numpy as np
import math
import time
from collections import deque

# -------------------------- TUNING --------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
REQUESTED_FPS = 60

# Background subtraction. Lower VAR_THRESHOLD detects subtler object/background
# differences, but can increase noise. Keep camera and scene stationary.
MOG2_HISTORY = 180
MOG2_VAR_THRESHOLD = 18
WARMUP_FRAMES = 30
STEADY_LEARNING_RATE = 0.0005

# Mask cleanup
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 0             # 0 = unlimited
MORPH_KERNEL = 3
OPEN_ITERATIONS = 1
CLOSE_ITERATIONS = 2

# Tracker
MAX_MISSED_FRAMES = 8
CENTER_SMOOTHING = 0.65          # higher = follow fast movement more closely
MAX_CENTER_JUMP_FRACTION = 0.30  # of max(frame width, frame height)
TRACK_HISTORY = 100

# Circularity status is informational only; it never blocks boundary display.
CHECK_CIRCULAR_MOTION = True
MIN_POINTS_FOR_CIRCLE = 20
MIN_CIRCLE_RADIUS = 12
MAX_RADIAL_ERROR_RATIO = 0.22
MIN_ANGLE_PROGRESS_DEG = 35

SHOW_STATUS = True
SHOW_DEBUG = False
# ------------------------------------------------------------


def make_kernel(size):
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fit_circle(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x * x + y * y)
    try:
        a, bb, c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = -a / 2.0, -bb / 2.0
    r2 = cx * cx + cy * cy - c
    if r2 <= 0 or not np.isfinite(r2):
        return None
    r = math.sqrt(r2)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(cx), float(cy), float(r), float(np.mean(np.abs(d - r)))


def angular_progress(points, cx, cy):
    pts = np.asarray(points, dtype=np.float64)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    delta = np.diff(angles)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(delta)) * 180.0 / np.pi)


def contour_center(contour):
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w / 2.0, y + h / 2.0)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def main():
    global SHOW_DEBUG, MIN_CONTOUR_AREA

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {CAMERA_INDEX}. Try CAMERA_INDEX=1 or close other camera apps."
        )

    # Request low-latency capture. The camera/driver may ignore some settings.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    kernel = make_kernel(MORPH_KERNEL)
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False,
    )

    centers = deque(maxlen=TRACK_HISTORY)
    last_center = None
    smoothed_center = None
    missed_frames = 0
    circular_confirmed = False
    frame_count = 0
    last_time = time.perf_counter()
    fps = 0.0

    print("Keep the camera/background static. Allow about 1 second for background warm-up.")
    print("Output shows only the selected moving object's contour on black.")
    print("Keys: q/ESC quit | d debug mask | r reset | +/- adjust minimum area")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            frame_count += 1
            # Minimal blur to preserve corners and narrow features.
            smooth = cv2.GaussianBlur(frame, (3, 3), 0)

            # MOG2 handles arbitrary colors; no white/black threshold is assumed.
            if frame_count <= WARMUP_FRAMES:
                learning_rate = 0.05
            else:
                learning_rate = STEADY_LEARNING_RATE

            mask = subtractor.apply(smooth, learningRate=learning_rate)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=OPEN_ITERATIONS)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=CLOSE_ITERATIONS)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            h, w = frame.shape[:2]
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA:
                    continue
                if MAX_CONTOUR_AREA > 0 and area > MAX_CONTOUR_AREA:
                    continue
                center = contour_center(contour)
                candidates.append((contour, area, center))

            selected = None
            if candidates:
                if smoothed_center is None:
                    selected = max(candidates, key=lambda x: x[1])
                else:
                    max_jump = max(w, h) * MAX_CENTER_JUMP_FRACTION
                    plausible = [
                        item for item in candidates
                        if math.dist(item[2], smoothed_center) <= max_jump
                    ]
                    if plausible:
                        # Distance dominates; area breaks near ties.
                        selected = min(
                            plausible,
                            key=lambda item: math.dist(item[2], smoothed_center)
                            - 0.0002 * item[1]
                        )
                    else:
                        # If target moved too far between frames, reacquire largest blob.
                        selected = max(candidates, key=lambda x: x[1])
                        centers.clear()
                        circular_confirmed = False
                        smoothed_center = None

            output = np.zeros_like(frame)
            status = "Searching"

            if selected is not None:
                contour, area, center = selected
                missed_frames = 0
                if smoothed_center is None:
                    smoothed_center = center
                else:
                    # Follow fast movement without completely trusting a noisy centroid.
                    alpha = CENTER_SMOOTHING
                    smoothed_center = (
                        alpha * center[0] + (1.0 - alpha) * smoothed_center[0],
                        alpha * center[1] + (1.0 - alpha) * smoothed_center[1],
                    )

                last_center = center
                centers.append(center)

                if CHECK_CIRCULAR_MOTION and len(centers) >= MIN_POINTS_FOR_CIRCLE:
                    model = fit_circle(list(centers))
                    if model is not None:
                        cx, cy, radius, radial_error = model
                        progress = angular_progress(list(centers), cx, cy)
                        if (radius >= MIN_CIRCLE_RADIUS
                                and radial_error / max(radius, 1e-6) <= MAX_RADIAL_ERROR_RATIO
                                and progress >= MIN_ANGLE_PROGRESS_DEG):
                            circular_confirmed = True

                # Always show the current candidate; circularity validation does not
                # delay or suppress the boundary.
                cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
                status = "Circular motion" if circular_confirmed else "Tracking object"

            else:
                missed_frames += 1
                status = "Target temporarily lost"
                if missed_frames > MAX_MISSED_FRAMES:
                    smoothed_center = None
                    last_center = None
                    centers.clear()
                    circular_confirmed = False

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_time, 1e-6)
            fps = inst_fps if fps == 0 else 0.9 * fps + 0.1 * inst_fps
            last_time = now

            if SHOW_STATUS:
                cv2.putText(
                    output, f"{status} | FPS: {fps:.1f} | min area: {MIN_CONTOUR_AREA}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 180, 255), 1, cv2.LINE_AA
                )

            cv2.namedWindow("Fast Circular Shape Tracker", cv2.WINDOW_NORMAL)
            cv2.imshow("Fast Circular Shape Tracker", output)

            if SHOW_DEBUG:
                cv2.imshow("Foreground Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                SHOW_DEBUG = not SHOW_DEBUG
                if not SHOW_DEBUG:
                    cv2.destroyWindow("Foreground Mask")
            elif key == ord("r"):
                subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=MOG2_HISTORY,
                    varThreshold=MOG2_VAR_THRESHOLD,
                    detectShadows=False,
                )
                centers.clear()
                last_center = None
                smoothed_center = None
                missed_frames = 0
                circular_confirmed = False
                frame_count = 0
            elif key in (ord("+"), ord("=")):
                MIN_CONTOUR_AREA = min(10000, MIN_CONTOUR_AREA + 25)
            elif key == ord("-"):
                MIN_CONTOUR_AREA = max(10, MIN_CONTOUR_AREA - 25)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
