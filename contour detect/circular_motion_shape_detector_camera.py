"""
Moving-shape boundary detector for a fixed camera.
The output is a black canvas with ONLY the selected moving object's green contour.
The source camera frame/background is never displayed.

Install:
    python -m pip install opencv-python numpy

Run:
    python circular_motion_shape_detector_camera.py

Keys:
    q / ESC : quit
    d       : toggle debug window (shows segmentation mask)
    r       : reset tracker
"""

import cv2
import numpy as np
import math
import time
from collections import deque

# ---------------- Configuration ----------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
REQUESTED_FPS = 60

# Foreground segmentation tuned for a colored object on a mostly static background.
# Lower DIFF_THRESHOLD if object pixels are being missed; raise it if background noise appears.
DIFF_THRESHOLD = 18
MIN_OBJECT_AREA = 120
MAX_OBJECT_AREA = 0           # 0 = unlimited
MORPH_KERNEL = 3
CLOSE_ITERATIONS = 2
OPEN_ITERATIONS = 1

# Temporal tracking
CENTER_HISTORY = 80
MIN_POINTS_FOR_CIRCULARITY = 18
MIN_CIRCLE_RADIUS = 15
MAX_RADIAL_ERROR_RATIO = 0.20
MIN_ANGLE_PROGRESS_DEG = 35

SHOW_STATUS = True
SHOW_DEBUG = False
# ------------------------------------------------


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
    if r2 <= 0:
        return None
    radius = math.sqrt(r2)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(cx), float(cy), float(radius), float(np.mean(np.abs(distances - radius)))


def angular_progress(points, cx, cy):
    pts = np.asarray(points, dtype=np.float64)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    delta = np.diff(angles)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(delta)) * 180.0 / np.pi)


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
            f"Cannot open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1 or close other camera apps."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    kernel = make_kernel(MORPH_KERNEL)
    centers = deque(maxlen=CENTER_HISTORY)
    previous_center = None
    circularity_confirmed = False
    last_time = time.perf_counter()
    fps = 0.0

    # Start with a short warm-up so MOG2 learns the stationary background.
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=250, varThreshold=20, detectShadows=False
    )
    warmup_frames = 35
    print("Keep the camera still and background static during the first second.")
    print("Output is black with only the moving object's contour.")
    print("Keys: q/ESC quit | d debug mask | r reset tracker")

    try:
        frame_counter = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            frame_counter += 1
            # Mild smoothing only; avoid erasing thin object boundaries.
            smooth = cv2.GaussianBlur(frame, (3, 3), 0)

            # MOG2 is adaptive: unlike a fixed background image, it tolerates
            # small illumination changes and does not require the object to be white.
            learning_rate = 0.03 if frame_counter <= warmup_frames else 0.001
            mask = subtractor.apply(smooth, learningRate=learning_rate)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    iterations=OPEN_ITERATIONS)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=CLOSE_ITERATIONS)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            candidates = []
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
                if previous_center is None:
                    # First acquisition: choose the largest moving component.
                    selected = max(candidates, key=lambda item: item[1])
                else:
                    # Prefer continuity; prevent switching to a distant background blob.
                    h, w = frame.shape[:2]
                    max_jump = max(w, h) * 0.22
                    nearby = [item for item in candidates
                              if math.dist(item[2], previous_center) <= max_jump]
                    if nearby:
                        selected = min(
                            nearby,
                            key=lambda item: math.dist(item[2], previous_center) - 0.0005 * item[1]
                        )
                    else:
                        # Reacquire if target moves quickly or was briefly lost.
                        selected = max(candidates, key=lambda item: item[1])
                        centers.clear()
                        circularity_confirmed = False

            output = np.zeros_like(frame)
            status = "Searching for moving object"

            if selected is not None:
                contour, area, center = selected
                previous_center = center
                centers.append(center)

                if len(centers) >= MIN_POINTS_FOR_CIRCULARITY:
                    model = fit_circle(list(centers))
                    if model is not None:
                        cx, cy, radius, radial_error = model
                        progress = angular_progress(list(centers), cx, cy)
                        if (radius >= MIN_CIRCLE_RADIUS
                                and radial_error / max(radius, 1e-6) <= MAX_RADIAL_ERROR_RATIO
                                and progress >= MIN_ANGLE_PROGRESS_DEG):
                            circularity_confirmed = True

                # Draw the candidate contour immediately; don't leave output blank
                # while waiting for the circular-motion validator.
                cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
                status = "Circular motion confirmed" if circularity_confirmed else "Tracking moving shape"

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            last_time = now

            if SHOW_STATUS:
                cv2.putText(output, f"{status} | FPS: {fps:.1f}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (0, 180, 255), 1, cv2.LINE_AA)

            cv2.namedWindow("Moving Shape Only", cv2.WINDOW_NORMAL)
            cv2.imshow("Moving Shape Only", output)

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
                centers.clear()
                previous_center = None
                circularity_confirmed = False

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
