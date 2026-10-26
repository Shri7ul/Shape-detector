import cv2
import numpy as np
import math
import time
from collections import deque

# Arbitrary color/shape. A visible circular track is NOT required.
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT, REQUESTED_FPS = 1280, 720, 60
HISTORY = 350
VAR_THRESHOLD = 24
MIN_OBJECT_AREA = 180
MAX_OBJECT_AREA = 0          # 0 = unlimited
MORPH_KERNEL = 5
TRACK_HISTORY = 100
MIN_POINTS_FOR_CIRCLE = 18
MIN_CIRCLE_RADIUS = 20
MAX_CIRCLE_RADIUS = 2000
MAX_RADIAL_ERROR_RATIO = 0.18
MIN_ANGLE_PROGRESS_DEG = 35
DRAW_THICKNESS = 2
SHOW_STATUS = True
SHOW_DEBUG = False


def fit_circle(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return None
    x, y = pts[:, 0], pts[:, 1]
    mat = np.column_stack((x, y, np.ones_like(x)))
    rhs = -(x * x + y * y)
    try:
        a, b, c = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = -a / 2.0, -b / 2.0
    radius_sq = cx * cx + cy * cy - c
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        return None
    radius = math.sqrt(radius_sq)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(cx), float(cy), float(radius), float(np.mean(np.abs(d - radius)))


def angular_progress(points, cx, cy):
    pts = np.asarray(points, dtype=np.float64)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    diffs = np.diff(angles)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    return float(np.sum(np.abs(diffs)) * 180.0 / np.pi)


def main():
    global SHOW_DEBUG

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}; try index 1 or close other camera apps.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    k = max(1, int(MORPH_KERNEL))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=HISTORY, varThreshold=VAR_THRESHOLD, detectShadows=False
    )
    centers = deque(maxlen=TRACK_HISTORY)
    previous_center = None
    circle_confirmed = False
    last_time = time.perf_counter()
    fps = 0.0

    print("Arbitrary-color moving shape detector; no circular track is drawn or required.")
    print("Keys: q/ESC quit | r reset tracking | d toggle debug window")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            smooth = cv2.GaussianBlur(frame, (5, 5), 0)
            fg = subtractor.apply(smooth, learningRate=-1)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_OBJECT_AREA or (MAX_OBJECT_AREA > 0 and area > MAX_OBJECT_AREA):
                    continue
                moments = cv2.moments(contour)
                if abs(moments["m00"]) < 1e-6:
                    continue
                center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                candidates.append((contour, area, center))

            selected = None
            if candidates:
                if previous_center is None:
                    selected = max(candidates, key=lambda item: item[1])
                else:
                    max_jump = max(frame.shape[1], frame.shape[0]) * 0.65
                    plausible = [item for item in candidates if math.dist(item[2], previous_center) <= max_jump]
                    if plausible:
                        selected = min(plausible, key=lambda item: math.dist(item[2], previous_center))

            # Always render on black; never show the camera frame or the circular path.
            output = np.zeros_like(frame)
            status = "Searching for moving shape"
            if selected is not None:
                contour, _, center = selected
                previous_center = center
                centers.append(center)

                if len(centers) >= MIN_POINTS_FOR_CIRCLE:
                    model = fit_circle(list(centers))
                    if model is not None:
                        cx, cy, radius, radial_error = model
                        progress = angular_progress(list(centers), cx, cy)
                        if (MIN_CIRCLE_RADIUS <= radius <= MAX_CIRCLE_RADIUS
                                and radial_error / max(radius, 1e-6) <= MAX_RADIAL_ERROR_RATIO
                                and progress >= MIN_ANGLE_PROGRESS_DEG):
                            circle_confirmed = True

                if circle_confirmed:
                    cv2.drawContours(output, [contour], -1, (0, 255, 0), DRAW_THICKNESS)
                    status = "Circular-motion shape"
                else:
                    status = "Validating circular motion"

            now = time.perf_counter()
            fps_now = 1.0 / max(now - last_time, 1e-6)
            fps = fps_now if fps == 0 else 0.9 * fps + 0.1 * fps_now
            last_time = now
            if SHOW_STATUS:
                cv2.putText(output, f"{status} | FPS: {fps:.1f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 1, cv2.LINE_AA)

            cv2.namedWindow("Circular Motion - Shape Only", cv2.WINDOW_NORMAL)
            cv2.imshow("Circular Motion - Shape Only", output)
            if SHOW_DEBUG:
                cv2.imshow("Foreground Debug", fg)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=HISTORY, varThreshold=VAR_THRESHOLD, detectShadows=False
                )
                centers.clear()
                previous_center = None
                circle_confirmed = False
            if key == ord("d"):
                SHOW_DEBUG = not SHOW_DEBUG
                if not SHOW_DEBUG:
                    cv2.destroyWindow("Foreground Debug")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
