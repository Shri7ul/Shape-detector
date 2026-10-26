import cv2
import numpy as np
import time

# ------------------------ CONFIG ------------------------
CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
REQUESTED_FPS = 60

# Bright shape on a dark background
BRIGHT_THRESHOLD = 150
MIN_COMPONENT_AREA = 20
MAX_COMPONENT_AREA = 0       # 0 = unlimited
MOTION_THRESHOLD = 18
MORPH_KERNEL = 3

# If the circular track is visible but should be ignored, motion filtering
# is essential. This uses temporal difference and only draws contours from
# moving, bright pixels.
SHOW_DEBUG = False            # True opens an additional diagnostic window
# --------------------------------------------------------


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1 "
            "or close other camera apps."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    k = max(1, int(MORPH_KERNEL))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    previous_gray = None
    last_time = time.perf_counter()
    fps = 0.0
    threshold = BRIGHT_THRESHOLD

    print("Output: only detected moving bright shape on a black background.")
    print("Keys: q/ESC = quit, [ = lower brightness threshold, ] = raise it.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Bright regions
            _, bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

            # Temporal difference: stationary circular track should largely vanish.
            if previous_gray is None:
                motion = np.zeros_like(gray)
            else:
                diff = cv2.absdiff(gray, previous_gray)
                _, motion = cv2.threshold(
                    diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY
                )
            previous_gray = gray.copy()

            # Require pixels to be both bright and changing between frames.
            moving_bright = cv2.bitwise_and(bright, motion)
            moving_bright = cv2.morphologyEx(
                moving_bright, cv2.MORPH_OPEN, kernel, iterations=1
            )
            moving_bright = cv2.morphologyEx(
                moving_bright, cv2.MORPH_CLOSE, kernel, iterations=2
            )

            # Connected components filter out tiny isolated noise.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                moving_bright, connectivity=8
            )

            clean = np.zeros_like(moving_bright)
            for i in range(1, count):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < MIN_COMPONENT_AREA:
                    continue
                if MAX_COMPONENT_AREA > 0 and area > MAX_COMPONENT_AREA:
                    continue
                clean[labels == i] = 255

            # Contours are drawn ONLY on a black canvas.
            contours, _ = cv2.findContours(
                clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            output = np.zeros_like(frame)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

            now = time.perf_counter()
            dt = max(now - last_time, 1e-6)
            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            last_time = now

            # Keep output clean; no circle or original camera frame is shown.
            cv2.namedWindow("Moving Shape Only", cv2.WINDOW_NORMAL)
            cv2.imshow("Moving Shape Only", output)

            if SHOW_DEBUG:
                debug = np.hstack((
                    cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR),
                ))
                cv2.imshow("Debug: Bright | Motion | Moving Bright", debug)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("]"):
                threshold = min(254, threshold + 5)
            elif key == ord("["):
                threshold = max(0, threshold - 5)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
