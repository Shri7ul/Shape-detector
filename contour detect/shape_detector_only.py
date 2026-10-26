import cv2
import numpy as np
import time

# ---------------- Configuration ----------------
CAMERA_INDEX = 0
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
REQUESTED_FPS = 60

# Tune these for your scene:
MIN_CONTOUR_AREA = 30       # reject tiny contours/noise
MAX_CONTOUR_AREA = 0        # 0 = no maximum
BRIGHT_THRESHOLD = 175      # higher = only brighter pixels
USE_CANNY = True            # combine bright pixels with Canny edges
CANNY_LOW = 35
CANNY_HIGH = 110

# Morphology: keep small to avoid distorting the shape
CLOSE_KERNEL = 3            # 0 disables; otherwise odd integer recommended
OPEN_KERNEL = 0             # 0 disables; otherwise odd integer recommended

# Set to True to try camera exposure controls (not supported by all cameras)
TRY_MANUAL_EXPOSURE = False
MANUAL_EXPOSURE = -6         # camera-specific; may be ignored
# ------------------------------------------------


def odd_kernel(size):
    size = int(size)
    if size <= 0:
        return None
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Try CAMERA_INDEX = 1, or close apps currently using the camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if TRY_MANUAL_EXPOSURE:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)

    close_kernel = odd_kernel(CLOSE_KERNEL)
    open_kernel = odd_kernel(OPEN_KERNEL)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {actual_w}x{actual_h}, reported FPS: {actual_fps:.1f}")
    print("Shows only motion-derived contours on a black background. Static scene is hidden.")
    print("Press q or ESC to quit. Press [ / ] to change brightness threshold.")

    threshold = BRIGHT_THRESHOLD
    prev_time = time.perf_counter()
    fps_smooth = 0.0
    previous_gray = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            # 1) Grayscale and mild denoising; avoid heavy blur on thin lines.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # 2) Bright outline mask: useful for a bright line on a dark interior.
            _, bright_mask = cv2.threshold(
                gray, threshold, 255, cv2.THRESH_BINARY
            )

            # 3) Detect temporal change. Static circular track/background should
            # mostly disappear; the moving shape remains in the difference mask.
            if previous_gray is None:
                motion_mask = np.zeros_like(gray)
            else:
                difference = cv2.absdiff(gray, previous_gray)
                _, motion_mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
                motion_mask = cv2.dilate(
                    motion_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1
                )
            previous_gray = gray.copy()

            # 4) Keep only bright pixels that also changed between frames.
            # This suppresses the stationary bright circle and favors the moving outline.
            moving_bright = cv2.bitwise_and(bright_mask, motion_mask)

            if USE_CANNY:
                edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
                moving_edges = cv2.bitwise_and(edges, motion_mask)
                combined = cv2.bitwise_or(moving_bright, moving_edges)
            else:
                edges = np.zeros_like(gray)
                combined = moving_bright

            # 5) Optional, conservative cleanup.
            if close_kernel is not None:
                combined = cv2.morphologyEx(
                    combined, cv2.MORPH_CLOSE, close_kernel, iterations=1
                )
            if open_kernel is not None:
                combined = cv2.morphologyEx(
                    combined, cv2.MORPH_OPEN, open_kernel, iterations=1
                )

            # 6) Extract contours from motion-filtered pixels, not the whole scene.
            contours, _ = cv2.findContours(
                combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            # Draw ONLY motion-derived contours on a black canvas.
            # The original frame and stationary circular track are not displayed.
            kept = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA:
                    continue
                if MAX_CONTOUR_AREA > 0 and area > MAX_CONTOUR_AREA:
                    continue
                kept.append(contour)

            output = np.zeros_like(frame)
            if kept:
                cv2.drawContours(output, kept, -1, (0, 255, 0), 2)

            # Optional status text can be disabled to keep only the detected shape.
            # Set SHOW_STATUS = True above if you want FPS/contour count displayed.

            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            instant_fps = 1.0 / dt
            fps_smooth = instant_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = now

            cv2.namedWindow("Moving Shape Only", cv2.WINDOW_NORMAL)
            cv2.imshow("Moving Shape Only", output)

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
