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
    print("Press q or ESC to quit. Press [ / ] to change brightness threshold.")

    threshold = BRIGHT_THRESHOLD
    prev_time = time.perf_counter()
    fps_smooth = 0.0

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

            # 3) Canny captures edge transitions, including less-uniform edges.
            if USE_CANNY:
                edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
                combined = cv2.bitwise_or(bright_mask, edges)
            else:
                edges = np.zeros_like(gray)
                combined = bright_mask

            # 4) Optional, conservative cleanup.
            if close_kernel is not None:
                combined = cv2.morphologyEx(
                    combined, cv2.MORPH_CLOSE, close_kernel, iterations=1
                )
            if open_kernel is not None:
                combined = cv2.morphologyEx(
                    combined, cv2.MORPH_OPEN, open_kernel, iterations=1
                )

            # 5) Extract all contours. CHAIN_APPROX_NONE retains boundary pixels.
            contours, _ = cv2.findContours(
                combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )

            display = frame.copy()
            kept = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_CONTOUR_AREA:
                    continue
                if MAX_CONTOUR_AREA > 0 and area > MAX_CONTOUR_AREA:
                    continue
                kept.append(contour)

            cv2.drawContours(display, kept, -1, (0, 255, 0), 2)

            # FPS overlay
            now = time.perf_counter()
            dt = max(now - prev_time, 1e-6)
            instant_fps = 1.0 / dt
            fps_smooth = instant_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * instant_fps
            prev_time = now

            cv2.putText(
                display,
                f"FPS: {fps_smooth:.1f} | contours: {len(kept)} | threshold: {threshold}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                cv2.LINE_AA
            )

            # One resizable window containing a 2x2 diagnostic dashboard.
            # Resize each panel to the same size before assembling the grid.
            panel_w = max(320, actual_w // 2)
            panel_h = max(240, actual_h // 2)

            boundary_panel = cv2.resize(
                display, (panel_w, panel_h), interpolation=cv2.INTER_AREA
            )

            combined_view = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            bright_view = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
            edges_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            def add_title(panel, title):
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (25, 25, 25), -1)
                cv2.putText(
                    panel, title, (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
                return panel

            boundary_panel = add_title(boundary_panel, "Rotating Shape - Detected Boundary")
            combined_panel = add_title(
                cv2.resize(combined_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST),
                "Combined Mask"
            )
            bright_panel = add_title(
                cv2.resize(bright_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST),
                "Bright Mask"
            )
            edges_panel = add_title(
                cv2.resize(edges_view, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST),
                "Canny Edge Map"
            )

            top_row = np.hstack((boundary_panel, combined_panel))
            bottom_row = np.hstack((bright_panel, edges_panel))
            dashboard = np.vstack((top_row, bottom_row))

            cv2.namedWindow("Shape Detector Dashboard", cv2.WINDOW_NORMAL)
            cv2.imshow("Shape Detector Dashboard", dashboard)

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
