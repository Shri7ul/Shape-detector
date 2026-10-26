#!/usr/bin/env python3
"""
Motion-based shape boundary detector with camera-motion compensation.

Install:
    pip install opencv-python numpy

Run:
    python motion_shape_detector.py
    python motion_shape_detector.py --source video.mp4
    python motion_shape_detector.py --source 0 --min-area 180

Controls:
    q / ESC : quit
    space   : pause/resume
    r       : reset target tracking
    s       : save the current outline image

Design:
1. Estimate global camera motion between frames using ORB feature matches + RANSAC homography.
2. Warp the previous frame into the current frame's coordinate system.
3. Detect residual motion (moving objects not explained by camera motion).
4. Refine the residual mask using color/edge information from the current frame.
5. Track the selected moving object by overlap/proximity, not by color alone.
6. Draw only its contour in green on a black canvas.

Limitations:
- If the camera moves while the entire scene has little texture, global motion estimation can fail.
- A moving object that is fully motion-blurred or visually identical to a moving background may not be separable.
- For best results use a stable frame rate, good lighting, and a target that occupies enough pixels.
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np


GREEN = (0, 255, 0)


def parse_source(value):
    """Interpret a numeric source as a webcam index; otherwise use it as a path/URL."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def resize_for_processing(frame, max_width=960):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale))), scale


def estimate_camera_transform(prev_gray, curr_gray):
    """
    Estimate previous-frame -> current-frame homography.
    ORB+RANSAC tolerates outlier feature matches. Returns (H, quality).
    """
    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    identity = np.eye(3, dtype=np.float32)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return identity, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 8:
        return identity, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)

    if H is None or not np.isfinite(H).all():
        return identity, 0.0

    inlier_ratio = float(inliers.mean()) if inliers is not None and len(inliers) else 0.0
    # Reject clearly unreliable transforms rather than applying a wild warp.
    if inlier_ratio < 0.18 or abs(H[2, 2]) < 1e-8:
        return identity, 0.0

    H = H.astype(np.float32)
    H /= H[2, 2]
    return H, inlier_ratio


def make_motion_mask(prev_gray, curr_gray, H, camera_quality):
    h, w = curr_gray.shape

    # Align previous image to current frame to cancel global camera movement.
    aligned_prev = cv2.warpPerspective(
        prev_gray, H, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    valid_prev = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), H, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Blur before differencing to reduce sensor noise/compression flicker.
    a = cv2.GaussianBlur(aligned_prev, (5, 5), 0)
    b = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    diff = cv2.absdiff(a, b)

    # Adaptive threshold with a practical floor for compressed videos.
    mean, std = cv2.meanStdDev(diff, mask=valid_prev)
    threshold = int(np.clip(float(mean[0, 0]) + 2.2 * float(std[0, 0]), 16, 45))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, valid_prev)

    # Remove isolated noise and connect nearby pixels on the moving target.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)

    # If camera estimation failed, be conservative: differencing can otherwise
    # classify most of a camera-shifted background as motion.
    if camera_quality < 0.18:
        mask = cv2.erode(mask, k3, iterations=1)

    return mask, aligned_prev, threshold


def candidate_contours(mask, min_area, max_area_fraction=0.45):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = mask.shape[0] * mask.shape[1]
    result = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > frame_area * max_area_fraction:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 4 or h < 4:
            continue
        result.append((c, area, (x, y, w, h)))
    return result


def select_candidate(candidates, previous_center, previous_box, frame_shape):
    """
    Choose a candidate using size + continuity. With no previous target, prefer
    a compact, meaningful residual-motion region rather than the largest blob.
    """
    if not candidates:
        return None

    fh, fw = frame_shape[:2]
    frame_area = fh * fw
    scored = []

    for contour, area, box in candidates:
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        fill = area / max(1.0, w * h)
        aspect = max(w / max(1.0, h), h / max(1.0, w))
        # Prefer coherent blobs and avoid giant residual regions from camera motion.
        score = np.log1p(area) + 1.5 * fill - 0.35 * min(aspect, 8.0)
        if area > frame_area * 0.12:
            score -= 5.0

        if previous_center is not None:
            distance = np.hypot(cx - previous_center[0], cy - previous_center[1])
            diag = np.hypot(fw, fh)
            score -= 4.0 * (distance / max(1.0, diag))
            if previous_box is not None:
                px, py, pw, ph = previous_box
                ix1, iy1 = max(x, px), max(y, py)
                ix2, iy2 = min(x + w, px + pw), min(y + h, py + ph)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = w * h + pw * ph - inter
                score += 2.5 * (inter / union if union else 0.0)

        scored.append((score, contour, area, box, (cx, cy)))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0]
    return best[1], best[2], best[3], best[4]


def smooth_contour(contour, epsilon_ratio=0.003):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return contour
    approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
    return approx if len(approx) >= 3 else contour


def main():
    parser = argparse.ArgumentParser(description="Moving shape boundary detector with camera-motion compensation.")
    parser.add_argument("--source", default="0", help="Webcam index (default 0), video path, or stream URL")
    parser.add_argument("--min-area", type=float, default=100.0, help="Minimum moving contour area in processing pixels")
    parser.add_argument("--max-width", type=int, default=960, help="Processing width cap; lower is faster")
    parser.add_argument("--lost-frames", type=int, default=8, help="Frames to retain tracking continuity after target is lost")
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    # Best-effort camera settings; files/URLs may ignore these.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    prev_gray = None
    previous_center = None
    previous_box = None
    lost_count = 0
    paused = False
    last_outline = None
    fps_times = deque(maxlen=20)
    min_area = max(20.0, args.min_area)

    cv2.namedWindow("Motion Shape Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Shape Detector", 1280, 760)

    while True:
        if not paused:
            ok, raw = cap.read()
            if not ok:
                # For a video file, stop at EOF; for webcam, retry briefly.
                if isinstance(source, str):
                    break
                time.sleep(0.02)
                continue

            frame, scale = resize_for_processing(raw, args.max_width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            status = "Initializing..."
            motion_mask = np.zeros((h, w), np.uint8)
            output = np.zeros_like(frame)
            camera_quality = 0.0
            threshold = 0

            if prev_gray is not None:
                H, camera_quality = estimate_camera_transform(prev_gray, gray)
                motion_mask, aligned_prev, threshold = make_motion_mask(
                    prev_gray, gray, H, camera_quality
                )

                # Detect candidate regions from residual motion.
                candidates = candidate_contours(motion_mask, min_area)
                selected = select_candidate(candidates, previous_center, previous_box, frame.shape)

                if selected is not None:
                    contour, area, box, center = selected
                    contour = smooth_contour(contour)

                    # Display only the current boundary, not the source frame/orbit.
                    cv2.drawContours(output, [contour], -1, GREEN, 2, cv2.LINE_AA)
                    last_outline = output.copy()
                    previous_center = center
                    previous_box = box
                    lost_count = 0
                    status = f"Target motion detected | area={area:.0f}px"
                else:
                    lost_count += 1
                    status = "Searching for moving object..."
                    if lost_count > args.lost_frames:
                        previous_center = None
                        previous_box = None
                        last_outline = None
            else:
                status = "Learning first frame..."

            # Single dashboard: left = black outline output, right = diagnostic motion mask.
            # No source video is rendered in the output pane.
            left = cv2.resize(output, (w, h))
            right = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            panel = np.hstack((left, right))
            cv2.putText(panel, "TARGET OUTLINE", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
            cv2.putText(panel, "RESIDUAL MOTION MASK", (w + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
            cv2.putText(panel, status, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cam_text = f"Camera-motion model: {'OK' if camera_quality >= 0.18 else 'LOW CONFIDENCE'} | diff threshold={threshold}"
            cv2.putText(panel, cam_text, (w + 15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

            now = time.time()
            fps_times.append(now)
            fps = (len(fps_times) - 1) / max(1e-6, fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0.0
            cv2.putText(panel, f"FPS: {fps:.1f} | q quit | space pause | r reset | s save", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            cv2.imshow("Motion Shape Detector", panel)
            prev_gray = gray.copy()

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            prev_gray = None
            previous_center = None
            previous_box = None
            lost_count = 0
            last_outline = None
        elif key == ord("s"):
            if last_outline is not None:
                filename = f"outline_{int(time.time())}.png"
                cv2.imwrite(filename, last_outline)
                print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
