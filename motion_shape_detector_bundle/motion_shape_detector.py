#!/usr/bin/env python3
"""
Optimized moving-shape boundary detector with camera-motion compensation.

Install:
    pip install opencv-python numpy

Run:
    python motion_shape_detector.py
    python motion_shape_detector.py --source "video.mp4"
    python motion_shape_detector.py --source 0 --scale 0.75 --min-area 100

Keys:
    q / ESC : quit
    space   : pause/resume
    r       : reset tracking
    s       : save current green-outline image

Approach:
- Sparse pyramidal Lucas-Kanade optical flow estimates camera motion efficiently.
- RANSAC affine transform aligns the previous frame with the current frame.
- Residual frame difference detects motion that remains after camera compensation.
- A lightweight contour tracker associates detections across frames.
- Output canvas is black with only the selected target contour in green.

This is a practical general-purpose baseline, not a guarantee for every moving camera.
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

GREEN = (0, 255, 0)


def parse_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def resize_frame(frame, scale):
    if scale >= 0.999:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def estimate_camera_motion(prev_gray, curr_gray, max_corners=350):
    """Fast global camera motion estimate using LK optical flow + RANSAC affine fit."""
    h, w = curr_gray.shape
    identity = np.eye(2, 3, dtype=np.float32)

    points = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=0.01,
        minDistance=8, blockSize=7, useHarrisDetector=False
    )
    if points is None or len(points) < 8:
        return identity, 0.0

    moved, status, errors = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    if moved is None or status is None:
        return identity, 0.0

    good_old = points[status.ravel() == 1].reshape(-1, 2)
    good_new = moved[status.ravel() == 1].reshape(-1, 2)
    if len(good_old) < 8:
        return identity, 0.0

    # Reject extreme flow vectors before fitting global camera motion.
    displacement = np.linalg.norm(good_new - good_old, axis=1)
    keep = displacement < 0.45 * np.hypot(w, h)
    good_old, good_new = good_old[keep], good_new[keep]
    if len(good_old) < 8:
        return identity, 0.0

    affine, inliers = cv2.estimateAffinePartial2D(
        good_old, good_new, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=1500,
        confidence=0.98, refineIters=5
    )
    if affine is None or not np.isfinite(affine).all():
        return identity, 0.0

    quality = float(inliers.mean()) if inliers is not None and len(inliers) else 0.0
    if quality < 0.20:
        return identity, quality

    # Guard against implausible transforms, which often indicate tracking failure.
    scale = np.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2)
    translation = np.hypot(affine[0, 2], affine[1, 2])
    if not (0.65 <= scale <= 1.5) or translation > 0.55 * np.hypot(w, h):
        return identity, 0.0

    return affine.astype(np.float32), quality


def build_motion_mask(prev_gray, curr_gray, affine, quality):
    h, w = curr_gray.shape
    aligned = cv2.warpAffine(
        prev_gray, affine, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    valid = cv2.warpAffine(
        np.full((h, w), 255, np.uint8), affine, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Mild blur makes differencing less sensitive to sensor/compression noise.
    old = cv2.GaussianBlur(aligned, (3, 3), 0)
    new = cv2.GaussianBlur(curr_gray, (3, 3), 0)
    diff = cv2.absdiff(old, new)

    valid_values = diff[valid > 0]
    if valid_values.size < 100:
        return np.zeros_like(curr_gray), aligned, 20

    # Percentile-based threshold is more stable than mean/std when camera edges
    # or a large moving object occupy a substantial part of the frame.
    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values.astype(np.float32) - median)))
    threshold = int(np.clip(median + 3.0 * 1.4826 * mad, 12, 42))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(mask, valid)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)

    # When global motion fit is weak, suppress unreliable border pixels.
    if quality < 0.20:
        border = max(2, int(min(h, w) * 0.015))
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0

    return mask, aligned, threshold


def get_candidates(mask, min_area, max_area_fraction=0.25):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = mask.shape[0] * mask.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > frame_area * max_area_fraction:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4:
            continue
        candidates.append((contour, area, (x, y, w, h)))
    return candidates


def choose_candidate(candidates, last_center, last_box, frame_shape):
    if not candidates:
        return None

    h, w = frame_shape[:2]
    diagonal = max(1.0, np.hypot(w, h))
    scored = []
    for contour, area, box in candidates:
        x, y, bw, bh = box
        center = (x + bw / 2.0, y + bh / 2.0)
        fill = area / max(1.0, bw * bh)
        aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
        score = np.log1p(area) + 1.4 * fill - 0.25 * min(aspect, 8.0)

        if last_center is not None:
            distance = np.hypot(center[0] - last_center[0], center[1] - last_center[1])
            score -= 5.0 * distance / diagonal
        if last_box is not None:
            px, py, pw, ph = last_box
            ix1, iy1 = max(x, px), max(y, py)
            ix2, iy2 = min(x + bw, px + pw), min(y + bh, py + ph)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + pw * ph - intersection
            score += 2.0 * (intersection / union if union else 0.0)

        scored.append((score, contour, area, box, center))

    _, contour, area, box, center = max(scored, key=lambda item: item[0])
    return contour, area, box, center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index, video path, or stream URL")
    parser.add_argument("--scale", type=float, default=0.75, help="Processing scale 0.4-1.0; lower is faster")
    parser.add_argument("--min-area", type=float, default=90, help="Minimum contour area in resized-frame pixels")
    parser.add_argument("--max-corners", type=int, default=350, help="Maximum optical-flow features")
    parser.add_argument("--lost-frames", type=int, default=10, help="Frames without a match before target resets")
    args = parser.parse_args()

    args.scale = float(np.clip(args.scale, 0.4, 1.0))
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_gray = None
    last_center = None
    last_box = None
    lost_frames = 0
    paused = False
    last_outline = None
    fps_samples = deque(maxlen=25)
    min_area = max(20.0, args.min_area)

    cv2.namedWindow("Motion Shape Detector - Optimized", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Shape Detector - Optimized", 1200, 650)

    while True:
        if not paused:
            ok, raw = cap.read()
            if not ok:
                break

            frame = resize_frame(raw, args.scale)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            output = np.zeros_like(frame)
            motion_mask = np.zeros((h, w), np.uint8)
            quality = 0.0
            threshold = 0
            status = "Initializing first frame"

            if prev_gray is not None:
                affine, quality = estimate_camera_motion(prev_gray, gray, args.max_corners)
                motion_mask, _, threshold = build_motion_mask(prev_gray, gray, affine, quality)
                candidates = get_candidates(motion_mask, min_area)
                selected = choose_candidate(candidates, last_center, last_box, frame.shape)

                if selected is not None:
                    contour, area, box, center = selected
                    perimeter = cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, max(1.0, 0.0025 * perimeter), True)
                    cv2.drawContours(output, [contour], -1, GREEN, 2, cv2.LINE_AA)
                    last_outline = output.copy()
                    last_center, last_box = center, box
                    lost_frames = 0
                    status = f"Moving contour | area {area:.0f}px | candidates {len(candidates)}"
                else:
                    lost_frames += 1
                    status = f"Searching | candidates {len(candidates)}"
                    if lost_frames > args.lost_frames:
                        last_center, last_box, last_outline = None, None, None
            prev_gray = gray.copy()

            now = time.perf_counter()
            fps_samples.append(now)
            fps = (len(fps_samples) - 1) / max(1e-6, fps_samples[-1] - fps_samples[0]) if len(fps_samples) > 1 else 0.0

            left = output
            right = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            panel = np.hstack((left, right))
            cv2.putText(panel, "GREEN OUTLINE ONLY", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2)
            cv2.putText(panel, "MOTION MASK", (w + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
            cv2.putText(panel, status, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
            quality_text = "OK" if quality >= 0.20 else "LOW"
            cv2.putText(panel, f"Camera model: {quality_text} ({quality:.2f}) | threshold {threshold} | FPS {fps:.1f}",
                        (w + 12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv2.imshow("Motion Shape Detector - Optimized", panel)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            prev_gray = None
            last_center = last_box = last_outline = None
            lost_frames = 0
        elif key == ord("s") and last_outline is not None:
            name = f"outline_{int(time.time())}.png"
            cv2.imwrite(name, last_outline)
            print(f"Saved {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
