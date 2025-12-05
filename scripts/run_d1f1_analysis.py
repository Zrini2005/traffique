"""Run single-video ReID+tracker analysis for a specified video (D1F1 by default).

Produces:
 - output/d1f1_vehicle_tracks.csv
 - output/d1f1_trajectories.jpg

Usage:
 python scripts/run_d1f1_analysis.py --video D1F1_stab.mp4 --frame 9861 --time_window 10
"""
import argparse
import json
from pathlib import Path
import random
import sys

# Ensure repo root is on sys.path so local imports work when running as a script
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import pandas as pd

from interactive_analytics import VehicleAnalyzer
from utils.reid import ReIDExtractor
from utils.onlinetracker import OnlineTracker
from utils.trajectory import smooth_kalman, linear_interpolate

def _estimate_transforms(video_path, start_frame, end_frame, max_features=1500):
    """Estimate affine (2x3) transforms mapping frame f -> frame f-1 for the range.

    Returns dict transforms[f] = 2x3 affine mapping points in frame f -> frame f-1
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read start frame {start_frame} for stabilization")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    transforms = {}

    for f in range(start_frame + 1, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, cur = cap.read()
        if not ret:
            break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(cur_gray, None)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            transforms[f] = M
            prev_gray = cur_gray
            continue

        try:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:300]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            if len(pts1) >= 6:
                M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is None:
                    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            else:
                M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        except Exception:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        transforms[f] = M.astype(np.float32)
        prev_gray = cur_gray

    cap.release()
    return transforms


def _accumulate(transforms, start_frame, end_frame):
    cumulative = {}
    cumulative[start_frame] = np.eye(3, dtype=np.float32)
    for f in range(start_frame + 1, end_frame + 1):
        T = transforms.get(f, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        T33 = np.eye(3, dtype=np.float32)
        T33[0:2, :] = T
        cumulative[f] = cumulative[f - 1] @ T33
    return cumulative


def run_analysis(video_name: str, frame_idx: int = 0, time_window: int = 10, confidence: float = 0.2, use_sahi: bool = True, stabilize: bool = False):
    repo_root = Path(__file__).resolve().parents[1]
    uploads = repo_root / 'uploads'
    video_path = uploads / video_name
    if not video_path.exists():
        # try local footage folder
        local = Path(r'C:/Users/sakth/Documents/traffique_footage')
        if (local / video_name).exists():
            video_path = local / video_name
        else:
            raise FileNotFoundError(f"Video not found: {video_name}")

    print(f"Running analysis on: {video_path}")

    analyzer = VehicleAnalyzer(model_conf=confidence, use_sahi=use_sahi, sahi_slice_size=640)
    analyzer.load_model()

    reid = ReIDExtractor(model_path=str(repo_root / 'models' / 'osnet.pth'))
    tracker = OnlineTracker(max_missed=int(5 * (analyzer.fps if hasattr(analyzer, 'fps') else 25)),
                           dist_thresh_px=120,
                           appearance_weight=0.6)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(frame_idx)
    end_frame = int(frame_idx + max(1, int(time_window * fps)))

    print(f"Processing frames {start_frame} -> {end_frame} (fps={fps})")

    # If requested, estimate transforms for the small window and build cumulative maps
    cumulative = None
    if stabilize:
        print("Estimating stabilization transforms (this runs only for the time window)")
        transforms = _estimate_transforms(video_path, start_frame, end_frame)
        cumulative = _accumulate(transforms, start_frame, end_frame)

    for f in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break

        # If stabilizing, warp the current frame into reference (start_frame) space
        proc_frame = frame
        if stabilize and cumulative is not None and f in cumulative:
            M33 = cumulative[f]
            try:
                Minv = np.linalg.inv(M33)
                Maff = Minv[0:2, :]
                h, w = frame.shape[:2]
                proc_frame = cv2.warpAffine(frame, Maff, (w, h), flags=cv2.INTER_LINEAR)
            except Exception:
                proc_frame = frame

        detections = analyzer._detect_vehicles(proc_frame)
        processed = []
        for d in detections:
            bbox = [int(round(x)) for x in d['bbox']]
            emb = reid.extract(proc_frame, bbox)
            processed.append({'bbox': bbox, 'score': float(d.get('confidence', 0.0)), 'class': d.get('class_name', ''), 'embedding': emb})

        tracker.update(processed, frame_idx=f)

    cap.release()

    tracks = tracker.get_active_tracks(min_length=2)
    results = []

    # For visualization use stabilized reference frame if requested
    vis_frame = None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame if stabilize else frame_idx)
    ret, vis_frame = cap.read()
    cap.release()
    if vis_frame is None:
        vis_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    out_dir = repo_root / 'output'
    out_dir.mkdir(exist_ok=True)

    colors = {}
    rows = []

    for t in tracks:
        centers_px = [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in t['bboxes']]

        # world mapping if available
        traj_world = []
        if hasattr(analyzer.coord_mapper, 'H') and analyzer.coord_mapper.calibrated:
            H = analyzer.coord_mapper.H.astype(np.float32)
            for c in centers_px:
                pt = np.array([[[c[0], c[1]]]], dtype=np.float32)
                w = cv2.perspectiveTransform(pt, H)[0][0]
                traj_world.append([float(w[0]), float(w[1])])
        else:
            traj_world = [[float(x), float(y)] for x, y in centers_px]

        traj_sm = smooth_kalman(traj_world, dt=1.0, process_var=1.0, meas_var=4.0)

        # compute avg speed in world units (or pixels if not calibrated)
        velocities = []
        for i in range(1, len(traj_sm)):
            dx = traj_sm[i][0] - traj_sm[i - 1][0]
            dy = traj_sm[i][1] - traj_sm[i - 1][1]
            velocities.append(np.sqrt(dx * dx + dy * dy))
        avg_speed = float(np.mean(velocities)) if velocities else 0.0

        results.append({
            'track_id': int(t['track_id']),
            'frames': t['frames'],
            'bbox_first': t['bboxes'][0],
            'bbox_last': t['bboxes'][-1],
            'trajectory_world': traj_sm,
            'avg_speed': avg_speed
        })

        rows.append({
            'track_id': int(t['track_id']),
            'frames': json.dumps(t['frames']),
            'avg_speed': avg_speed,
            'trajectory_len': len(traj_sm),
            'trajectory_world': json.dumps(traj_sm)
        })

        # draw on vis_frame
        cid = t['track_id']
        if cid not in colors:
            colors[cid] = tuple(int(x) for x in np.random.randint(50, 230, size=3).tolist())
        col = colors[cid]

        # draw smoothed trajectory (map back to pixel space if needed)
        if hasattr(analyzer.coord_mapper, 'H') and analyzer.coord_mapper.calibrated:
            # map world back to image using inverse homography
            Hinv = np.linalg.inv(analyzer.coord_mapper.H.astype(np.float32))
            pts = []
            for p in traj_sm:
                pt = np.array([[[p[0], p[1]]]], dtype=np.float32)
                img_pt = cv2.perspectiveTransform(pt, Hinv)[0][0]
                pts.append((int(img_pt[0]), int(img_pt[1])))
        else:
            pts = [(int(x), int(y)) for x, y in centers_px]

        for i in range(1, len(pts)):
            cv2.line(vis_frame, pts[i - 1], pts[i], col, 2)
        if pts:
            cv2.circle(vis_frame, pts[-1], 6, col, -1)
            cv2.putText(vis_frame, f"#{cid}", (pts[-1][0] + 6, pts[-1][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{video_name[:-4]}_vehicle_tracks.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save visualization
    vis_path = out_dir / f"{video_name[:-4]}_trajectories.jpg"
    cv2.imwrite(str(vis_path), vis_frame)
    print(f"Saved visualization: {vis_path}")

    return csv_path, vis_path, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='D1F1_stab.mp4')
    parser.add_argument('--frame', type=int, default=9861)
    parser.add_argument('--time_window', type=int, default=15)
    parser.add_argument('--confidence', type=float, default=0.2)
    parser.add_argument('--use_sahi', action='store_true')
    parser.add_argument('--stabilize', action='store_true', help='Estimate and apply per-frame stabilization for the analysis window')

    args = parser.parse_args()

    csv_path, vis_path, results = run_analysis(args.video, frame_idx=args.frame, time_window=args.time_window, confidence=args.confidence, use_sahi=args.use_sahi)
    print('Done. Results:', len(results), 'tracks')
