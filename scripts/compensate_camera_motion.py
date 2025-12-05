"""Compensate for small camera motion and stabilize saved trajectories.

This script:
 - Loads a vehicle tracks CSV (default: output/D1F1_stab_vehicle_tracks.csv)
 - Estimates frame-to-frame affine transforms using ORB feature matches
 - Accumulates transforms to a reference frame (min frame in CSV)
 - Transforms each trajectory point (which is associated with frames) into
   the stabilized reference frame
 - Writes a new CSV with stabilized trajectories and saves a visualization image

Usage:
 python scripts/compensate_camera_motion.py --csv output/D1F1_stab_vehicle_tracks.csv --video D1F1_stab.mp4
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import sys

# repo-safe imports when run from repo root
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def parse_frames_list(frames_field):
    if isinstance(frames_field, str):
        try:
            return json.loads(frames_field)
        except Exception:
            try:
                return eval(frames_field)
            except Exception:
                return []
    elif isinstance(frames_field, list):
        return frames_field
    else:
        return []


def parse_trajectory(traj_field):
    if isinstance(traj_field, str):
        try:
            return json.loads(traj_field)
        except Exception:
            try:
                return eval(traj_field)
            except Exception:
                return []
    elif isinstance(traj_field, list):
        return traj_field
    else:
        return []


def build_frame_range_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    all_frames = []
    for _, r in df.iterrows():
        frames = parse_frames_list(r['frames']) if 'frames' in r else []
        if frames:
            all_frames.extend(frames)
    if not all_frames:
        raise Exception('No frame indices found in CSV')
    return int(min(all_frames)), int(max(all_frames))


def estimate_affine_transforms(video_path: Path, start_frame: int, end_frame: int, max_features=2000):
    """Estimate affine (2x3) transform from frame i to frame i-1 for i in [start+1..end].

    Returns a dict transforms[f] = 2x3 affine that maps points in frame f -> frame f-1.
    """
    cap = cv2.VideoCapture(str(video_path))
    transforms = {}

    # read start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise FileNotFoundError(f"Could not read start frame {start_frame} from {video_path}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for f in range(start_frame + 1, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, cur = cap.read()
        if not ret:
            break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # detect and match
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(cur_gray, None)
        if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
            # fallback to identity translation
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            transforms[f] = M
            prev_gray = cur_gray
            continue

        try:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:200]
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


def accumulate_transforms(transforms: dict, start_frame: int, end_frame: int):
    """Compute cumulative transform from frame f to reference start_frame.

    cumulative[f] maps points in frame f -> reference frame (start_frame).
    """
    cumulative = {}
    # identity for start_frame
    cumulative[start_frame] = np.eye(3, dtype=np.float32)

    # For f = start+1..end, cumulative[f] = cumulative[f-1] @ T where T maps f->f-1
    for f in range(start_frame + 1, end_frame + 1):
        T = transforms.get(f, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        # make 3x3
        T33 = np.eye(3, dtype=np.float32)
        T33[0:2, :] = T
        # cumulative[f] = cumulative[f-1] @ T33
        cumulative[f] = cumulative[f - 1] @ T33
    return cumulative


def transform_point(pt, M33):
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float32)
    res = M33 @ vec
    return [float(res[0]), float(res[1])]


def stabilize_csv(csv_path: Path, video_path: Path, out_csv: Path = None, visualize=True):
    df = pd.read_csv(csv_path)
    # get frame range
    min_f, max_f = build_frame_range_from_csv(csv_path)
    print(f"Estimating transforms for frames {min_f}..{max_f} (this may take a while)")
    transforms = estimate_affine_transforms(video_path, min_f, max_f)
    cumulative = accumulate_transforms(transforms, min_f, max_f)

    # Apply stabilization: for each trajectory point, map using cumulative[frame]
    stabilized_rows = []
    stabilized_results = []

    for _, r in df.iterrows():
        frames = parse_frames_list(r['frames']) if 'frames' in r else []
        traj = parse_trajectory(r.get('trajectory_world') if 'trajectory_world' in r else r.get('trajectory'))
        if not frames or not traj:
            continue
        if len(frames) != len(traj):
            # Try best-effort: use index mapping
            L = min(len(frames), len(traj))
            frames = frames[:L]
            traj = traj[:L]

        stabilized_traj = []
        for f_idx, p in zip(frames, traj):
            if f_idx < min_f or f_idx > max_f:
                # outside processed range; keep as-is
                stabilized_traj.append([float(p[0]), float(p[1])])
            else:
                M33 = cumulative.get(f_idx, np.eye(3, dtype=np.float32))
                # map to reference: apply M33
                sp = transform_point(p, M33)
                stabilized_traj.append(sp)

        # simple smoothing (median) to reduce residual noise
        traj_np = np.array(stabilized_traj, dtype=np.float32)
        if traj_np.shape[0] >= 3:
            # median filter with kernel 3
            traj_sm = traj_np.copy()
            for i in range(1, traj_np.shape[0] - 1):
                traj_sm[i, 0] = np.median(traj_np[i - 1:i + 2, 0])
                traj_sm[i, 1] = np.median(traj_np[i - 1:i + 2, 1])
            stabilized_traj = traj_sm.tolist()

        stabilized_rows.append({
            'track_id': int(r['track_id']),
            'frames': json.dumps(frames),
            'avg_speed': float(r.get('avg_speed', 0.0)),
            'trajectory_len': len(stabilized_traj),
            'trajectory_world': json.dumps(stabilized_traj)
        })
        stabilized_results.append((int(r['track_id']), stabilized_traj))

    out_csv = out_csv or (csv_path.parent / f"{csv_path.stem}_stabilized.csv")
    pd.DataFrame(stabilized_rows).to_csv(out_csv, index=False)
    print(f"Wrote stabilized CSV: {out_csv}")

    if visualize and stabilized_results:
        # draw all stabilized trajectories on reference frame (min_f)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)
        ret, ref_frame = cap.read()
        cap.release()
        if not ret:
            ref_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        h, w = ref_frame.shape[:2]
        for tid, traj in stabilized_results:
            pts = []
            for p in traj:
                x = int(round(p[0]))
                y = int(round(p[1]))
                # clip
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                pts.append((x, y))
            color = tuple(int(x) for x in np.random.randint(50, 230, 3).tolist())
            if len(pts) >= 2:
                for i in range(1, len(pts)):
                    cv2.line(ref_frame, pts[i - 1], pts[i], color, 2)
            if pts:
                cv2.circle(ref_frame, pts[-1], 6, color, -1)
                cv2.putText(ref_frame, f"#{tid}", (pts[-1][0] + 6, pts[-1][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_img = csv_path.parent / f"{csv_path.stem}_stabilized.jpg"
        cv2.imwrite(str(out_img), ref_frame)
        print(f"Saved stabilized visualization: {out_img}")

    return out_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=str(repo_root / 'output' / 'D1F1_stab_vehicle_tracks.csv'))
    parser.add_argument('--video', type=str, default='D1F1_stab.mp4')
    parser.add_argument('--no_vis', action='store_true', help='Do not write visualization image')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    # find video path
    uploads = repo_root / 'uploads'
    video_path = uploads / args.video
    if not video_path.exists() and Path(args.video).exists():
        video_path = Path(args.video)

    if not video_path.exists():
        local = Path(r'C:/Users/sakth/Documents/traffique_footage') / args.video
        if local.exists():
            video_path = local
        else:
            raise FileNotFoundError(f"Video not found: {args.video}")

    stabilize_csv(csv_path, video_path, visualize=not args.no_vis)
