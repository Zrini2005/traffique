"""Simple script to show trajectories for up to N vehicles from a CSV.

Reads an existing vehicle_tracks CSV produced by the analysis scripts (default
`output/D1F1_stab_vehicle_tracks.csv`), picks the top-N tracks (by
`trajectory_len`) and draws those smoothed trajectories on the requested video
frame. Saves image to `output/{video_name[:-4]}_topN_trajectories.jpg`.

Usage examples:
 python scripts/show_top5_trajectories.py --video D1F1_stab.mp4 --frame 9861 --n 5
 python scripts/show_top5_trajectories.py --csv output/D1F1_stab_vehicle_tracks.csv --frame 9861 --n 5
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import sys

# make repo root import-safe if run from repo root
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def load_csv(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def parse_trajectory(traj_field):
    # traj_field may already be a list (if pandas parsed it) or a JSON string
    if isinstance(traj_field, str):
        try:
            return json.loads(traj_field)
        except Exception:
            # fallback: try to evaluate naive Python list
            try:
                return eval(traj_field)
            except Exception:
                return []
    elif isinstance(traj_field, list):
        return traj_field
    else:
        return []


def find_video_path(video_name: str):
    uploads = repo_root / 'uploads'
    candidate = uploads / video_name
    if candidate.exists():
        return candidate
    local = Path(r'C:/Users/sakth/Documents/traffique_footage') / video_name
    if local.exists():
        return local
    # fallback: the video name as an absolute path
    if Path(video_name).exists():
        return Path(video_name)
    raise FileNotFoundError(f"Video not found: {video_name}")


def draw_top_n(csv_path: Path, video_path: Path, frame_idx: int = 0, n: int = 5, out_dir: Path = None):
    df = load_csv(csv_path)
    # Default selection: choose tracks with motion (avg_speed) when available
    if 'avg_speed' in df.columns:
        # sort by avg_speed descending (most moving vehicles first)
        df_sorted = df.sort_values(by='avg_speed', ascending=False)
    elif 'trajectory_len' in df.columns:
        df_sorted = df.sort_values(by='trajectory_len', ascending=False)
    else:
        df_sorted = df

    top = df_sorted.head(n)
    selected = []
    for _, row in top.iterrows():
        try:
            tid = int(row['track_id'])
        except Exception:
            tid = None
        traj = parse_trajectory(row.get('trajectory_world') if 'trajectory_world' in row else row.get('trajectory'))
        if traj:
            selected.append((tid, traj, float(row.get('avg_speed', 0.0))))

    if not selected:
        raise Exception('No trajectories found in CSV to draw')

    # load frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        # fallback: blank image
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    h, w = frame.shape[:2]

    colors = {}
    for i, (tid, traj, avg_speed) in enumerate(selected):
        # choose deterministic color per track (if tid available) else by index
        if tid is not None:
            rng = np.random.RandomState(tid)
            color = tuple(int(x) for x in rng.randint(50, 230, size=3).tolist())
        else:
            color = tuple(int(x) for x in np.random.randint(50, 230, size=3).tolist())

        pts = []
        for p in traj:
            try:
                x = int(round(p[0]))
                y = int(round(p[1]))
            except Exception:
                continue
            # clip to image
            if x < 0 or x >= w or y < 0 or y >= h:
                # still include but clip
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
            pts.append((x, y))

        if len(pts) >= 2:
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j - 1], pts[j], color, 2)
        if pts:
            cv2.circle(frame, pts[-1], 6, color, -1)
            # Show avg speed (units/frame) in label to help pick moving vehicles
            speed_label = f" {avg_speed:.2f}" if avg_speed and not np.isnan(avg_speed) else ""
            label = f"#{tid}{speed_label}" if tid is not None else f"#{i}{speed_label}"
            cv2.putText(frame, label, (pts[-1][0] + 6, pts[-1][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_dir = out_dir or (repo_root / 'output')
    out_dir.mkdir(exist_ok=True)
    video_name = video_path.name
    out_path = out_dir / f"{video_name[:-4]}_top{len(selected)}_trajectories.jpg"
    cv2.imwrite(str(out_path), frame)
    return out_path, [tid for tid, _, _ in selected]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=str(repo_root / 'output' / 'D1F1_stab_vehicle_tracks.csv'))
    parser.add_argument('--video', type=str, default='D1F1_stab.mp4')
    parser.add_argument('--frame', type=int, default=9861)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--out', type=str, default=str(repo_root / 'output'))

    args = parser.parse_args()

    csv_path = Path(args.csv)
    try:
        video_path = find_video_path(args.video)
    except FileNotFoundError:
        # if video not found, but a visualization image exists, use that as background
        vis_img = Path(args.csv).parent / f"{Path(args.video).stem}_trajectories.jpg"
        if vis_img.exists():
            # create a tiny wrapper that will copy vis_img to the topN file and exit
            dest = Path(args.out) / f"{Path(args.video).stem}_top{args.n}_trajectories.jpg"
            dest.parent.mkdir(exist_ok=True)
            import shutil
            shutil.copy(str(vis_img), str(dest))
            print(f"Video not found. Copied existing visualization to: {dest}")
            sys.exit(0)
        else:
            raise

    out_path, tids = draw_top_n(csv_path, video_path, frame_idx=args.frame, n=args.n, out_dir=Path(args.out))
    print(f"Saved top-{len(tids)} trajectories image: {out_path}")
    print("Tracks drawn:", tids)
