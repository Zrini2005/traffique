import cv2
import sys
from pathlib import Path
from interactive_analytics import VehicleAnalyzer

"""
Scan the video and save a frame that contains a clear car.
It samples every N frames, runs detection, and picks the frame
with the largest detected vehicle or highest vehicle count.

Usage:
  python find_car_frame.py <video_path> [stride] [output_path]
Example:
  python find_car_frame.py C:\\path\\to\\video.mp4 50 output/frames/best_car_frame.png
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_car_frame.py <video_path> [stride] [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    output_path = sys.argv[3] if len(sys.argv) >= 4 else "output/frames/best_car_frame.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    analyzer = VehicleAnalyzer(model_conf=0.25, use_sahi=True, sahi_slice_size=640)
    print("Loading VisDrone detection model...")
    analyzer.load_model()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best_score = -1.0
    best_frame = None
    best_index = -1

    # Full-frame polygon for ROI
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    full_poly = [(0,0),(width,0),(width,height),(0,height)]

    print(f"Scanning {total} frames with stride {stride}...")
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        result = analyzer.analyze_frame(frame, full_poly)
        dets = result.get('roi_vehicles', [])
        # Score: max bbox area + count bonus
        areas = []
        for d in dets:
            x1,y1,x2,y2 = d['bbox']
            areas.append(max(0, (x2-x1) * (y2-y1)))
        score = (max(areas) if areas else 0) + 5000 * len(dets)
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_index = idx
        # Progress
        if (idx // stride) % 20 == 0:
            print(f"... at frame {idx} / {total}")
        idx += stride

    cap.release()

    if best_frame is None:
        print("No suitable frame found.")
        sys.exit(1)

    cv2.imwrite(output_path, best_frame)
    print(f"Saved best car frame (index {best_index}) to: {output_path}")

if __name__ == '__main__':
    main()
