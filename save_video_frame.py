import sys
import cv2
from pathlib import Path

"""
Save a specific frame from a video to an image file.
Usage:
  python save_video_frame.py <video_path> <frame_number> [output_path]
Example:
  python save_video_frame.py C:\\path\\to\\video.mp4 100 output/frame_0100.png
"""

def main():
    if len(sys.argv) < 3:
        print("Usage: python save_video_frame.py <video_path> <frame_number> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_no = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) >= 4 else f"output/frames/frame_{frame_no:06d}.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_no < 0 or frame_no >= total:
        print(f"Invalid frame_number {frame_no}. Video has {total} frames (0..{total-1}).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Failed to read frame.")
        sys.exit(1)

    cv2.imwrite(output_path, frame)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
