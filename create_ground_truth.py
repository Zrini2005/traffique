#!/usr/bin/env python3
"""
Ground Truth Annotation Tool for Trajectory Accuracy Testing

Simple GUI to manually annotate vehicle positions for ground truth.
Click on vehicles to mark their center positions.

Usage:
  python create_ground_truth.py --video <video_path> --output <gt_csv> [--frames frame1,frame2,...]
"""

import cv2
import pandas as pd
import argparse
from pathlib import Path
import numpy as np


class GroundTruthAnnotator:
    """Manual annotation tool for ground truth trajectories"""
    
    def __init__(self, video_path: str, output_csv: str):
        self.video_path = video_path
        self.output_csv = output_csv
        self.annotations = []
        self.current_vehicle_id = 1
        self.current_frame = None
        self.current_frame_no = 0
        
    def annotate_frame(self, frame_no: int):
        """Annotate a single frame"""
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Failed to read frame {frame_no}")
            return
        
        self.current_frame = frame.copy()
        self.current_frame_no = frame_no
        display = frame.copy()
        
        print(f"\n📍 Annotating Frame {frame_no}")
        print("  • Left click: Mark vehicle center")
        print("  • 'n': Next frame")
        print("  • 'u': Undo last point")
        print("  • 's': Save and exit")
        print("  • 'q': Quit without saving")
        
        frame_annotations = []
        
        def on_mouse(event, x, y, flags, param):
            nonlocal display, frame_annotations
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add annotation
                frame_annotations.append({
                    'vehicle_id': self.current_vehicle_id,
                    'frame': frame_no,
                    'x_px': x,
                    'y_px': y
                })
                self.current_vehicle_id += 1
                
                # Draw marker
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display, f"V{self.current_vehicle_id-1}", (x+15, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('annotate', display)
                print(f"    ✓ Vehicle {self.current_vehicle_id-1}: ({x}, {y})")
        
        cv2.namedWindow('annotate', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('annotate', 1280, 720)
        cv2.imshow('annotate', display)
        cv2.setMouseCallback('annotate', on_mouse)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                # Save annotations for this frame
                self.annotations.extend(frame_annotations)
                print(f"  ✅ Saved {len(frame_annotations)} annotations for frame {frame_no}")
                break
            elif key == ord('u'):
                # Undo last annotation
                if frame_annotations:
                    removed = frame_annotations.pop()
                    self.current_vehicle_id -= 1
                    display = self.current_frame.copy()
                    for ann in frame_annotations:
                        cv2.circle(display, (ann['x_px'], ann['y_px']), 8, (0, 255, 0), -1)
                        cv2.circle(display, (ann['x_px'], ann['y_px']), 10, (255, 255, 255), 2)
                        cv2.putText(display, f"V{ann['vehicle_id']}", 
                                  (ann['x_px']+15, ann['y_px']-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow('annotate', display)
                    print(f"    ↩️  Undid Vehicle {removed['vehicle_id']}")
            elif key == ord('s'):
                self.annotations.extend(frame_annotations)
                self.save()
                cv2.destroyAllWindows()
                return True
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def save(self):
        """Save annotations to CSV"""
        if not self.annotations:
            print("No annotations to save.")
            return
        
        df = pd.DataFrame(self.annotations)
        output_path = Path(self.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(self.annotations)} annotations to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual ground truth annotation tool"
    )
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--output", required=True, help="Output ground truth CSV")
    parser.add_argument("--frames", help="Comma-separated frame numbers (e.g., 100,200,300)")
    parser.add_argument("--stride", type=int, default=100, 
                       help="Annotate every Nth frame (if --frames not specified)")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of frames to annotate (if --frames not specified)")
    
    args = parser.parse_args()
    
    # Determine frames to annotate
    if args.frames:
        frame_list = [int(f.strip()) for f in args.frames.split(',')]
    else:
        # Auto-select frames
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_list = list(range(0, min(total_frames, args.count * args.stride), args.stride))
    
    print(f"\n{'='*70}")
    print(f"  GROUND TRUTH ANNOTATION TOOL")
    print(f"{'='*70}")
    print(f"\nVideo: {args.video}")
    print(f"Frames to annotate: {frame_list}")
    print(f"Output: {args.output}\n")
    
    annotator = GroundTruthAnnotator(args.video, args.output)
    
    for frame_no in frame_list:
        cont = annotator.annotate_frame(frame_no)
        if not cont:
            print("\n⚠️  Annotation cancelled.")
            break
    else:
        annotator.save()
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
