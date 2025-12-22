#!/usr/bin/env python3
"""
Annotate a single vehicle across multiple frames for RMSE validation.
Shows the predicted position as reference.

Usage:
  python annotate_single_vehicle.py --frames <frame_list> --vehicle-id <id> --video <video_path> --output <gt_csv>
"""

import cv2
import pandas as pd
import argparse
from pathlib import Path


class SingleVehicleAnnotator:
    """Annotate one vehicle across multiple frames"""
    
    def __init__(self, video_path: str, vehicle_id: int, frames: list, output_csv: str, predictions_csv: str = None):
        self.video_path = video_path
        self.vehicle_id = vehicle_id
        self.frames = frames
        self.output_csv = output_csv
        self.annotations = []
        
        # Load predictions if provided (to show reference)
        self.predictions = {}
        if predictions_csv and Path(predictions_csv).exists():
            pred_df = pd.read_csv(predictions_csv)
            for _, row in pred_df.iterrows():
                self.predictions[int(row['frame'])] = (float(row['x_px']), float(row['y_px']))
    
    def annotate_all_frames(self):
        """Annotate the vehicle in all frames"""
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {self.video_path}")
            return
        
        print(f"\n{'='*70}")
        print(f"  SINGLE VEHICLE ANNOTATION - Vehicle {self.vehicle_id}")
        print(f"{'='*70}")
        print(f"\nFrames to annotate: {self.frames}")
        print(f"\nInstructions:")
        print(f"  • Left-click on the CENTER of vehicle {self.vehicle_id}")
        print(f"  • Yellow crosshair shows predicted position (if available)")
        print(f"  • Press SPACE to confirm and move to next frame")
        print(f"  • Press 'u' to undo current frame")
        print(f"  • Press 's' to save and exit")
        print(f"  • Press 'q' to quit without saving\n")
        
        for frame_no in self.frames:
            success = self.annotate_frame(cap, frame_no)
            if not success:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.annotations:
            self.save()
    
    def annotate_frame(self, cap, frame_no):
        """Annotate a single frame"""
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to read frame {frame_no}")
            return True
        
        display = frame.copy()
        current_point = None
        
        # Show prediction if available
        if frame_no in self.predictions:
            pred_x, pred_y = self.predictions[frame_no]
            cv2.drawMarker(display, (int(pred_x), int(pred_y)), (0, 255, 255), 
                          cv2.MARKER_CROSS, 40, 3)
            cv2.putText(display, "Predicted", (int(pred_x)+25, int(pred_y)-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions overlay
        cv2.putText(display, f"Frame {frame_no} - Vehicle {self.vehicle_id}", (30, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, "Click vehicle center | SPACE=confirm | u=undo | s=save | q=quit", (30, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        def on_mouse(event, x, y, flags, param):
            nonlocal current_point, display
            if event == cv2.EVENT_LBUTTONDOWN:
                current_point = (x, y)
                # Redraw
                display = frame.copy()
                
                # Show prediction
                if frame_no in self.predictions:
                    pred_x, pred_y = self.predictions[frame_no]
                    cv2.drawMarker(display, (int(pred_x), int(pred_y)), (0, 255, 255), 
                                  cv2.MARKER_CROSS, 40, 3)
                    cv2.putText(display, "Predicted", (int(pred_x)+25, int(pred_y)-25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw annotation
                cv2.circle(display, current_point, 10, (0, 255, 0), -1)
                cv2.circle(display, current_point, 15, (255, 255, 255), 2)
                cv2.putText(display, f"Ground Truth ({x}, {y})", (x+20, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Instructions
                cv2.putText(display, f"Frame {frame_no} - Vehicle {self.vehicle_id}", (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(display, "Click vehicle center | SPACE=confirm | u=undo | s=save | q=quit", (30, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                cv2.imshow('annotate', display)
        
        cv2.namedWindow('annotate', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('annotate', 1280, 720)
        cv2.imshow('annotate', display)
        cv2.setMouseCallback('annotate', on_mouse)
        
        print(f"\n📍 Frame {frame_no}:")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' '):  # Space - confirm
                if current_point:
                    self.annotations.append({
                        'vehicle_id': self.vehicle_id,
                        'frame': frame_no,
                        'x_px': current_point[0],
                        'y_px': current_point[1]
                    })
                    print(f"  ✓ Annotated: ({current_point[0]}, {current_point[1]})")
                    
                    # Show comparison if prediction available
                    if frame_no in self.predictions:
                        pred_x, pred_y = self.predictions[frame_no]
                        error = ((current_point[0]-pred_x)**2 + (current_point[1]-pred_y)**2)**0.5
                        print(f"  → Error from prediction: {error:.1f} px")
                    
                    return True
                else:
                    print("  ⚠️  No point marked. Click on vehicle first.")
            
            elif key == ord('u'):  # Undo
                current_point = None
                display = frame.copy()
                if frame_no in self.predictions:
                    pred_x, pred_y = self.predictions[frame_no]
                    cv2.drawMarker(display, (int(pred_x), int(pred_y)), (0, 255, 255), 
                                  cv2.MARKER_CROSS, 40, 3)
                cv2.putText(display, f"Frame {frame_no} - Vehicle {self.vehicle_id}", (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(display, "Click vehicle center | SPACE=confirm | u=undo | s=save | q=quit", (30, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow('annotate', display)
                print("  ↩️  Cleared point")
            
            elif key == ord('s'):  # Save
                if current_point:
                    self.annotations.append({
                        'vehicle_id': self.vehicle_id,
                        'frame': frame_no,
                        'x_px': current_point[0],
                        'y_px': current_point[1]
                    })
                    print(f"  ✓ Annotated: ({current_point[0]}, {current_point[1]})")
                self.save()
                return False
            
            elif key == ord('q'):  # Quit
                print("  ⚠️  Quit without saving")
                return False
    
    def save(self):
        """Save annotations to CSV"""
        if not self.annotations:
            print("\n⚠️  No annotations to save.")
            return
        
        df = pd.DataFrame(self.annotations)
        output_path = Path(self.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(self.annotations)} annotations to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate single vehicle for RMSE validation"
    )
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--vehicle-id", type=int, required=True, help="Vehicle ID")
    parser.add_argument("--frames", required=True, help="Comma-separated frame numbers")
    parser.add_argument("--output", required=True, help="Output ground truth CSV")
    parser.add_argument("--predictions", help="Prediction CSV (optional, for reference overlay)")
    
    args = parser.parse_args()
    
    frames = [int(f.strip()) for f in args.frames.split(',')]
    
    annotator = SingleVehicleAnnotator(args.video, args.vehicle_id, frames, args.output, args.predictions)
    annotator.annotate_all_frames()


if __name__ == "__main__":
    main()
