#!/usr/bin/env python3
"""
Simple Optical Flow Tracker - Fast & Accurate
Uses optical flow to track vehicles between YOLO detections
Much faster than detecting every frame
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

class OpticalFlowTracker:
    """Simple tracker using optical flow + periodic re-detection"""
    
    def __init__(self, video_path: str, confidence: float = 0.4, 
                 redetect_interval: int = 10, min_trajectory_length: int = 30):
        self.video_path = video_path
        self.confidence = confidence
        self.redetect_interval = redetect_interval
        self.min_trajectory_length = min_trajectory_length
        
        # Load YOLO model (using existing VehicleAnalyzer infrastructure)
        print("Loading YOLO model...")
        from interactive_analytics import VehicleAnalyzer
        self.analyzer = VehicleAnalyzer(model_conf=confidence, use_sahi=False)
        self.analyzer.load_model()
        self.model = self.analyzer.model
        
        # Vehicle class IDs for VisDrone
        self.vehicle_classes = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car',
                               4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle',
                               8: 'bus', 9: 'motor'}
        self.vehicle_ids = [3, 4, 5, 8, 9]  # car, van, truck, bus, motor
        
        # Tracking data
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1
        self.trajectories: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        
        # Optical flow params (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles using YOLO"""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                
                if cls in self.vehicle_ids:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Filter small/malformed detections
                    width, height = x2 - x1, y2 - y1
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    if area >= 600 and 0.3 <= aspect_ratio <= 4.0:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': np.array([[center_x, center_y]], dtype=np.float32),
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': self.vehicle_classes.get(cls, 'vehicle')
                        })
        
        return detections
    
    def track_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> None:
        """Track all active tracks using optical flow"""
        
        if not self.tracks:
            return
        
        # Collect all centers to track
        track_ids = []
        points_prev = []
        
        for track_id, track in self.tracks.items():
            if track['active']:
                track_ids.append(track_id)
                points_prev.append(track['center'])
        
        if not points_prev:
            return
        
        points_prev = np.array(points_prev, dtype=np.float32).reshape(-1, 1, 2)
        
        # Calculate optical flow
        points_curr, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points_prev, None, **self.lk_params
        )
        
        # Update tracks with new positions
        for i, track_id in enumerate(track_ids):
            if status[i] == 1:  # Successfully tracked
                new_center = points_curr[i].reshape(1, 2)
                
                # Update track
                self.tracks[track_id]['center'] = new_center
                self.tracks[track_id]['bbox'] = self._bbox_from_center(
                    new_center[0], self.tracks[track_id]['size']
                )
                self.tracks[track_id]['age'] = 0
            else:
                # Lost track
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > 30:
                    self.tracks[track_id]['active'] = False
    
    def _bbox_from_center(self, center: np.ndarray, size: Tuple[float, float]) -> List[float]:
        """Create bbox from center point and size"""
        w, h = size
        return [center[0] - w/2, center[1] - h/2, center[0] + w/2, center[1] + h/2]
    
    def match_detections_to_tracks(self, detections: List[Dict], threshold: float = 50) -> None:
        """Match new detections to existing tracks"""
        
        matched_tracks = set()
        matched_detections = set()
        
        # Match detections to tracks
        for det_idx, det in enumerate(detections):
            best_dist = float('inf')
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if not track['active']:
                    continue
                
                # Calculate distance between detection and track
                dist = np.linalg.norm(det['center'][0] - track['center'][0])
                
                if dist < threshold and dist < best_dist:
                    # Check class match
                    if det['class_name'] == track['class_name']:
                        best_dist = dist
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id].update({
                    'center': det['center'],
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'age': 0,
                    'size': (det['bbox'][2] - det['bbox'][0], det['bbox'][3] - det['bbox'][1])
                })
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                self.tracks[self.next_id] = {
                    'center': det['center'],
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'age': 0,
                    'active': True,
                    'size': (det['bbox'][2] - det['bbox'][0], det['bbox'][3] - det['bbox'][1])
                }
                self.next_id += 1
    
    def track_video(self, start_frame: int = 0, num_frames: int = 400, 
                   roi_polygon: List[Tuple[int, int]] = None) -> str:
        """Track vehicles through video using optical flow"""
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        end_frame = min(start_frame + num_frames, total_frames)
        
        print(f"\n{'='*70}")
        print(f"SIMPLE OPTICAL FLOW TRACKER")
        print(f"{'='*70}")
        print(f"📹 Video: {self.video_path}")
        print(f"📊 Frames: {start_frame} → {end_frame} ({num_frames} frames)")
        print(f"⚡ FPS: {fps:.1f}")
        print(f"🔄 Re-detect interval: every {self.redetect_interval} frames")
        print(f"🎯 Min trajectory: {self.min_trajectory_length} frames")
        print(f"{'='*70}\n")
        
        # Jump to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        # Apply ROI crop if provided
        if roi_polygon and len(roi_polygon) == 4:
            y_min = min(p[1] for p in roi_polygon)
            y_max = max(p[1] for p in roi_polygon)
            x_min = min(p[0] for p in roi_polygon)
            x_max = max(p[0] for p in roi_polygon)
            prev_frame = prev_frame[y_min:y_max, x_min:x_max]
            offset_x, offset_y = x_min, y_min
        else:
            offset_x, offset_y = 0, 0
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Initial detection
        print("🔍 Initial detection...")
        detections = self.detect_vehicles(prev_frame)
        self.match_detections_to_tracks(detections)
        print(f"   Found {len(detections)} vehicles")
        
        frame_count = start_frame + 1
        processed = 1
        detection_count = 1
        
        while frame_count < end_frame:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Apply ROI crop
            if roi_polygon and len(roi_polygon) == 4:
                curr_frame = curr_frame[y_min:y_max, x_min:x_max]
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Every Nth frame: re-detect to correct drift
            if processed % self.redetect_interval == 0:
                detections = self.detect_vehicles(curr_frame)
                self.match_detections_to_tracks(detections)
                detection_count += 1
            else:
                # Track using optical flow (FAST!)
                self.track_optical_flow(prev_gray, curr_gray)
            
            # Save trajectory points for all active tracks
            for track_id, track in self.tracks.items():
                if track['active']:
                    center = track['center'][0]
                    self.trajectories[track_id].append((
                        frame_count,
                        center[0] + offset_x,
                        center[1] + offset_y
                    ))
            
            # Progress
            if processed % 1 == 0:
                active_tracks = sum(1 for t in self.tracks.values() if t['active'])
                print(f"  ⏳ Processed {processed}/{num_frames} frames... "
                      f"({active_tracks} active tracks, {detection_count} detections)")
            
            prev_gray = curr_gray
            frame_count += 1
            processed += 1
        
        cap.release()
        
        print(f"\n✅ Tracking complete!")
        print(f"   Total vehicles tracked: {len(self.trajectories)}")
        print(f"   YOLO inferences: {detection_count} (vs {num_frames} with per-frame detection)")
        print(f"   Speed improvement: {num_frames/detection_count:.1f}x faster!")
        
        return self._export_csv(start_frame)
    
    def _export_csv(self, start_frame: int) -> str:
        """Export trajectories to CSV"""
        
        # Filter by minimum length
        valid_trajectories = {
            vid: traj for vid, traj in self.trajectories.items()
            if len(traj) >= self.min_trajectory_length
        }
        
        print(f"\n📊 Valid trajectories: {len(valid_trajectories)}")
        print(f"   (filtered {len(self.trajectories) - len(valid_trajectories)} short tracks)")
        
        # Create CSV data
        rows = []
        for track_id, trajectory in valid_trajectories.items():
            class_name = self.tracks[track_id]['class_name']
            
            for frame, x, y in trajectory:
                rows.append({
                    'Frame': int(frame),
                    'Class': class_name,
                    'VehicleID': f"{class_name}_{track_id}",
                    'X_pixel': float(x),
                    'Y_pixel': float(y),
                    'Time': float(frame / 25.0),  # Assuming 25 fps
                    'X_world': 0.0,  # Not calibrated
                    'Y_world': 0.0
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        
        # Generate output filename
        base_dir = Path("/mnt/c/Users/srini/Downloads")
        counter = 1
        while (base_dir / f"optical_trajectories_{counter:03d}.csv").exists():
            counter += 1
        
        output_path = base_dir / f"optical_trajectories_{counter:03d}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"   📄 Saved: {output_path}")
        print(f"   Points: {len(df)}")
        
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Simple Optical Flow Vehicle Tracker")
    parser.add_argument('--video', default="/mnt/c/Users/srini/Downloads/D2F1_stab.mp4",
                       help="Path to video file")
    parser.add_argument('--start-frame', type=int, default=4500,
                       help="Starting frame")
    parser.add_argument('--frames', type=int, default=400,
                       help="Number of frames to process")
    parser.add_argument('--confidence', type=float, default=0.4,
                       help="Detection confidence threshold")
    parser.add_argument('--redetect', type=int, default=10,
                       help="Re-detect every N frames (lower=more accurate, higher=faster)")
    parser.add_argument('--min-length', type=int, default=30,
                       help="Minimum trajectory length")
    parser.add_argument('--roi', default='road', choices=['full', 'road'],
                       help="ROI selection")
    
    args = parser.parse_args()
    
    # ROI polygon (road region)
    if args.roi == 'road':
        roi_polygon = [(4, 873), (3827, 877), (3831, 1086), (4, 1071)]
    else:
        roi_polygon = None
    
    # Create tracker
    tracker = OpticalFlowTracker(
        video_path=args.video,
        confidence=args.confidence,
        redetect_interval=args.redetect,
        min_trajectory_length=args.min_length
    )
    
    # Track video
    output_csv = tracker.track_video(
        start_frame=args.start_frame,
        num_frames=args.frames,
        roi_polygon=roi_polygon
    )
    
    print(f"\n{'='*70}")
    print("✅ DONE! Optical flow tracking complete.")
    print(f"{'='*70}")
    print(f"\n📁 Output: {output_csv}")
    print("\n💡 Compare this with ground truth using:")
    print(f"   python compare_trajectories.py {output_csv} <ground_truth.csv>")
    

if __name__ == "__main__":
    main()
