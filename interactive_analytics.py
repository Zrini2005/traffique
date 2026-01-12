"""
Interactive Vehicle Analytics System
Provides real-time vehicle detection and tracking for traffic analysis
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

class PolygonROI:
    """Handles polygon region of interest operations"""
    
    @staticmethod
    def point_in_polygon(point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    @staticmethod
    def bbox_in_polygon(bbox, polygon):
        """Check if a bounding box center is inside a polygon"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return PolygonROI.point_in_polygon((center_x, center_y), polygon)


class CoordinateMapper:
    """Handles coordinate transformations for real-world measurements"""
    
    def __init__(self):
        self.H = None  # Homography matrix
        self.calibrated = False
    
    def pixel_to_real(self, pixel_coords):
        """Convert pixel coordinates to real-world coordinates"""
        if not self.calibrated or self.H is None:
            return pixel_coords
        
        point = np.array([[[pixel_coords[0], pixel_coords[1]]]], dtype=np.float32)
        real_point = cv2.perspectiveTransform(point, self.H)
        return tuple(real_point[0][0])


class VehicleTracker:
    """Simple IoU-based vehicle tracker"""
    
    def __init__(self, max_age=30, min_iou=0.3):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.min_iou = min_iou
    
    def update(self, detections):
        """Update tracks with new detections"""
        # Simple IoU matching
        matched_tracks = {}
        
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track['age'] < self.max_age:
                    iou = self._calculate_iou(det['bbox'], track['bbox'])
                    if iou > best_iou and iou > self.min_iou:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id:
                matched_tracks[best_track_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'age': 0
                }
            else:
                # New track
                matched_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'age': 0
                }
                self.next_id += 1
        
        # Age unmatched tracks
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track['age'] += 1
                if track['age'] < self.max_age:
                    matched_tracks[track_id] = track
        
        self.tracks = matched_tracks
        return self.tracks
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class VehicleAnalyzer:
    """Main vehicle detection and analysis system"""
    
    def __init__(self, model_conf=0.20, use_sahi=False, sahi_slice_size=640):
        """
        Initialize the analyzer
        
        Args:
            model_conf: Confidence threshold for detections (default 0.20)
            use_sahi: Whether to use SAHI (Slicing Aided Hyper Inference)
            sahi_slice_size: Size of slices for SAHI (default 640)
        """
        self.model_conf = model_conf
        self.use_sahi = use_sahi
        self.sahi_slice_size = sahi_slice_size
        self.model = None
        self.polygon_roi = PolygonROI()
        self.coord_mapper = CoordinateMapper()
        self.tracker = VehicleTracker()
        
        # Vehicle class mapping for VisDrone/Traffic dataset
        self.class_names = {
            0: 'pedestrian',
            1: 'people',
            2: 'bicycle',
            3: 'car',
            4: 'van',
            5: 'truck',
            6: 'tricycle',
            7: 'awning-tricycle',
            8: 'bus',
            9: 'motor'
        }
        
        # Vehicle classes to detect (excluding pedestrians)
        self.vehicle_classes = [2, 3, 4, 5, 6, 7, 8, 9]
    
    def load_model(self):
        """Load the YOLOv8 VisDrone model"""
        print("Loading VisDrone model...")
        
        # Use the specific VisDrone-trained model that actually works
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading VisDrone model from HuggingFace...")
            
            # This is the actual working VisDrone model repo
            model_path = hf_hub_download(
                repo_id="mshamrai/yolov8s-visdrone",
                filename="best.pt"  # Changed from yolov8s.pt to best.pt
            )
            self.model = YOLO(model_path)
            print("VisDrone model loaded successfully!")
        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            print("Trying alternative VisDrone repo...")
            
            try:
                model_path = hf_hub_download(
                    repo_id="keremberke/yolov8m-vehicles-detection",
                    filename="best.pt"
                )
                self.model = YOLO(model_path)
                print("Alternative VisDrone model loaded!")
            except Exception as e2:
                print(f"Alternative also failed: {e2}")
                print(f"Alternative also failed: {e2}")
                print("Falling back to standard YOLOv8s (will auto-download if needed)...")
                try:
                    self.model = YOLO('yolov8s.pt')
                    print("⚠️  WARNING: Using COCO model (yolov8s.pt), not VisDrone! Detections will be standard.")
                except Exception as e3:
                    raise Exception(f"All model loading approaches failed. Error: {e3}")
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        
        sahi_status = "with SAHI" if self.use_sahi else "without SAHI"
        print(f"Model loaded {sahi_status}! Device: {device}")
        
    def _detect_vehicles(self, frame):
        """
        Detect vehicles in a single frame
        
        Args:
            frame: OpenCV image (BGR format)
            
        Returns:
            List of detections with bbox, confidence, class_name
        """
        if self.model is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        detections = []
        
        if self.use_sahi:
            # SAHI mode: Slice and detect
            detections = self._detect_with_sahi(frame)
        else:
            # Standard mode: Detect on full frame
            results = self.model.predict(
                frame,
                conf=self.model_conf,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    cls = int(box.cls[0])
                    
                    # Only include vehicle classes
                    if cls in self.vehicle_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class_name': self.class_names.get(cls, 'unknown'),
                            'class_id': cls
                        })
        
        return detections
    
    def _detect_with_sahi(self, frame):
        """Detect vehicles using SAHI (Slicing Aided Hyper Inference)"""
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        # Wrap YOLO model for SAHI
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model=self.model,
            confidence_threshold=self.model_conf,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Perform sliced inference with reduced overlap for speed
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=0.1,  # Reduced from 0.2 for speed
            overlap_width_ratio=0.1,   # Reduced from 0.2 for speed
            verbose=0
        )
        
        detections = []
        for obj in result.object_prediction_list:
            cls = int(obj.category.id)
            
            # Only include vehicle classes
            if cls in self.vehicle_classes:
                bbox = obj.bbox.to_xyxy()
                
                detections.append({
                    'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    'confidence': float(obj.score.value),
                    'class_name': self.class_names.get(cls, 'unknown'),
                    'class_id': cls
                })
        
        return detections
    
    def analyze_frame(self, frame, polygon_points):
        """
        Analyze a single frame with polygon ROI
        
        Args:
            frame: OpenCV image (BGR format)
            polygon_points: List of (x, y) tuples defining the ROI
            
        Returns:
            Dictionary with detections and statistics
        """
        # Detect all vehicles
        all_detections = self._detect_vehicles(frame)
        
        # Filter by polygon
        filtered_detections = [
            d for d in all_detections 
            if self.polygon_roi.bbox_in_polygon(d['bbox'], polygon_points)
        ]
        
        return {
            'total_detections': len(all_detections),
            'roi_detections': len(filtered_detections),
            'all_vehicles': all_detections,
            'roi_vehicles': filtered_detections
        }
    
    def analytics_mode(self, video_path, frame_idx=0, time_window=5, calibrate=False, polygon_points=None):
        """
        Run full analytics mode with tracking over a time window
        
        Args:
            video_path: Path to video file
            frame_idx: Starting frame index
            time_window: Number of seconds to analyze
            calibrate: Whether to use calibration
            polygon_points: List of (x, y) tuples for ROI filtering
            
        Returns:
            List of vehicle analytics with trajectories, velocity, etc.
        """
        import csv
        from pathlib import Path
        
        # Reset tracker for new analysis
        self.tracker = VehicleTracker()
        
        # Ensure model is loaded
        if not hasattr(self, 'model') or self.model is None:
            print("Model not loaded, loading now...")
            self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = max(0, frame_idx - int(fps * time_window / 2))
        end_frame = min(total_frames - 1, frame_idx + int(fps * time_window / 2))
        num_frames = end_frame - start_frame
        
        print(f"ANALYTICS MODE - Full Tracking")
        print(f"Video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        print(f"Frame range: {start_frame} to {end_frame} ({num_frames} frames)")
        print(f"Polygon filtering: {'Yes' if polygon_points else 'No'}")
        print(f"Processing {num_frames} frames...")
        
        # Track vehicles across frames
        vehicle_trajectories = {}
        
        for frame_num in range(start_frame, end_frame):
            if frame_num % 50 == 0:
                print(f"  Processing frame {frame_num - start_frame + 1}/{num_frames}...")
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect vehicles
            detections = self._detect_vehicles(frame)
            
            # Filter by polygon if provided
            if polygon_points:
                detections = [d for d in detections if self.polygon_roi.bbox_in_polygon(d['bbox'], polygon_points)]
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Store trajectories
            for track_id, track in tracks.items():
                if track_id not in vehicle_trajectories:
                    vehicle_trajectories[track_id] = {
                        'id': track_id,
                        'class': track['class_name'],
                        'frames': [],
                        'positions': [],
                        'confidences': [],
                        'timestamps': []
                    }
                
                # Calculate center
                bbox = track['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                vehicle_trajectories[track_id]['frames'].append(frame_num)
                vehicle_trajectories[track_id]['positions'].append((center_x, center_y))
                vehicle_trajectories[track_id]['confidences'].append(track['confidence'])
                vehicle_trajectories[track_id]['timestamps'].append(frame_num / fps)
        
        cap.release()
        
        # Calculate analytics with velocity
        analytics = []
        csv_data = []
        
        for track_id, trajectory in vehicle_trajectories.items():
            if len(trajectory['frames']) >= 3:  # Minimum 3 frames
                # Calculate velocity (pixels per second)
                positions = trajectory['positions']
                timestamps = trajectory['timestamps']
                
                velocities = []
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    dt = timestamps[i] - timestamps[i-1]
                    
                    if dt > 0:
                        velocity = np.sqrt(dx**2 + dy**2) / dt  # pixels/second
                        velocities.append(velocity)
                
                avg_velocity = np.mean(velocities) if velocities else 0
                max_velocity = np.max(velocities) if velocities else 0
                
                # Total distance traveled
                total_distance = 0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    total_distance += np.sqrt(dx**2 + dy**2)
                
                # Time in scene
                time_in_scene = timestamps[-1] - timestamps[0]
                
                vehicle_data = {
                    'vehicle_id': track_id,
                    'class': trajectory['class'],
                    'first_frame': trajectory['frames'][0],
                    'last_frame': trajectory['frames'][-1],
                    'num_frames': len(trajectory['frames']),
                    'time_in_scene': round(time_in_scene, 2),
                    'avg_confidence': round(sum(trajectory['confidences']) / len(trajectory['confidences']), 3),
                    'trajectory_length': len(trajectory['positions']),
                    'start_position': trajectory['positions'][0],
                    'end_position': trajectory['positions'][-1],
                    'avg_velocity_px_per_sec': round(avg_velocity, 2),
                    'max_velocity_px_per_sec': round(max_velocity, 2),
                    'total_distance_px': round(total_distance, 2)
                }
                
                analytics.append(vehicle_data)
                
                # CSV row
                csv_data.append({
                    'vehicle_id': track_id,
                    'class': trajectory['class'],
                    'first_frame': trajectory['frames'][0],
                    'last_frame': trajectory['frames'][-1],
                    'num_frames': len(trajectory['frames']),
                    'time_in_scene_sec': round(time_in_scene, 2),
                    'avg_confidence': round(sum(trajectory['confidences']) / len(trajectory['confidences']), 3),
                    'start_x': trajectory['positions'][0][0],
                    'start_y': trajectory['positions'][0][1],
                    'end_x': trajectory['positions'][-1][0],
                    'end_y': trajectory['positions'][-1][1],
                    'avg_velocity_px_per_sec': round(avg_velocity, 2),
                    'max_velocity_px_per_sec': round(max_velocity, 2),
                    'total_distance_px': round(total_distance, 2),
                    'trajectory_points': len(trajectory['positions'])
                })
        
        # Save to CSV
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / 'vehicle_analytics.csv'
        
        if csv_data:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"✅ Saved analytics to: {csv_path}")
        
        print(f"Tracked {len(analytics)} vehicles with full trajectories")
        
        # Generate annotated visualization for the target frame
        annotated_frame = None
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            annotated_frame = frame.copy()
            
            # Draw polygon if provided
            if polygon_points:
                pts = np.array(polygon_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [pts], (23, 77, 56))  # #174D38
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
                cv2.polylines(annotated_frame, [pts], True, (23, 77, 56), 2)
            
            # Draw trajectories for each vehicle
            for track_id, trajectory in vehicle_trajectories.items():
                if len(trajectory['frames']) < 3:
                    continue
                    
                # Check if this track exists at the target frame
                if frame_idx not in trajectory['frames']:
                    continue
                
                # Get position at target frame
                frame_index = trajectory['frames'].index(frame_idx)
                current_pos = trajectory['positions'][frame_index]
                
                # Draw trajectory path
                for i in range(1, len(trajectory['positions'])):
                    pt1 = trajectory['positions'][i-1]
                    pt2 = trajectory['positions'][i]
                    cv2.line(annotated_frame, pt1, pt2, (255, 165, 0), 2)  # Orange trajectory
                
                # Draw current position marker
                cv2.circle(annotated_frame, current_pos, 8, (23, 77, 56), -1)  # Green center
                cv2.circle(annotated_frame, current_pos, 12, (255, 255, 255), 2)  # White ring
                
                # Draw vehicle ID
                label = f"#{track_id} {trajectory['class']}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, 
                            (current_pos[0] - w//2 - 5, current_pos[1] - 35),
                            (current_pos[0] + w//2 + 5, current_pos[1] - 15),
                            (23, 77, 56), -1)
                cv2.putText(annotated_frame, label,
                           (current_pos[0] - w//2, current_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add info panel
            info_panel_height = 100
            info_panel = np.ones((info_panel_height, annotated_frame.shape[1], 3), dtype=np.uint8) * 240
            
            # Add statistics
            stats_text = [
                f"Full Analytics Mode | Frame: {frame_idx}",
                f"Total Tracks: {len(analytics)} | Time Window: {time_window}s",
                f"Frame Range: {start_frame} - {end_frame}"
            ]
            
            y_offset = 25
            for text in stats_text:
                cv2.putText(info_panel, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
                y_offset += 30
            
            annotated_frame = np.vstack([annotated_frame, info_panel])
        
        return analytics, annotated_frame
