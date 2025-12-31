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
    """Enhanced IoU-based vehicle tracker with ID persistence priority"""
    
    def __init__(self, max_age=60, min_iou=0.20):  # ID PERSISTENCE: Higher max_age, lower IoU
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age  # MEDIUM - keeps tracks alive to prevent ID switching
        self.min_iou = min_iou  # LOW - very permissive to maintain same ID
        # Track history for motion prediction
        self.track_history = {}  # track_id -> list of (frame, center, bbox)
        self.frame_count = 0
    
    def update(self, detections):
        """Update tracks with new detections using IoU + spatial proximity + motion"""
        self.frame_count += 1
        matched_tracks = {}
        matched_detections = set()
        
        # Debug counters
        iou_matches = 0
        proximity_matches = 0
        new_tracks = 0
        
        # Phase 1: IoU-based matching (primary method)
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            # Class-specific IoU thresholds (more permissive for small vehicles)
            class_name = det.get('class_name', '')
            if class_name in ['bicycle', 'tricycle', 'motor', 'motorcycle']:
                min_iou_threshold = 0.15  # Very permissive for 2-wheelers
            elif class_name in ['truck', 'bus']:
                min_iou_threshold = 0.25  # Slightly higher for large vehicles
            else:
                min_iou_threshold = self.min_iou  # Default (0.20)
            
            for track_id, track in self.tracks.items():
                if track['age'] < self.max_age:
                    iou = self._calculate_iou(det['bbox'], track['bbox'])
                    if iou > best_iou and iou > min_iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id:
                iou_matches += 1
                det_center = self._get_center(det['bbox'])
                matched_tracks[best_track_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'age': 0,
                    'center': det_center
                }
                matched_detections.add(det_idx)
                
                # Update history
                if best_track_id not in self.track_history:
                    self.track_history[best_track_id] = []
                self.track_history[best_track_id].append((self.frame_count, det_center, det['bbox']))
                # Keep only last 10 frames
                if len(self.track_history[best_track_id]) > 10:
                    self.track_history[best_track_id].pop(0)
        
        # Phase 2: Spatial proximity + motion prediction for unmatched detections
        unmatched_dets = [det for idx, det in enumerate(detections) if idx not in matched_detections]
        
        for det in unmatched_dets:
            det_center = self._get_center(det['bbox'])
            best_distance = float('inf')
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue  # Already matched
                
                if track['age'] < self.max_age:
                    # Predict next position based on motion history
                    predicted_center = self._predict_position(track_id)
                    
                    # Calculate distance to predicted position
                    distance = np.sqrt((det_center[0] - predicted_center[0])**2 + 
                                     (det_center[1] - predicted_center[1])**2)
                    
                    # Get expected velocity and current track info
                    expected_velocity = self._get_expected_velocity(track_id)
                    last_center = self.track_history[track_id][-1][1] if track_id in self.track_history else predicted_center
                    
                    # CRITICAL: Direction and velocity consistency for congested traffic
                    # Prevent ID mixing between nearby vehicles
                    
                    # Calculate detection's implied velocity from last known position
                    if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                        last_two = self.track_history[track_id][-2:]
                        track_velocity = (
                            (last_two[1][1][0] - last_two[0][1][0]) / max(last_two[1][0] - last_two[0][0], 1),
                            (last_two[1][1][1] - last_two[0][1][1]) / max(last_two[1][0] - last_two[0][0], 1)
                        )
                        
                        # Calculate detection velocity from last position
                        det_velocity = (
                            det_center[0] - last_center[0],
                            det_center[1] - last_center[1]
                        )
                        
                        # Check velocity similarity (cosine similarity)
                        track_speed = np.sqrt(track_velocity[0]**2 + track_velocity[1]**2)
                        det_speed = np.sqrt(det_velocity[0]**2 + det_velocity[1]**2)
                        
                        if track_speed > 1 and det_speed > 1:
                            # Normalize and compare direction
                            track_dir = (track_velocity[0] / track_speed, track_velocity[1] / track_speed)
                            det_dir = (det_velocity[0] / det_speed, det_velocity[1] / det_speed)
                            
                            # Dot product gives direction similarity
                            direction_similarity = track_dir[0] * det_dir[0] + track_dir[1] * det_dir[1]
                            
                            # Reject if moving in significantly different direction
                            if direction_similarity < 0.5:  # Less than 60 degrees difference
                                continue
                    
                    # RELAXED anti-jump checks - allow more natural motion
                    # Was too strict and caused ID switching
                    
                    # Calculate actual movement from last position
                    if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                        last_center = self.track_history[track_id][-1][1]
                        dx = det_center[0] - last_center[0]
                        dy = det_center[1] - last_center[1]
                        
                        # Get expected velocity
                        expected_velocity = self._get_expected_velocity(track_id)
                        
                        # RULE 1: Reasonable forward movement (balanced for congested traffic)
                        if expected_velocity > 0:  # Vehicle is moving forward
                            # Max allowed: 2.5x current velocity (prevents jumping to vehicle ahead)
                            max_forward_jump = max(expected_velocity * 2.5, 50)  # Conservative
                            if dx > max_forward_jump:
                                # Likely jumped to vehicle ahead - reject!
                                continue
                        
                        # RULE 2: Reasonable frame-to-frame movement
                        if abs(dx) > 70:  # Balanced - prevents ID jumping in congestion
                            continue
                    
                    # RULE 3: Class-specific distance threshold
                    # 2-wheelers move faster and more erratically
                    if det['class_name'] in ['bicycle', 'tricycle', 'motor', 'motorcycle']:
                        max_reasonable_movement = 80  # More permissive for 2-wheelers
                    else:
                        max_reasonable_movement = 50  # Standard for cars/trucks
                    
                    if distance > max_reasonable_movement:
                        continue
                    
                    # Check class similarity
                    class_match = (det['class_name'] == track['class_name'] or
                                 self._similar_class(det['class_name'], track['class_name']))
                    
                    # Check bounding box size consistency (prevent car/truck confusion)
                    det_area = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
                    track_area = (track['bbox'][2] - track['bbox'][0]) * (track['bbox'][3] - track['bbox'][1])
                    area_ratio = min(det_area, track_area) / max(det_area, track_area)
                    
                    # Reject if size too different (likely different vehicle)
                    if area_ratio < 0.5:  # Size must be within 2x
                        continue
                    
                    # BALANCED: Accept close proximity with consistency checks
                    # Balance ID persistence with avoiding ID mixing
                    if distance < 50 and class_match and distance < best_distance:  # Tighter for congestion
                        best_distance = distance
                        best_track_id = track_id
            
            if best_track_id:
                proximity_matches += 1
                matched_tracks[best_track_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'age': 0,
                    'center': det_center
                }
                
                # Update history
                if best_track_id not in self.track_history:
                    self.track_history[best_track_id] = []
                self.track_history[best_track_id].append((self.frame_count, det_center, det['bbox']))
                if len(self.track_history[best_track_id]) > 10:
                    self.track_history[best_track_id].pop(0)
            else:
                # New track
                new_tracks += 1
                matched_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class_name': det['class_name'],
                    'age': 0,
                    'center': det_center
                }
                self.track_history[self.next_id] = [(self.frame_count, det_center, det['bbox'])]
                self.next_id += 1
        
        # Age unmatched tracks
        aged_tracks = 0
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track['age'] += 1
                aged_tracks += 1
                if track['age'] < self.max_age:
                    matched_tracks[track_id] = track
        
        # Debug output
        if new_tracks > 0 or aged_tracks > 3:
            print(f"      Tracker: {len(detections)} dets → IoU:{iou_matches} Prox:{proximity_matches} "
                  f"New:{new_tracks} | Aged:{aged_tracks} tracks")
        
        self.tracks = matched_tracks
        return self.tracks
    
    def _get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _predict_position(self, track_id):
        """Predict next position based on motion history"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            # No history, use current position
            if track_id in self.tracks:
                return self.tracks[track_id].get('center', (0, 0))
            return (0, 0)
        
        history = self.track_history[track_id]
        
        # Use last 3 frames for velocity estimation
        recent = history[-min(3, len(history)):]
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent)):
            frame_diff = recent[i][0] - recent[i-1][0]
            if frame_diff > 0:
                vx = (recent[i][1][0] - recent[i-1][1][0]) / frame_diff
                vy = (recent[i][1][1] - recent[i-1][1][1]) / frame_diff
                velocities.append((vx, vy))
        
        if velocities:
            avg_vx = np.mean([v[0] for v in velocities])
            avg_vy = np.mean([v[1] for v in velocities])
            
            # Predict forward by age+1 frames
            age = self.tracks.get(track_id, {}).get('age', 0)
            
            # Apply moderate velocity damping - balance between following vehicle and preventing overshoot
            # Gradual reduction as track ages (less confidence in old velocity)
            if age <= 5:
                damping = 0.8  # Recent detection, trust velocity
            elif age <= 12:
                damping = 0.6  # Medium age, moderate damping
            else:
                damping = 0.4  # Old detection, strong damping (likely slowing down)
            
            last_center = history[-1][1]
            predicted_x = last_center[0] + avg_vx * damping * (age + 1)
            predicted_y = last_center[1] + avg_vy * damping * (age + 1)
            
            return (predicted_x, predicted_y)
        
        # Fallback to last known position
        return history[-1][1]
    
    def _get_expected_velocity(self, track_id):
        """Get expected velocity magnitude for a track"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return 0
        
        history = self.track_history[track_id]
        recent = history[-min(3, len(history)):]
        
        velocities = []
        for i in range(1, len(recent)):
            frame_diff = recent[i][0] - recent[i-1][0]
            if frame_diff > 0:
                vx = (recent[i][1][0] - recent[i-1][1][0]) / frame_diff
                vy = (recent[i][1][1] - recent[i-1][1][1]) / frame_diff
                velocities.append(np.sqrt(vx**2 + vy**2))
        
        return np.mean(velocities) if velocities else 0
    
    def _similar_class(self, class1, class2):
        """Check if two vehicle classes are similar enough to be the same vehicle"""
        # Group similar classes
        similar_groups = [
            {'car', 'van', 'truck', 'bus'},  # All vehicles
            {'motor', 'bicycle', 'bike'},     # Two-wheelers
        ]
        
        for group in similar_groups:
            if class1.lower() in group and class2.lower() in group:
                return True
        
        return False
    
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
                print("Falling back to local yolov8s.pt (COCO-trained)...")
                local_model = Path(__file__).parent / 'yolov8s.pt'
                if local_model.exists():
                    self.model = YOLO(str(local_model))
                    print("⚠️  WARNING: Using COCO model, not VisDrone! Detections will be limited.")
                else:
                    raise Exception("No model available. Please check model files.")
        
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
                        
                        # Quality filter: reject suspiciously small or malformed detections
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # STRICT quality filters for precision:
                        # - Min area: 600 pixels (larger vehicles only)
                        # - Max aspect ratio: 4.0 (tighter bounds)
                        # - Min aspect ratio: 0.3 (reject extreme shapes)
                        if area < 600 or aspect_ratio > 4.0 or aspect_ratio < 0.3:
                            continue
                        
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
        
        # Perform sliced inference with MAXIMUM overlap for best precision
        # Maximum overlap ensures consistent high-quality detections
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=0.4,  # Maximum overlap for precision
            overlap_width_ratio=0.4,   # Maximum overlap for precision
            verbose=0
        )
        
        detections = []
        for obj in result.object_prediction_list:
            cls = int(obj.category.id)
            
            # Only include vehicle classes
            if cls in self.vehicle_classes:
                bbox = obj.bbox.to_xyxy()
                
                # Quality filter: reject suspiciously small or malformed detections
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                # STRICT quality filters for precision:
                # - Min area: 600 pixels (larger vehicles only)
                # - Max aspect ratio: 4.0 (tighter bounds)
                # - Min aspect ratio: 0.3 (reject extreme shapes)
                if area < 600 or aspect_ratio > 4.0 or aspect_ratio < 0.3:
                    continue
                
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
