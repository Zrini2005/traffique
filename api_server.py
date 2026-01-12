"""
Flask API Backend for Interactive Vehicle Analytics
Connects the Python analytics system to the React frontend
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from pathlib import Path
import json
import base64
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Import our analytics system
import sys
# Ensure repository root is first on sys.path so local packages (utils, etc.) are importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from interactive_analytics import VehicleAnalyzer
# New tracking/reid utilities
from utils.reid import ReIDExtractor
from utils.onlinetracker import OnlineTracker
from utils.trajectory import smooth_kalman, median_filter, linear_interpolate
# Import the tracking method from generate_lowconf_highres
from generate_lowconf_highres import generate_tracks_lowconf_highres

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 2GB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global analyzer instance
analyzer = None

# Cache for tracking results to avoid recalculation
# Key: (filename, start_frame, end_frame, confidence)
# Value: (tracks_data, timestamp)
tracking_cache = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'})


@app.route('/api/local-videos', methods=['GET'])
def list_local_videos():
    """List videos from local directory"""
    local_video_dir = Path(r'C:\Users\sakth\Documents\traffique_footage')
    
    if not local_video_dir.exists():
        return jsonify({'videos': []})
    
    videos = []
    for video_path in local_video_dir.glob('*.mp4'):
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            videos.append({
                'filename': video_path.name,
                'path': str(video_path),
                'size_mb': video_path.stat().st_size / (1024 * 1024),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            })
            cap.release()
    
    return jsonify({'videos': videos})


@app.route('/api/use-local/<filename>', methods=['POST'])
def use_local_video(filename):
    """Use a local video file without uploading"""
    local_video_dir = Path(r'C:\Users\sakth\Documents\traffique_footage')
    filepath = local_video_dir / filename
    
    if not filepath.exists():
        return jsonify({'error': 'Video not found'}), 404
    
    # Create symlink in uploads folder to avoid copying
    upload_path = UPLOAD_FOLDER / filename
    if not upload_path.exists():
        try:
            # For Windows, just store the path reference
            with open(UPLOAD_FOLDER / f"{filename}.path", 'w') as f:
                f.write(str(filepath))
        except:
            pass
    
    # Get metadata
    cap = cv2.VideoCapture(str(filepath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return jsonify({
        'success': True,
        'filename': filename,
        'metadata': {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration_seconds': total_frames / fps
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = UPLOAD_FOLDER / filename
            
            # Stream save for large files
            print(f"Uploading {filename}...")
            file.save(filepath)
            print(f"Upload complete: {filepath}")
            
            # Get video metadata
            cap = cv2.VideoCapture(str(filepath))
            if not cap.isOpened():
                return jsonify({'error': 'Failed to open video file'}), 400
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"Video info: {total_frames} frames, {fps} fps, {width}x{height}")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'metadata': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration_seconds': total_frames / fps if fps > 0 else 0
                }
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/frame/<filename>/<int:frame_idx>', methods=['GET'])
def get_frame(filename, frame_idx):
    """Get specific frame from video as base64 image"""
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    
    # Check if it's a local video reference
    path_file = UPLOAD_FOLDER / f"{secure_filename(filename)}.path"
    if path_file.exists():
        with open(path_file, 'r') as f:
            filepath = Path(f.read().strip())
    
    if not filepath.exists():
        return jsonify({'error': 'Video not found'}), 404
    
    cap = cv2.VideoCapture(str(filepath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': 'Failed to read frame'}), 400
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'frame_idx': frame_idx,
        'frame': img_base64  # Return just the base64 string, frontend will add the data URI prefix
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Generate tracks CSV using generate_lowconf_highres.py for FULL video"""
    import pandas as pd
    import ast
    
    data = request.json
    filename = data.get('filename')
    frame_idx = data.get('frame_idx', 0)
    polygon = data.get('polygon') or []
    confidence = data.get('confidence', 0.15)

    if not filename:
        return jsonify({'error': 'Missing filename'}), 400

    filepath = UPLOAD_FOLDER / secure_filename(filename)
    path_file = UPLOAD_FOLDER / f"{secure_filename(filename)}.path"
    if path_file.exists():
        with open(path_file, 'r') as f:
            filepath = Path(f.read().strip())

    if not filepath.exists():
        return jsonify({'error': 'Video not found'}), 404

    try:
        # Setup paths
        model_path = r"C:\Users\srini\Downloads\visdrone_finetuned.pt"
        if not Path(model_path).exists():
            model_path = "yolov8n.pt"
        
        print(f"🚀 Processing FULL VIDEO with LowConf({confidence}) + HighRes(1920)")
        
        # Get video info
        cap = cv2.VideoCapture(str(filepath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        # Setup output
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / f'vehicle_tracks_{Path(filename).stem}.csv'
        
        print(f"   Total frames: {total_frames}")
        
        # Run full video tracking using generate_lowconf_highres
        generate_tracks_lowconf_highres(
            video_path=str(filepath),
            model_path=model_path,
            output_path=str(csv_path),
            conf_thresh=confidence,
            img_size=1920,
            frames_to_process=total_frames
        )
        
        # Load the CSV
        if not csv_path.exists():
            return jsonify({'error': 'Tracking failed - no output generated'}), 500
        
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} detections from full video")
        
        # Apply polygon filtering if provided
        if polygon:
            polygon_pts = np.array([(int(p['x']), int(p['y'])) for p in polygon], dtype=np.int32)
            
            def is_in_polygon(row):
                cx, cy = row['X_pixel'], row['Y_pixel']
                return cv2.pointPolygonTest(polygon_pts, (int(cx), int(cy)), False) >= 0
            
            df = df[df.apply(is_in_polygon, axis=1)]
            csv_path_filtered = output_dir / f'vehicle_tracks_{Path(filename).stem}_filtered.csv'
            df.to_csv(csv_path_filtered, index=False)
            csv_path = csv_path_filtered
            print(f"   After polygon filter: {len(df)} detections")
        
        # Add BBox column if not present
        if 'BBox' not in df.columns:
            df['BBox'] = df.apply(lambda row: [
                float(row['X_pixel'] - 25),
                float(row['Y_pixel'] - 25),
                float(row['X_pixel'] + 25),
                float(row['Y_pixel'] + 25)
            ], axis=1)
            df.to_csv(csv_path, index=False)
        
        if df.empty:
            return jsonify({'error': 'No vehicles detected in the specified region'}), 400
        if df.empty:
            return jsonify({'error': 'No vehicles detected in the specified region'}), 400
        
        # Get vehicles on selected frame for display
        frame_tracks = df[df['Frame'] == frame_idx]
        vehicles_on_frame = []
        for _, row in frame_tracks.iterrows():
            bbox = row['BBox']
            if isinstance(bbox, str):
                bbox = ast.literal_eval(bbox)
            vehicles_on_frame.append({
                'vehicle_id': int(row['VehicleID']),
                'bbox': bbox,
                'class': int(row['Class'])
            })
        
        # Draw frame with GREEN BOXES
        cap = cv2.VideoCapture(str(filepath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            vis = frame.copy()
            
            # Draw polygon if provided
            if polygon:
                polygon_pts = np.array([(int(p['x']), int(p['y'])) for p in polygon], dtype=np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [polygon_pts], (23, 77, 56))
                cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                cv2.polylines(vis, [polygon_pts], True, (23, 77, 56), 3)
            
            # Draw GREEN BOXES around vehicles
            for v in vehicles_on_frame:
                x1, y1, x2, y2 = [int(x) for x in v['bbox']]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                label = f"ID:{v['vehicle_id']}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - 25), (x1 + 70, y1), (0, 255, 0), -1)
                cv2.putText(vis, label, (x1 + 5, y1 - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Info text
            info_text = f"Frame {frame_idx} | Vehicles: {len(vehicles_on_frame)} | Click to see trajectory"
            cv2.rectangle(vis, (10, 10), (800, 50), (0, 0, 0), -1)
            cv2.putText(vis, info_text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            annotated_frame = base64.b64encode(buffer).decode('utf-8')
        else:
            annotated_frame = None
        
        return jsonify({
            'success': True,
            'total_vehicles': df['VehicleID'].nunique(),
            'total_detections': len(df),
            'frames_processed': total_frames,
            'vehicles_on_frame': vehicles_on_frame,
            'annotated_frame': annotated_frame,
            'csv_file': str(csv_path)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/vehicle-trajectory', methods=['POST'])
def vehicle_trajectory():
    """Generate trajectory overlay for a specific vehicle - like overlay_frm_csv.py"""
    data = request.json
    filename = data.get('filename')
    vehicle_id = data.get('vehicle_id')
    
    if not filename or vehicle_id is None:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Read the CSV file for this video
    import pandas as pd
    csv_path = None
    for csv_file in sorted(Path('output').glob(f'vehicle_tracks_frame_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True):
        csv_path = csv_file
        break
    
    if not csv_path or not csv_path.exists():
        return jsonify({'error': 'No tracking data found. Please run analysis first.'}), 404
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter for this vehicle
        vehicle_data = df[df['VehicleID'] == vehicle_id].sort_values('Frame')
        
        if vehicle_data.empty:
            return jsonify({'error': 'Vehicle not found'}), 404
        
        # Get the video frame
        filepath = UPLOAD_FOLDER / secure_filename(filename)
        path_file = UPLOAD_FOLDER / f"{secure_filename(filename)}.path"
        if path_file.exists():
            with open(path_file, 'r') as f:
                filepath = Path(f.read().strip())
        
        # Get a frame from the middle of the trajectory
        mid_frame = int(vehicle_data['Frame'].median())
        
        cap = cv2.VideoCapture(str(filepath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Failed to read frame'}), 400
        
        # Draw trajectory on frame - EXACTLY like overlay_frm_csv.py
        vis = frame.copy()
        
        # Extract trajectory points
        trajectory = [(int(row['X_pixel']), int(row['Y_pixel'])) for _, row in vehicle_data.iterrows()]
        
        # Choose a color (green for this vehicle)
        color = (0, 255, 0)  # Green
        
        # Draw trajectory lines (like overlay_frm_csv.py line 95)
        for i in range(1, len(trajectory)):
            cv2.line(vis, trajectory[i-1], trajectory[i], color, 3, cv2.LINE_AA)
        
        # Draw start point (bigger green circle)
        start_pt = trajectory[0]
        cv2.circle(vis, start_pt, 10, (0, 255, 0), -1)
        cv2.circle(vis, start_pt, 12, (255, 255, 255), 2)  # White border
        cv2.putText(vis, "START", (start_pt[0] - 40, start_pt[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
        # Draw end point (bigger red circle)
        end_pt = trajectory[-1]
        cv2.circle(vis, end_pt, 10, (0, 0, 255), -1)
        cv2.circle(vis, end_pt, 12, (255, 255, 255), 2)  # White border
        cv2.putText(vis, "END", (end_pt[0] - 30, end_pt[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Draw current position (larger circle at middle frame)
        for idx, (frame_num, x, y) in enumerate(zip(vehicle_data['Frame'], 
                                                      vehicle_data['X_pixel'], 
                                                      vehicle_data['Y_pixel'])):
            if int(frame_num) == mid_frame:
                current_pos = (int(x), int(y))
                cv2.circle(vis, current_pos, 8, color, -1)
                cv2.circle(vis, current_pos, 10, (255, 255, 255), 2)
                break
        
        # Add info panel at top (like overlay_frm_csv.py line 118)
        info_bg = vis.copy()
        cv2.rectangle(info_bg, (10, 10), (850, 80), (0, 0, 0), -1)
        cv2.addWeighted(info_bg, 0.7, vis, 0.3, 0, vis)
        
        cv2.putText(vis, f"Vehicle ID: {vehicle_id} - Complete Trajectory", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(vis, f"Frames: {int(vehicle_data['Frame'].min())} to {int(vehicle_data['Frame'].max())} | Points: {len(trajectory)} | Current: {mid_frame}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode
        _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        trajectory_image = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ Generated trajectory for vehicle #{vehicle_id}: {len(trajectory)} points")
        
        return jsonify({
            'success': True,
            'vehicle_id': vehicle_id,
            'trajectory_image': trajectory_image,
            'num_frames': len(vehicle_data),
            'start_frame': int(vehicle_data['Frame'].min()),
            'end_frame': int(vehicle_data['Frame'].max()),
            'trajectory_points': len(trajectory)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/frame-vehicles', methods=['POST'])
def frame_vehicles():
    """Load vehicles from existing CSV for a specific frame - for smooth playback"""
    data = request.json
    filename = data.get('filename')
    frame_idx = data.get('frame_idx')
    
    if not filename or frame_idx is None:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Read the most recent CSV file
    import pandas as pd
    csv_path = None
    for csv_file in sorted(Path('output').glob(f'vehicle_tracks_frame_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True):
        csv_path = csv_file
        break
    
    if not csv_path or not csv_path.exists():
        return jsonify({'success': True, 'vehicles': []})
    
    try:
        df = pd.read_csv(csv_path)
        
        # Get vehicles on this specific frame
        frame_data = df[df['Frame'] == frame_idx]
        
        vehicles = []
        for _, row in frame_data.iterrows():
            vehicles.append({
                'vehicle_id': int(row['VehicleID']),
                'bbox': row['BBox'] if isinstance(row['BBox'], list) else eval(row['BBox']),
                'class': int(row['Class'])
            })
        
        return jsonify({
            'success': True,
            'vehicles': vehicles,
            'frame_idx': frame_idx
        })
        
    except Exception as e:
        print(f"Error loading frame vehicles: {str(e)}")
        return jsonify({'success': True, 'vehicles': []})
    """Quick mode analysis - single frame"""
    global analyzer
    
    data = request.json
    filename = data.get('filename')
    frame_idx = data.get('frame_idx')
    polygon = data.get('polygon')  # List of [x, y] points
    confidence = data.get('confidence', 0.20)
    
    if not all([filename, frame_idx is not None, polygon]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    filepath = UPLOAD_FOLDER / secure_filename(filename)
    
    # Check if it's a local video reference
    path_file = UPLOAD_FOLDER / f"{secure_filename(filename)}.path"
    if path_file.exists():
        with open(path_file, 'r') as f:
            filepath = Path(f.read().strip())
    
    if not filepath.exists():
        return jsonify({'error': 'Video not found'}), 404
    
    try:
        # Initialize analyzer if needed (without SAHI)
        if analyzer is None or analyzer.model_conf != confidence:
            print(f"Initializing analyzer with confidence={confidence}")
            analyzer = VehicleAnalyzer(model_conf=confidence, use_sahi=False, sahi_slice_size=640)
            analyzer.load_model()
            print("Analyzer ready!")
        
        # Extract frame
        cap = cv2.VideoCapture(str(filepath))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Failed to read frame'}), 400
        
        # Detect vehicles
        detections = analyzer._detect_vehicles(frame)
        
        # Filter by polygon
        polygon_points = [(int(p['x']), int(p['y'])) for p in polygon]
        filtered = [d for d in detections if analyzer.polygon_roi.bbox_in_polygon(d['bbox'], polygon_points)]
        
        # Draw visualization
        vis = frame.copy()
        
        # Draw ALL detections first (in gray - outside polygon)
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            is_in_polygon = det in filtered
            
            # Color: Green if in polygon, Gray if outside
            color = (23, 77, 56) if is_in_polygon else (150, 150, 150)  # BGR format
            thickness = 3 if is_in_polygon else 2
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            label = f"#{idx+1} {det['class_name']} {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background rectangle for text
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # White text
            cv2.putText(vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
        
        # Draw polygon with semi-transparent fill
        overlay = vis.copy()
        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (23, 77, 56, 50))  # Semi-transparent green
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        # Draw polygon border (thick)
        cv2.polylines(vis, [pts], True, (23, 77, 56), 4)
        
        # Add info panel at top
        panel_height = 100
        panel = np.zeros((panel_height, vis.shape[1], 3), dtype=np.uint8)
        panel[:] = (242, 242, 242)  # Light gray background
        
        # Add statistics
        cv2.putText(panel, f"Total Detections: {len(detections)}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 77, 56), 2)
        cv2.putText(panel, f"In ROI: {len(filtered)}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (77, 23, 23), 2)
        cv2.putText(panel, f"Confidence: {confidence:.2f}", (400, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 77, 56), 2)
        cv2.putText(panel, f"Frame: {frame_idx}", (400, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 77, 56), 2)
        
        # Combine panel with visualization
        vis = np.vstack([panel, vis])
        
        # Encode visualization
        _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare results with center coordinates
        results = {
            'success': True,
            'total_detections': len(detections),
            'polygon_detections': len(filtered),
            'vehicles': [
                {
                    'id': idx + 1,
                    'bbox': det['bbox'],
                    'confidence': float(det['confidence']),
                    'class': det['class_name'],
                    'center_x': int((det['bbox'][0] + det['bbox'][2]) / 2),
                    'center_y': int((det['bbox'][1] + det['bbox'][3]) / 2),
                    'in_roi': True
                }
                for idx, det in enumerate(filtered)
            ],
            'visualization': vis_base64  # Just base64, no prefix
        }
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/csv', methods=['GET'])
def download_csv():
    """Download analytics CSV"""
    csv_path = Path('output/vehicle_analytics.csv')
    
    if not csv_path.exists():
        return jsonify({'error': 'CSV file not found'}), 404
    
    return send_file(csv_path, as_attachment=True, download_name='vehicle_analytics.csv')




# ============================================================================
# MULTI-CAMERA ENDPOINTS (Advanced Mode)
# ============================================================================

@app.route('/api/calibrate-camera', methods=['POST'])
def calibrate_camera():
    """
    Calculate homography matrix for a camera
    Expects 4 image points and their corresponding real-world coordinates
    """
    try:
        data = request.json
        image_points = np.array(data['image_points'], dtype=np.float32)  # [[x,y], [x,y], [x,y], [x,y]]
        world_points = np.array(data['world_points'], dtype=np.float32)  # [[x,y], [x,y], [x,y], [x,y]] in meters
        
        # Validate input
        if len(image_points) != 4 or len(world_points) != 4:
            return jsonify({'error': 'Exactly 4 points required for calibration'}), 400
        
        # Calculate homography matrix
        H, status = cv2.findHomography(image_points, world_points, method=0)
        
        if H is None:
            return jsonify({'error': 'Failed to calculate homography matrix'}), 500
        
        # Test transformation
        test_point = np.array([[image_points[0]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(test_point, H)[0][0]
        
        return jsonify({
            'success': True,
            'homography_matrix': H.tolist(),
            'test_transform': {
                'image_point': image_points[0].tolist(),
                'world_point': transformed.tolist(),
                'expected': world_points[0].tolist()
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/multi-camera', methods=['POST'])
def analyze_multi_camera():
    """
    Analyze multiple videos with camera fusion
    Expects:
    - cameras: [{video_path, homography_matrix, camera_id}]
    - polygon_points: [[x,y], ...]
    - time_window: seconds
    - use_sahi: boolean
    """
    try:
        data = request.json
        cameras = data.get('cameras', [])
        polygon_points = data.get('polygon_points', [])
        time_window = data.get('time_window', 30)
        frame_idx = data.get('frame_idx', 0)
        use_sahi = data.get('use_sahi', False)
        
        if not cameras:
            return jsonify({'error': 'No cameras provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"🎬 MULTI-CAMERA ANALYSIS")
        print(f"{'='*60}")
        print(f"Cameras: {len(cameras)}")
        print(f"Time window: {time_window}s")
        print(f"SAHI: {use_sahi}")
        
        # Process each camera separately
        all_detections = []
        camera_results = []
        
        for idx, camera in enumerate(cameras):
            print(f"\n📹 Processing Camera {idx + 1}/{len(cameras)}: {camera['camera_id']}")
            
            video_path = camera.get('video_path') or camera.get('local_path')
            if not video_path:
                continue
            
            # Construct full path for video
            video_file = Path(video_path)
            if not video_file.exists():
                # Try uploaded folder
                video_file = UPLOAD_FOLDER / video_path
            if not video_file.exists():
                # Try local video folder
                local_video_dir = Path(r'C:\Users\sakth\Documents\traffique_footage')
                video_file = local_video_dir / video_path
            
            if not video_file.exists():
                print(f"   ❌ Video not found: {video_path}")
                continue
            
            print(f"   📁 Using video: {video_file}")
            
            # Initialize analyzer for this camera
            cam_analyzer = VehicleAnalyzer(
                use_sahi=use_sahi
            )
            
            # Load the model
            cam_analyzer.load_model()
            
            # Run analytics
            analytics, annotated_frame = cam_analyzer.analytics_mode(
                video_path=str(video_file),
                polygon_points=polygon_points,
                frame_idx=frame_idx,
                time_window=time_window
            )
            
            # Transform detections to world coordinates
            H = np.array(camera['homography_matrix'], dtype=np.float32)
            
            for detection in analytics:
                # Extract start/end positions (they might be tuples or separate keys)
                if 'start_x' in detection:
                    start_x, start_y = detection['start_x'], detection['start_y']
                    end_x, end_y = detection['end_x'], detection['end_y']
                elif 'start_position' in detection:
                    start_x, start_y = detection['start_position']
                    end_x, end_y = detection['end_position']
                else:
                    print(f"⚠️ Detection missing position data: {detection.keys()}")
                    continue
                
                # Transform start and end positions
                start_pt = np.array([[[start_x, start_y]]], dtype=np.float32)
                end_pt = np.array([[[end_x, end_y]]], dtype=np.float32)
                
                world_start = cv2.perspectiveTransform(start_pt, H)[0][0]
                world_end = cv2.perspectiveTransform(end_pt, H)[0][0]
                
                # Transform trajectory points (handle different possible keys)
                trajectory_world = []
                trajectory_points = detection.get('trajectory_points', [])
                
                # If trajectory_points is a number (count), skip trajectory transformation
                if isinstance(trajectory_points, (int, float)):
                    trajectory_points = []
                
                for point in trajectory_points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
                        world_pt = cv2.perspectiveTransform(pt, H)[0][0]
                        trajectory_world.append([float(world_pt[0]), float(world_pt[1])])
                
                # Calculate world-space velocity (m/s instead of px/s)
                distance_world = np.sqrt((world_end[0] - world_start[0])**2 + 
                                        (world_end[1] - world_start[1])**2)
                time_in_scene = detection.get('time_in_scene', 0)
                velocity_world = distance_world / time_in_scene if time_in_scene > 0 else 0
                
                all_detections.append({
                    'camera_id': camera['camera_id'],
                    'vehicle_id': detection['vehicle_id'],
                    'class': detection['class'],
                    'frames_tracked': detection.get('num_frames', detection.get('frames_tracked', 0)),
                    'world_x_start': float(world_start[0]),
                    'world_y_start': float(world_start[1]),
                    'world_x_end': float(world_end[0]),
                    'world_y_end': float(world_end[1]),
                    'velocity_mps': float(velocity_world),
                    'distance_m': float(distance_world),
                    'time_in_scene': time_in_scene,
                    'trajectory_world': trajectory_world,
                    'pixel_x_start': start_x,
                    'pixel_y_start': start_y,
                    'pixel_x_end': end_x,
                    'pixel_y_end': end_y
                })
            
            # Store camera result
            camera_results.append({
                'camera_id': camera['camera_id'],
                'vehicles_detected': len(analytics),
                'annotated_frame': base64.b64encode(cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]).decode('utf-8') if annotated_frame is not None else None
            })
            
            print(f"   ✅ Camera {camera['camera_id']}: {len(analytics)} vehicles detected")
        
        # Fuse detections (simple spatial clustering for now)
        fused_detections = fuse_detections(all_detections, spatial_threshold=2.0)  # 2 meters
        
        # Generate fused CSV
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / 'multicamera_analytics.csv'
        
        import pandas as pd
        df = pd.DataFrame(fused_detections)
        df.to_csv(csv_path, index=False)
        
        # Count cross-camera tracks
        camera_counts = {}
        for det in fused_detections:
            cameras_seen = det.get('cameras_seen', [])
            if not cameras_seen and 'camera_id' in det:
                cameras_seen = [det['camera_id']]
            for cam_id in cameras_seen:
                camera_counts[cam_id] = camera_counts.get(cam_id, 0) + 1
        
        cross_camera_tracks = len([d for d in fused_detections if len(d.get('cameras_seen', [])) > 1])
        
        print(f"\n{'='*60}")
        print(f"✅ FUSION COMPLETE")
        print(f"   Total unique vehicles: {len(fused_detections)}")
        print(f"   Cross-camera tracks: {cross_camera_tracks}")
        print(f"   CSV saved: {csv_path}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'total_vehicles': len(fused_detections),
            'cross_camera_tracks': cross_camera_tracks,
            'camera_results': camera_results,
            'fused_detections': fused_detections[:100],  # Limit to first 100 for response size
            'csv_available': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def fuse_detections(detections, spatial_threshold=2.0):
    """
    Fuse detections from multiple cameras
    Groups vehicles that are within spatial_threshold meters
    """
    if not detections:
        return []
    
    # Simple spatial clustering based on average position
    fused = []
    used_indices = set()
    
    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue
        
        # Calculate average position
        avg_x = (det1['world_x_start'] + det1['world_x_end']) / 2
        avg_y = (det1['world_y_start'] + det1['world_y_end']) / 2
        
        # Find nearby detections from other cameras
        cluster = [det1]
        cluster_cameras = [det1['camera_id']]
        used_indices.add(i)
        
        for j, det2 in enumerate(detections):
            if j in used_indices or det2['camera_id'] == det1['camera_id']:
                continue
            
            # Calculate distance between vehicles
            avg_x2 = (det2['world_x_start'] + det2['world_x_end']) / 2
            avg_y2 = (det2['world_y_start'] + det2['world_y_end']) / 2
            
            distance = np.sqrt((avg_x - avg_x2)**2 + (avg_y - avg_y2)**2)
            
            if distance < spatial_threshold:
                cluster.append(det2)
                cluster_cameras.append(det2['camera_id'])
                used_indices.add(j)
        
        # Merge cluster (weighted average by frames_tracked)
        if len(cluster) == 1:
            # Single detection
            merged = det1.copy()
            merged['cameras_seen'] = cluster_cameras
            merged['global_id'] = f"V{len(fused) + 1:04d}"
        else:
            # Multiple detections - merge
            total_weight = sum(d['frames_tracked'] for d in cluster)
            
            merged = {
                'global_id': f"V{len(fused) + 1:04d}",
                'cameras_seen': cluster_cameras,
                'class': cluster[0]['class'],  # Take most common class
                'frames_tracked': sum(d['frames_tracked'] for d in cluster),
                'world_x_start': sum(d['world_x_start'] * d['frames_tracked'] for d in cluster) / total_weight,
                'world_y_start': sum(d['world_y_start'] * d['frames_tracked'] for d in cluster) / total_weight,
                'world_x_end': sum(d['world_x_end'] * d['frames_tracked'] for d in cluster) / total_weight,
                'world_y_end': sum(d['world_y_end'] * d['frames_tracked'] for d in cluster) / total_weight,
                'velocity_mps': sum(d['velocity_mps'] * d['frames_tracked'] for d in cluster) / total_weight,
                'distance_m': sum(d['distance_m'] for d in cluster) / len(cluster),
                'time_in_scene': sum(d['time_in_scene'] for d in cluster) / len(cluster),
                'trajectory_world': cluster[0]['trajectory_world']  # Take first trajectory
            }
        
        fused.append(merged)
    
    return fused


@app.route('/api/detect-overlap', methods=['POST'])
def detect_overlap():
    """
    Detect overlap between two cameras using feature matching
    """
    try:
        data = request.json
        cam1_path = data.get('cam1_video')
        cam2_path = data.get('cam2_video')
        cam1_frame_idx = data.get('cam1_frame_idx', 0)
        cam2_frame_idx = data.get('cam2_frame_idx', 0)
        cam1_homography = np.array(data.get('cam1_homography'))
        cam2_scale = data.get('cam2_scale')  # pixels per meter
        
        # Resolve video paths
        local_video_dir = Path(r'C:\Users\sakth\Documents\traffique_footage')
        
        def resolve_path(path_str):
            path = Path(path_str)
            if path.exists():
                return path
            path = UPLOAD_FOLDER / path_str
            if path.exists():
                return path
            path = local_video_dir / path_str
            if path.exists():
                return path
            return None
        
        cam1_path = resolve_path(cam1_path)
        cam2_path = resolve_path(cam2_path)
        
        if not cam1_path or not cam2_path:
            return jsonify({'error': 'Video files not found'}), 404
        
        # Read frames
        cap1 = cv2.VideoCapture(str(cam1_path))
        cap1.set(cv2.CAP_PROP_POS_FRAMES, cam1_frame_idx)
        ret1, frame1 = cap1.read()
        cap1.release()
        
        cap2 = cv2.VideoCapture(str(cam2_path))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, cam2_frame_idx)
        ret2, frame2 = cap2.read()
        cap2.release()
        
        if not ret1 or not ret2:
            return jsonify({'error': 'Could not read frames'}), 400
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Detect features using ORB (faster than SIFT, no patent issues)
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # Not enough features found
            return jsonify({
                'overlap_detected': False,
                'overlap_x': 0,
                'overlap_y': 0,
                'confidence': 0,
                'message': 'Not enough features detected for matching'
            })
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep only good matches (top 30%)
        num_good_matches = int(len(matches) * 0.3)
        good_matches = matches[:max(num_good_matches, 10)]
        
        if len(good_matches) < 10:
            # Not enough good matches - no overlap
            return jsonify({
                'overlap_detected': False,
                'overlap_x': 0,
                'overlap_y': 0,
                'confidence': 0,
                'message': f'Only {len(good_matches)} good matches found (need 10+)'
            })
        
        # Extract matched keypoint coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Transform CAM1 matched points to world coordinates using its homography
        pts1_homog = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts1_world = (cam1_homography @ pts1_homog.T).T
        pts1_world = pts1_world[:, :2] / pts1_world[:, 2:3]
        
        # For CAM2 points, we need to figure out where they map in world coords
        # We know CAM2's scale (pixels per meter), but not its position yet
        # The matched features tell us: "these CAM2 pixels correspond to these world positions"
        
        # Calculate where CAM2's origin (0,0) would be in world coordinates
        # Using: world_coord = cam2_origin + (pixel_coord / scale)
        # Rearranged: cam2_origin = world_coord - (pixel_coord / scale)
        
        cam2_origins_x = []
        cam2_origins_y = []
        
        for i in range(len(good_matches)):
            # This pixel in CAM2 corresponds to this world position (from CAM1)
            cam2_pixel_x, cam2_pixel_y = pts2[i]
            world_x, world_y = pts1_world[i]
            
            # Calculate what CAM2's origin must be
            cam2_origin_x = world_x - (cam2_pixel_x / cam2_scale)
            cam2_origin_y = world_y - (cam2_pixel_y / cam2_scale)
            
            cam2_origins_x.append(cam2_origin_x)
            cam2_origins_y.append(cam2_origin_y)
        
        # Take median to be robust against outliers
        cam2_origin_x = np.median(cam2_origins_x)
        cam2_origin_y = np.median(cam2_origins_y)
        
        # Calculate overlap
        # CAM1 starts at 0, ends at some X value (get from homography's world coverage)
        # CAM2 starts at cam2_origin_x
        # If cam2_origin_x < cam1_end_x, there's overlap
        
        # Get CAM1's coverage from its world points (we need to pass this)
        cam1_coverage = data.get('cam1_coverage', {})
        cam1_end_x = cam1_coverage.get('end_x', 0)
        cam1_end_y = cam1_coverage.get('end_y', 0)
        
        overlap_x = max(0, cam1_end_x - cam2_origin_x)
        overlap_y = max(0, cam1_end_y - cam2_origin_y)
        
        # Calculate confidence based on:
        # 1. Number of good matches
        # 2. Consistency of calculated origins (low std dev = high confidence)
        std_x = np.std(cam2_origins_x)
        std_y = np.std(cam2_origins_y)
        
        match_confidence = min(1.0, len(good_matches) / 50.0)  # More matches = higher confidence
        consistency_confidence = 1.0 / (1.0 + std_x / 10.0)  # Lower std = higher confidence
        overall_confidence = (match_confidence + consistency_confidence) / 2
        
        return jsonify({
            'overlap_detected': overlap_x > 0.5 or overlap_y > 0.5,  # Threshold: 0.5m
            'overlap_x': float(overlap_x),
            'overlap_y': float(overlap_y),
            'confidence': float(overall_confidence),
            'num_matches': len(good_matches),
            'cam2_origin_x': float(cam2_origin_x),
            'cam2_origin_y': float(cam2_origin_y),
            'message': f'Found {len(good_matches)} feature matches (confidence: {overall_confidence:.2f})'
        })
        
    except Exception as e:
        print(f"❌ Overlap detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("🚀 VEHICLE ANALYTICS API SERVER")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /api/upload - Upload video")
    print("  GET  /api/frame/<filename>/<frame_idx> - Get frame")
    print("  POST /api/analyze/quick - Quick mode analysis")
    print("  POST /api/analyze/full - Full analytics mode")
    print("  POST /api/calibrate-camera - Calibrate camera")
    print("  POST /api/analyze/multi-camera - Multi-camera fusion")
    print("  POST /api/detect-overlap - Detect overlap using feature matching")
    print("  GET  /api/download/csv - Download results CSV")
    print("\nStarting server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
