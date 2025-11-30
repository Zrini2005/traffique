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
sys.path.append(str(Path(__file__).parent))
from interactive_analytics import VehicleAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global analyzer instance
analyzer = None


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


@app.route('/api/analyze/quick', methods=['POST'])
def analyze_quick():
    """Quick mode analysis - single frame"""
    global analyzer
    
    data = request.json
    filename = data.get('filename')
    frame_idx = data.get('frame_idx')
    polygon = data.get('polygon')  # List of [x, y] points
    use_sahi = data.get('use_sahi', False)
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
        # Initialize analyzer if needed
        if analyzer is None or analyzer.model_conf != confidence or analyzer.use_sahi != use_sahi:
            print(f"Initializing analyzer with confidence={confidence}, use_sahi={use_sahi}")
            analyzer = VehicleAnalyzer(model_conf=confidence, use_sahi=use_sahi, sahi_slice_size=640)
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
        cv2.putText(panel, f"SAHI: {'ON' if use_sahi else 'OFF'}", (400, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (23, 77, 56), 2)
        cv2.putText(panel, f"Confidence: {confidence:.2f}", (400, 65),
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


@app.route('/api/analyze/full', methods=['POST'])
def analyze_full():
    """Analytics mode - full tracking and trajectories"""
    global analyzer
    
    data = request.json
    filename = data.get('filename')
    frame_idx = data.get('frame_idx')
    polygon = data.get('polygon')
    time_window = data.get('time_window', 5)  # seconds
    use_sahi = data.get('use_sahi', False)
    confidence = data.get('confidence', 0.20)
    calibration_points = data.get('calibration_points')  # Optional
    real_world_dims = data.get('real_world_dims')  # {width: m, height: m}
    
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
        # Initialize analyzer
        if analyzer is None or analyzer.model_conf != confidence or analyzer.use_sahi != use_sahi:
            analyzer = VehicleAnalyzer(model_conf=confidence, use_sahi=use_sahi, sahi_slice_size=640)
            analyzer.load_model()
        
        # Calibrate if points provided
        if calibration_points and real_world_dims:
            cap = cv2.VideoCapture(str(filepath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                src_points = np.float32(calibration_points)
                dst_points = np.float32([
                    [0, 0],
                    [real_world_dims['width'], 0],
                    [real_world_dims['width'], real_world_dims['height']],
                    [0, real_world_dims['height']]
                ])
                analyzer.coord_mapper.H = cv2.getPerspectiveTransform(src_points, dst_points)
                analyzer.coord_mapper.calibrated = True
        
        # Filter by polygon
        polygon_points = [(int(p['x']), int(p['y'])) for p in polygon]
        
        # Run analytics mode with polygon filtering
        analytics, annotated_frame = analyzer.analytics_mode(
            str(filepath),
            frame_idx=frame_idx,
            time_window=time_window,
            calibrate=False,  # Already calibrated above if needed
            polygon_points=polygon_points
        )
        
        # Encode annotated frame
        annotated_image = None
        if annotated_frame is not None:
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            annotated_image = base64.b64encode(buffer).decode('utf-8')
        
        # Return analytics
        return jsonify({
            'success': True,
            'total_tracks': len(analytics),
            'analytics': analytics[:100],  # Limit to first 100 for performance
            'annotated_image': annotated_image,
            'csv_file': 'output/vehicle_analytics.csv'
        })
    
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


if __name__ == '__main__':
    print("="*60)
    print("🚀 VEHICLE ANALYTICS API SERVER")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /api/upload - Upload video")
    print("  GET  /api/frame/<filename>/<frame_idx> - Get frame")
    print("  POST /api/analyze/quick - Quick mode analysis")
    print("  POST /api/analyze/full - Full analytics mode")
    print("  GET  /api/download/csv - Download results CSV")
    print("\nStarting server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
