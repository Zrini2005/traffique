"""
Professional Traffic Analysis GUI using PyQt5
Features:
- Smooth video playback
- Interactive polygon ROI drawing
- Vehicle trajectory visualization
- Click vehicle to see its full path
- Professional UI design
"""
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QFileDialog, QListWidget, QSplitter, QMessageBox,
                             QComboBox, QProgressBar, QFrame, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush
import requests
import json
import base64
from io import BytesIO
from PIL import Image as PILImage


API_BASE = "http://localhost:5000"
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


class VideoCanvas(QLabel):
    """Custom canvas for video display with polygon drawing and trajectory overlay"""
    clicked = pyqtSignal(int, int)
    vehicle_clicked = pyqtSignal(int)  # Signal for vehicle click
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setScaledContents(False)
        self.setStyleSheet("border: 2px solid #2c3e50; background-color: #000000;")
        self.setAlignment(Qt.AlignCenter)
        
        self.current_frame = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.polygon_points = []
        self.drawing_polygon = False
        self.selected_vehicle_trajectory = []
        self.current_frame_vehicles = []
        
    def set_frame(self, frame):
        """Set the current video frame"""
        self.current_frame = frame
        self.update()
        
    def mousePressEvent(self, event):
        """Handle mouse clicks for polygon drawing and vehicle selection"""
        if event.button() == Qt.LeftButton:
            # Convert click coordinates to image coordinates
            x = int((event.x() - self.offset_x) / self.scale_factor)
            y = int((event.y() - self.offset_y) / self.scale_factor)
            
            if 0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]:
                if self.drawing_polygon:
                    # Add polygon point
                    self.polygon_points.append({'x': x, 'y': y})
                    self.update()
                else:
                    # Check if clicked on a vehicle
                    for vehicle in self.current_frame_vehicles:
                        x1, y1, x2, y2 = vehicle['bbox']
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            # Emit signal with vehicle ID
                            self.vehicle_clicked.emit(vehicle['id'])
                            break
                
    def paintEvent(self, event):
        """Custom paint event to draw frame, polygon, and trajectories"""
        super().paintEvent(event)
        
        if self.current_frame is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the video frame
        frame_copy = self.current_frame.copy()
        
        # Draw polygon on frame if exists
        if len(self.polygon_points) >= 2:
            pts = np.array([[p['x'], p['y']] for p in self.polygon_points], np.int32)
            
            # Fill polygon with semi-transparent green
            if len(self.polygon_points) >= 3:
                overlay = frame_copy.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame_copy, 0.7, 0, frame_copy)
            
            # Draw polygon outline
            cv2.polylines(frame_copy, [pts], True, (0, 255, 0), 3)
            
            # Draw points
            for p in self.polygon_points:
                cv2.circle(frame_copy, (p['x'], p['y']), 8, (0, 255, 0), -1)
                cv2.circle(frame_copy, (p['x'], p['y']), 8, (255, 255, 255), 2)
        
        # Draw current frame vehicles (bounding boxes)
        for vehicle in self.current_frame_vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            vehicle_id = vehicle.get('id', 'N/A')
            conf = vehicle.get('confidence', 0)
            
            # Draw bounding box
            color = (0, 255, 255)  # Yellow
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"ID:{vehicle_id} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (int(x1), int(y1) - h - 5), (int(x1) + w, int(y1)), color, -1)
            cv2.putText(frame_copy, label, (int(x1), int(y1) - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw selected vehicle trajectory
        if len(self.selected_vehicle_trajectory) > 1:
            pts = np.array(self.selected_vehicle_trajectory, np.int32)
            cv2.polylines(frame_copy, [pts], False, (255, 0, 255), 3)
            
            # Draw points along trajectory
            for i, pt in enumerate(pts):
                if i % 5 == 0:  # Draw every 5th point
                    cv2.circle(frame_copy, tuple(pt), 5, (255, 0, 255), -1)
                    cv2.circle(frame_copy, tuple(pt), 5, (255, 255, 255), 1)
            
            # Draw start and end points
            cv2.circle(frame_copy, tuple(pts[0]), 10, (0, 255, 0), -1)  # Green for start
            cv2.circle(frame_copy, tuple(pts[-1]), 10, (255, 0, 0), -1)  # Red for end
        
        # Convert to QPixmap
        height, width, channel = frame_copy.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_copy.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Calculate scaling to fit in widget
        widget_width = self.width()
        widget_height = self.height()
        
        self.scale_factor = min(widget_width / width, widget_height / height)
        
        scaled_width = int(width * self.scale_factor)
        scaled_height = int(height * self.scale_factor)
        
        self.offset_x = (widget_width - scaled_width) // 2
        self.offset_y = (widget_height - scaled_height) // 2
        
        pixmap = QPixmap.fromImage(q_img).scaled(scaled_width, scaled_height, 
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        painter.drawPixmap(self.offset_x, self.offset_y, pixmap)


class AnalysisWorker(QThread):
    """Background worker for video analysis"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, polygon, current_frame_idx):
        super().__init__()
        self.video_path = video_path
        self.polygon = polygon
        self.current_frame_idx = current_frame_idx
        
    def run(self):
        """Run analysis in background"""
        try:
            self.progress.emit("Uploading video and starting analysis...")
            
            # Prepare payload
            payload = {
                'filename': Path(self.video_path).name,
                'frame_idx': self.current_frame_idx,
                'polygon': self.polygon,
                'confidence': 0.15
            }
            
            # Send to API (no timeout - analysis can take a long time)
            response = requests.post(f"{API_BASE}/api/analyze", json=payload, timeout=None)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.finished.emit(data)
                else:
                    self.error.emit(f"Analysis failed: {data.get('error', 'Unknown error')}")
            else:
                self.error.emit(f"API error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.error.emit("Cannot connect to API server. Please start the server with: python api_server.py")
        except Exception as e:
            self.error.emit(f"Analysis error: {str(e)}")


class TrafficAnalysisGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Analysis - Professional GUI")
        self.setGeometry(100, 100, 1400, 900)
        
        # State variables
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame_idx = 0
        self.playing = False
        self.analysis_results = None
        self.tracks_data = None
        
        # Timer for playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.init_ui()
        self.auto_load_default_video()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Title with authors
        title_layout = QHBoxLayout()
        title = QLabel("🚗 Traffic Analysis System")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50; padding: 10px;")
        title_layout.addWidget(title)
        
        # Authors (small text on the right)
        authors = QLabel("by Sakthivel, Srinivas, Yashwanth")
        authors.setStyleSheet("font-size: 11px; color: #7f8c8d; padding: 10px; font-style: italic;")
        authors.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_layout.addWidget(authors)
        
        left_layout.addLayout(title_layout)
        
        # Video canvas
        self.canvas = VideoCanvas()
        self.canvas.vehicle_clicked.connect(self.on_vehicle_clicked_on_video)
        left_layout.addWidget(self.canvas)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setStyleSheet("font-weight: bold;")
        slider_layout.addWidget(self.frame_label)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        slider_layout.addWidget(self.frame_slider)
        
        left_layout.addLayout(slider_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📁 Load Video")
        self.load_btn.setStyleSheet(self.button_style("#3498db"))
        self.load_btn.clicked.connect(self.load_video)
        button_layout.addWidget(self.load_btn)
        
        self.local_videos_btn = QPushButton("📂 Local Videos")
        self.local_videos_btn.setStyleSheet(self.button_style("#9b59b6"))
        self.local_videos_btn.clicked.connect(self.show_local_videos)
        button_layout.addWidget(self.local_videos_btn)
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setStyleSheet(self.button_style("#27ae60"))
        self.play_btn.clicked.connect(self.toggle_play)
        button_layout.addWidget(self.play_btn)
        
        self.draw_roi_btn = QPushButton("✏ Draw ROI")
        self.draw_roi_btn.setStyleSheet(self.button_style("#e67e22"))
        self.draw_roi_btn.clicked.connect(self.start_draw_roi)
        button_layout.addWidget(self.draw_roi_btn)
        
        self.finish_roi_btn = QPushButton("✓ Finish ROI")
        self.finish_roi_btn.setStyleSheet(self.button_style("#16a085"))
        self.finish_roi_btn.clicked.connect(self.finish_draw_roi)
        self.finish_roi_btn.setEnabled(False)
        button_layout.addWidget(self.finish_roi_btn)
        
        self.analyze_btn = QPushButton("🔍 Run Analysis")
        self.analyze_btn.setStyleSheet(self.button_style("#c0392b"))
        self.analyze_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(self.analyze_btn)
        
        left_layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready. Load a video to begin.")
        self.status_label.setStyleSheet("padding: 8px; background-color: #ecf0f1; border-radius: 4px; color: #2c3e50;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Right panel - Vehicle list and info
        right_panel = QWidget()
        right_panel.setMaximumWidth(350)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Video selector
        video_label = QLabel("📹 Current Video:")
        video_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        right_layout.addWidget(video_label)
        
        self.video_name_label = QLabel("No video loaded")
        self.video_name_label.setStyleSheet("padding: 5px; background-color: #ecf0f1; border-radius: 4px;")
        self.video_name_label.setWordWrap(True)
        right_layout.addWidget(self.video_name_label)
        
        # ROI info
        roi_label = QLabel("📍 ROI Points:")
        roi_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 10px;")
        right_layout.addWidget(roi_label)
        
        self.roi_info_label = QLabel("No ROI defined")
        self.roi_info_label.setStyleSheet("padding: 5px; background-color: #ecf0f1; border-radius: 4px;")
        right_layout.addWidget(self.roi_info_label)
        
        # Vehicle list
        vehicle_label = QLabel("🚙 Detected Vehicles:")
        vehicle_label.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 10px;")
        right_layout.addWidget(vehicle_label)
        
        self.vehicle_list = QListWidget()
        self.vehicle_list.setStyleSheet("""
            QListWidget {
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e8f4f8;
            }
        """)
        self.vehicle_list.itemClicked.connect(self.on_vehicle_selected)
        right_layout.addWidget(self.vehicle_list)
        
        # Vehicle info
        self.vehicle_info_label = QLabel("Select a vehicle to see trajectory")
        self.vehicle_info_label.setStyleSheet("""
            padding: 10px; 
            background-color: #fff3cd; 
            border: 1px solid #ffc107; 
            border-radius: 4px;
            color: #856404;
        """)
        self.vehicle_info_label.setWordWrap(True)
        right_layout.addWidget(self.vehicle_info_label)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
    def button_style(self, color):
        """Generate button style"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #95a5a6;
            }}
        """
        
    def lighten_color(self, hex_color):
        """Lighten a hex color"""
        return hex_color  # Simplified
        
    def darken_color(self, hex_color):
        """Darken a hex color"""
        return hex_color  # Simplified
        
    def auto_load_default_video(self):
        """Auto-load default video on startup"""
        default_video = UPLOAD_FOLDER / "D2F1_stab_10sec.mp4"
        if not default_video.exists():
            default_video = UPLOAD_FOLDER / "D2F1_stab_60sec.mp4"
            
        if default_video.exists():
            self.load_video_file(str(default_video))
            
    def load_video(self):
        """Open file dialog to load video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", str(UPLOAD_FOLDER),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_path:
            self.load_video_file(file_path)
            
    def show_local_videos(self):
        """Show list of local videos"""
        videos = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            videos.extend(list(UPLOAD_FOLDER.glob(f'*{ext}')))
            
        if not videos:
            QMessageBox.information(self, "No Videos", "No videos found in uploads folder.")
            return
            
        # Create dialog with video list
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Local Video")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        list_widget = QListWidget()
        for v in videos:
            size_mb = v.stat().st_size / (1024 * 1024)
            item = QListWidgetItem(f"{v.name} ({size_mb:.1f} MB)")
            item.setData(Qt.UserRole, str(v))
            list_widget.addItem(item)
            
        list_widget.itemDoubleClicked.connect(lambda item: self.load_from_dialog(item, dialog))
        layout.addWidget(list_widget)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.load_from_dialog(list_widget.currentItem(), dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def load_from_dialog(self, item, dialog):
        """Load video from dialog selection"""
        if item:
            video_path = item.data(Qt.UserRole)
            self.load_video_file(video_path)
            dialog.accept()
            
    def load_video_file(self, file_path):
        """Load a video file"""
        try:
            self.status_label.setText(f"Loading video: {Path(file_path).name}...")
            
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open video file")
                
            self.video_path = file_path
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.current_frame_idx = 0
            
            self.video_name_label.setText(f"{Path(file_path).name}\n{width}x{height}, {self.fps:.1f} fps, {self.total_frames} frames")
            self.status_label.setText(f"✓ Video loaded successfully: {self.total_frames} frames")
            
            self.load_frame(0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            
    def load_frame(self, frame_idx):
        """Load and display a specific frame"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame_idx = frame_idx
            self.canvas.set_frame(frame)
            self.frame_label.setText(f"Frame: {frame_idx + 1} / {self.total_frames}")
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_idx)
            self.frame_slider.blockSignals(False)
            
            # Update vehicles on current frame
            self.update_current_frame_vehicles()
            
    def seek_frame(self, frame_idx):
        """Seek to a specific frame"""
        self.load_frame(frame_idx)
        
    def toggle_play(self):
        """Toggle video playback"""
        if not self.cap:
            return
            
        self.playing = not self.playing
        
        if self.playing:
            self.play_btn.setText("⏸ Pause")
            interval = int(1000 / self.fps) if self.fps > 0 else 33
            self.play_timer.start(interval)
        else:
            self.play_btn.setText("▶ Play")
            self.play_timer.stop()
            
    def next_frame(self):
        """Go to next frame during playback"""
        if self.current_frame_idx < self.total_frames - 1:
            self.load_frame(self.current_frame_idx + 1)
        else:
            self.toggle_play()  # Stop at end
            
    def start_draw_roi(self):
        """Start drawing ROI polygon"""
        self.canvas.polygon_points = []
        self.canvas.drawing_polygon = True
        self.finish_roi_btn.setEnabled(True)
        self.status_label.setText("🖊 Drawing ROI: Click on the video to add points. Click 'Finish ROI' when done.")
        self.roi_info_label.setText("Drawing... (0 points)")
        self.canvas.update()
        
    def finish_draw_roi(self):
        """Finish drawing ROI polygon"""
        self.canvas.drawing_polygon = False
        self.finish_roi_btn.setEnabled(False)
        
        num_points = len(self.canvas.polygon_points)
        if num_points < 3:
            self.status_label.setText("⚠ ROI needs at least 3 points!")
            self.roi_info_label.setText("Invalid ROI (< 3 points)")
        else:
            self.status_label.setText(f"✓ ROI defined with {num_points} points")
            self.roi_info_label.setText(f"{num_points} points defined")
        
        self.canvas.update()
        
    def run_analysis(self):
        """Run vehicle tracking analysis"""
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
            
        if len(self.canvas.polygon_points) < 3:
            QMessageBox.warning(self, "No ROI", "Please draw a region of interest (ROI) with at least 3 points.")
            return
            
        # Disable buttons during analysis
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        # Start analysis worker
        self.worker = AnalysisWorker(self.video_path, self.canvas.polygon_points, self.current_frame_idx)
        self.worker.progress.connect(self.on_analysis_progress)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
        
    def on_analysis_progress(self, message):
        """Handle analysis progress updates"""
        self.status_label.setText(f"🔄 {message}")
        
    def on_analysis_finished(self, results):
        """Handle analysis completion"""
        self.analysis_results = results
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        print(f"Analysis results keys: {results.keys()}")
        print(f"Analysis results: {results}")
        
        # Load tracks data - try multiple approaches
        # API returns 'csv_file' not 'tracks_csv'
        tracks_file = results.get('csv_file') or results.get('tracks_csv')
        
        if tracks_file:
            print(f"Tracks file from API: {tracks_file}")
            
            # Try multiple paths
            possible_paths = [
                Path(tracks_file),  # Direct path
                Path("output") / Path(tracks_file).name,  # Output folder
                Path("output") / tracks_file,  # Output with relative path
                Path(".") / tracks_file,  # Current directory
            ]
            
            csv_path = None
            for path in possible_paths:
                print(f"Trying path: {path}")
                if path.exists():
                    csv_path = path
                    print(f"Found at: {path}")
                    break
            
            if csv_path:
                try:
                    self.tracks_data = pd.read_csv(csv_path)
                    print(f"Loaded {len(self.tracks_data)} rows from CSV")
                    print(f"Columns: {self.tracks_data.columns.tolist()}")
                    
                    # Handle different CSV column formats
                    if 'VehicleID' in self.tracks_data.columns:
                        # New format: Frame, VehicleID, Class, X_pixel, Y_pixel, BBox, Time
                        print("Converting CSV format: VehicleID -> track_id")
                        self.tracks_data['track_id'] = self.tracks_data['VehicleID']
                        self.tracks_data['frame_id'] = self.tracks_data['Frame']
                        
                        # Parse BBox column [x1, y1, x2, y2]
                        if 'BBox' in self.tracks_data.columns:
                            import ast
                            bbox_data = self.tracks_data['BBox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                            self.tracks_data['bbox_x1'] = bbox_data.apply(lambda x: x[0])
                            self.tracks_data['bbox_y1'] = bbox_data.apply(lambda x: x[1])
                            self.tracks_data['bbox_x2'] = bbox_data.apply(lambda x: x[2])
                            self.tracks_data['bbox_y2'] = bbox_data.apply(lambda x: x[3])
                        
                        self.tracks_data['confidence'] = 0.5  # Default confidence
                    
                    if 'track_id' in self.tracks_data.columns:
                        self.populate_vehicle_list()
                        num_vehicles = len(self.tracks_data['track_id'].unique())
                        self.status_label.setText(f"✓ Analysis complete! Found {num_vehicles} vehicles.")
                    else:
                        self.status_label.setText(f"⚠ CSV loaded but no 'track_id' column. Columns: {self.tracks_data.columns.tolist()}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.status_label.setText(f"⚠ Failed to load tracks: {str(e)}")
            else:
                # List output folder contents for debugging
                output_dir = Path("output")
                if output_dir.exists():
                    files = list(output_dir.glob("*.csv"))
                    print(f"CSV files in output folder: {[f.name for f in files]}")
                    self.status_label.setText(f"⚠ Tracks file not found. CSV files in output: {len(files)}")
                else:
                    self.status_label.setText(f"⚠ Output folder doesn't exist")
        else:
            print("No tracks_csv in results")
            # Maybe the data is embedded in the response?
            if 'vehicles' in results:
                print(f"Found 'vehicles' in results: {len(results['vehicles'])} vehicles")
                self.status_label.setText(f"⚠ No CSV file but found {len(results['vehicles'])} vehicles in response")
            else:
                self.status_label.setText(f"⚠ No tracks data. Response keys: {list(results.keys())}")
            
    def on_analysis_error(self, error_message):
        """Handle analysis errors"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.status_label.setText(f"❌ {error_message}")
        
    def populate_vehicle_list(self):
        """Populate the vehicle list from tracks data"""
        self.vehicle_list.clear()
        
        if self.tracks_data is None or self.tracks_data.empty:
            return
            
        # Get unique vehicles and count frames for each
        unique_vehicles = self.tracks_data['track_id'].unique()
        
        # Build list of (vehicle_id, num_frames) and sort by num_frames descending
        vehicle_frame_counts = []
        for vehicle_id in unique_vehicles:
            vehicle_tracks = self.tracks_data[self.tracks_data['track_id'] == vehicle_id]
            num_frames = len(vehicle_tracks)
            vehicle_frame_counts.append((vehicle_id, num_frames))
        
        # Sort by num_frames (descending), then by vehicle_id
        vehicle_frame_counts.sort(key=lambda x: (-x[1], x[0]))
        
        for vehicle_id, num_frames in vehicle_frame_counts:
            item = QListWidgetItem(f"🚗 Vehicle {vehicle_id} ({num_frames} frames)")
            item.setData(Qt.UserRole, int(vehicle_id))
            self.vehicle_list.addItem(item)
            
    def update_current_frame_vehicles(self):
        """Update vehicles visible on current frame"""
        self.canvas.current_frame_vehicles = []
        
        if self.tracks_data is None or self.tracks_data.empty:
            return
            
        # Get vehicles on current frame
        frame_data = self.tracks_data[self.tracks_data['frame_id'] == self.current_frame_idx]
        
        for _, row in frame_data.iterrows():
            vehicle = {
                'id': int(row['track_id']),
                'bbox': [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
                'confidence': row.get('confidence', 0.5)
            }
            self.canvas.current_frame_vehicles.append(vehicle)
            
        self.canvas.update()
        
    def on_vehicle_selected(self, item):
        """Handle vehicle selection from list"""
        vehicle_id = item.data(Qt.UserRole)
        self.show_vehicle_trajectory(vehicle_id)
    
    def on_vehicle_clicked_on_video(self, vehicle_id):
        """Handle vehicle click on video canvas"""
        if self.tracks_data is None or self.tracks_data.empty:
            return
        
        # Call the same handler as list selection
        self.show_vehicle_trajectory(vehicle_id)
        
        # Also select it in the list
        for i in range(self.vehicle_list.count()):
            item = self.vehicle_list.item(i)
            if item.data(Qt.UserRole) == vehicle_id:
                self.vehicle_list.setCurrentItem(item)
                break
    
    def show_vehicle_trajectory(self, vehicle_id):
        """Show trajectory for a specific vehicle"""
        if self.tracks_data is None or self.tracks_data.empty:
            return
            
        # Get full trajectory for this vehicle
        vehicle_tracks = self.tracks_data[self.tracks_data['track_id'] == vehicle_id].sort_values('frame_id')
        
        if vehicle_tracks.empty:
            return
            
        # Extract trajectory points (center of bounding box)
        trajectory = []
        for _, row in vehicle_tracks.iterrows():
            center_x = (row['bbox_x1'] + row['bbox_x2']) / 2
            center_y = (row['bbox_y1'] + row['bbox_y2']) / 2
            trajectory.append([int(center_x), int(center_y)])
            
        self.canvas.selected_vehicle_trajectory = trajectory
        self.canvas.update()
        
        # Update info label
        num_frames = len(vehicle_tracks)
        first_frame = vehicle_tracks['frame_id'].min()
        last_frame = vehicle_tracks['frame_id'].max()
        
        self.vehicle_info_label.setText(
            f"🚗 Vehicle {vehicle_id}\n"
            f"📊 Frames: {first_frame} → {last_frame} ({num_frames} total)\n"
            f"📍 Trajectory: {len(trajectory)} points\n"
            f"💚 Trajectory shown in MAGENTA on video"
        )
        
        # Jump to first frame of this vehicle
        self.load_frame(int(first_frame))
        
    def closeEvent(self, event):
        """Clean up on close"""
        if self.cap:
            self.cap.release()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = TrafficAnalysisGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
