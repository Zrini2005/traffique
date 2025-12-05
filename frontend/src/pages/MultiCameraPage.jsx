import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, MapPin, Download, Car, Loader, Check, X, AlertCircle, Trash2 } from 'lucide-react';

const API_BASE_URL = 'http://localhost:5000';

export default function MultiCameraPage() {
  const [cameras, setCameras] = useState([]);
  const [calibratingCamera, setCalibratingCamera] = useState(null);
  const [calibrationMode, setCalibrationMode] = useState('manual'); // 'manual' or 'auto'
  const [selectedVehicle, setSelectedVehicle] = useState(null);
  const [timeWindow, setTimeWindow] = useState(30);
  const [useSahi, setUseSahi] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [localVideos, setLocalVideos] = useState([]);
  const [showLocalVideos, setShowLocalVideos] = useState(false);
  
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Vehicle reference sizes (in meters)
  const VEHICLE_SIZES = {
    car: { length: 4.5, width: 1.8, name: 'Car (4.5m × 1.8m)' },
    truck: { length: 10.0, width: 2.5, name: 'Truck (10m × 2.5m)' },
    bus: { length: 12.0, width: 2.5, name: 'Bus (12m × 2.5m)' },
    motorcycle: { length: 2.0, width: 0.8, name: 'Motorcycle (2m × 0.8m)' }
  };

  useEffect(() => {
    loadLocalVideos();
  }, []);

  const loadLocalVideos = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/local-videos`);
      const data = await response.json();
      // Extract just filenames if videos is an array of objects
      const videoList = (data.videos || []).map(v => 
        typeof v === 'string' ? v : v.filename
      );
      setLocalVideos(videoList);
    } catch (error) {
      console.error('Failed to load local videos:', error);
    }
  };

  const addCamera = async (files) => {
    const filesArray = Array.from(files);
    
    for (const file of filesArray) {
      const cameraId = `CAM${cameras.length + 1}`;
      const newCamera = {
        id: cameraId,
        file,
        name: file.name,
        calibrated: false,
        imagePoints: [],
        worldPoints: [],
        homographyMatrix: null,
        videoPath: null,
        previewFrame: null,
        uploading: true
      };
      
      setCameras(prev => [...prev, newCamera]);
      
      // Upload immediately
      await uploadCamera(newCamera);
    }
  };

  const useLocalVideo = async (filename) => {
    try {
      const cameraId = `CAM${cameras.length + 1}`;
      const newCamera = {
        id: cameraId,
        name: filename,
        calibrated: false,
        imagePoints: [],
        worldPoints: [],
        homographyMatrix: null,
        videoPath: null,
        previewFrame: null,
        uploading: true,
        isLocal: true
      };
      
      setCameras(prev => [...prev, newCamera]);
      
      // Use local video endpoint
      const response = await fetch(`${API_BASE_URL}/api/use-local/${filename}`, {
        method: 'POST',
      });
      
      if (!response.ok) throw new Error('Failed to use local video');
      
      const data = await response.json();
      
      // Update camera with video info
      setCameras(prev => prev.map(c => 
        c.id === cameraId 
          ? { 
              ...c, 
              videoPath: data.filename, 
              frameCount: data.metadata.total_frames, 
              fps: data.metadata.fps, 
              uploading: false 
            }
          : c
      ));
      
      // Get first frame
      const frameResponse = await fetch(`${API_BASE_URL}/api/frame/${data.filename}/0`);
      const frameData = await frameResponse.json();
      
      setCameras(prev => prev.map(c => 
        c.id === cameraId 
          ? { ...c, previewFrame: frameData.frame }
          : c
      ));
      
      setShowLocalVideos(false);
      
    } catch (err) {
      setError(`Failed to load local video: ${err.message}`);
    }
  };

  const removeCamera = (id) => {
    setCameras(cameras.filter(c => c.id !== id));
  };

  const uploadCamera = async (camera) => {
    const formData = new FormData();
    formData.append('video', camera.file);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      
      // Update camera with video path
      setCameras(prev => prev.map(c => 
        c.id === camera.id 
          ? { ...c, videoPath: data.filename, frameCount: data.metadata.total_frames, fps: data.metadata.fps, uploading: false }
          : c
      ));
      
      // Get first frame for calibration
      const frameResponse = await fetch(`${API_BASE_URL}/api/frame/${data.filename}/0`);
      const frameData = await frameResponse.json();
      
      setCameras(prev => prev.map(c => 
        c.id === camera.id 
          ? { ...c, previewFrame: frameData.frame }
          : c
      ));
      
    } catch (err) {
      setError(`Failed to upload ${camera.name}: ${err.message}`);
      setCameras(prev => prev.map(c => 
        c.id === camera.id ? { ...c, uploading: false, error: true } : c
      ));
    }
  };

  const openCalibration = (camera) => {
    // Calculate origin offset based on previous cameras (accounting for overlap)
    let cumulativeOffsetX = 0;
    let cumulativeOffsetY = 0;
    
    // Find the index of this camera
    const cameraIndex = cameras.findIndex(c => c.id === camera.id);
    
    // Sum up the coverage of all previous cameras, subtracting overlaps
    for (let i = 0; i < cameraIndex; i++) {
      const prevCamera = cameras[i];
      if (prevCamera.calibrated && prevCamera.worldPoints) {
        // Get the width of the previous camera's coverage
        const prevWidth = prevCamera.worldPoints[1][0] - prevCamera.worldPoints[0][0];
        const prevHeight = prevCamera.worldPoints[2][1] - prevCamera.worldPoints[1][1];
        
        // Get overlap for the NEXT camera (current camera's overlap with this prev camera)
        const overlapX = camera.overlapX || 0;
        const overlapY = camera.overlapY || 0;
        
        // Add to cumulative offset (subtract overlap)
        cumulativeOffsetX += prevWidth - overlapX;
        cumulativeOffsetY += prevHeight - overlapY;
      }
    }
    
    setCalibratingCamera({
      ...camera,
      imagePoints: camera.imagePoints || [],
      worldPoints: camera.worldPoints || [],
      vehicleType: 'truck',
      vehicleBbox: null,
      originOffset: camera.originOffset || { x: cumulativeOffsetX, y: cumulativeOffsetY },
      overlapX: camera.overlapX || 0,
      overlapY: camera.overlapY || 0
    });
    setCalibrationMode('auto'); // Default to auto mode
  };

  const handleCanvasClick = (e) => {
    if (!calibratingCamera) return;

    if (calibrationMode === 'auto') {
      // In auto mode, select vehicle bounding box (2 clicks: top-left, bottom-right)
      if (!calibratingCamera.vehicleBbox || calibratingCamera.vehicleBbox.length < 2) {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        const imageX = x * scaleX;
        const imageY = y * scaleY;
        
        if (!calibratingCamera.vehicleBbox || calibratingCamera.vehicleBbox.length === 0) {
          // First click - start of bbox
          setCalibratingCamera({
            ...calibratingCamera,
            vehicleBbox: [[imageX, imageY]]
          });
        } else if (calibratingCamera.vehicleBbox.length === 1) {
          // Second click - complete bbox
          setCalibratingCamera({
            ...calibratingCamera,
            vehicleBbox: [...calibratingCamera.vehicleBbox, [imageX, imageY]]
          });
        }
      }
    } else {
      // Manual mode - original 4-point selection
      if (calibratingCamera.imagePoints.length >= 4) return;
      
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      
      const imageX = x * scaleX;
      const imageY = y * scaleY;
      
      setCalibratingCamera({
        ...calibratingCamera,
        imagePoints: [...calibratingCamera.imagePoints, [imageX, imageY]],
        worldPoints: [...calibratingCamera.worldPoints, [0, 0]]
      });
    }
  };

  const updateWorldPoint = (index, coord, value) => {
    const newWorldPoints = [...calibratingCamera.worldPoints];
    newWorldPoints[index][coord === 'x' ? 0 : 1] = parseFloat(value) || 0;
    setCalibratingCamera({ ...calibratingCamera, worldPoints: newWorldPoints });
  };

  const calculateHomography = async () => {
    try {
      let imagePoints, worldPoints;

      if (calibrationMode === 'auto') {
        // Auto calibration using vehicle size
        if (!calibratingCamera.vehicleBbox || calibratingCamera.vehicleBbox.length !== 2) {
          setError('Please select a vehicle bounding box (2 clicks)');
          return;
        }

        const [topLeft, bottomRight] = calibratingCamera.vehicleBbox;
        const vehicleSize = VEHICLE_SIZES[calibratingCamera.vehicleType];
        
        // Calculate pixel size of vehicle
        const pixelWidth = Math.abs(bottomRight[0] - topLeft[0]);
        const pixelHeight = Math.abs(bottomRight[1] - topLeft[1]);
        
        // Calculate pixels per meter
        const pixelsPerMeterX = pixelWidth / vehicleSize.width;
        const pixelsPerMeterY = pixelHeight / vehicleSize.length;
        
        // Use average scale
        const pixelsPerMeter = (pixelsPerMeterX + pixelsPerMeterY) / 2;
        
        // Get origin offset for this camera (for sequential/overlapping setup)
        const originOffsetX = calibratingCamera.originOffset?.x || 0;
        const originOffsetY = calibratingCamera.originOffset?.y || 0;
        
        // Create a calibration grid based on image size
        // Use 4 corners of the image as calibration points
        const imgWidth = canvasRef.current.width;
        const imgHeight = canvasRef.current.height;
        
        imagePoints = [
          [0, 0],
          [imgWidth, 0],
          [imgWidth, imgHeight],
          [0, imgHeight]
        ];
        
        // Apply origin offset so this camera's coordinates start from offset position
        worldPoints = [
          [originOffsetX, originOffsetY],
          [originOffsetX + imgWidth / pixelsPerMeter, originOffsetY],
          [originOffsetX + imgWidth / pixelsPerMeter, originOffsetY + imgHeight / pixelsPerMeter],
          [originOffsetX, originOffsetY + imgHeight / pixelsPerMeter]
        ];
      } else {
        // Manual calibration
        if (calibratingCamera.imagePoints.length !== 4) {
          setError('Please select 4 calibration points');
          return;
        }
        imagePoints = calibratingCamera.imagePoints;
        worldPoints = calibratingCamera.worldPoints;
      }

      const response = await fetch(`${API_BASE_URL}/api/calibrate-camera`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_points: imagePoints,
          world_points: worldPoints
        })
      });
      
      if (!response.ok) throw new Error('Calibration failed');
      
      const data = await response.json();
      
      // If this is camera 2+, try to detect overlap using feature matching
      const cameraIndex = cameras.findIndex(c => c.id === calibratingCamera.id);
      let detectedOverlap = { overlapX: calibratingCamera.overlapX || 0, overlapY: calibratingCamera.overlapY || 0 };
      
      if (cameraIndex > 0 && calibrationMode === 'auto') {
        console.log('🔍 Attempting feature-based overlap detection...');
        const prevCamera = cameras[cameraIndex - 1];
        
        if (prevCamera.calibrated && prevCamera.homographyMatrix) {
          try {
            const overlapResponse = await fetch(`${API_BASE_URL}/api/detect-overlap`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                cam1_video: prevCamera.videoPath,
                cam2_video: calibratingCamera.videoPath,
                cam1_frame_idx: 0,  // Use first frame
                cam2_frame_idx: 0,  // Use first frame (or current frame index if available)
                cam1_homography: prevCamera.homographyMatrix,
                cam2_scale: (pixelsPerMeterX + pixelsPerMeterY) / 2,
                cam1_coverage: {
                  end_x: prevCamera.worldPoints[1][0],  // Top-right X
                  end_y: prevCamera.worldPoints[2][1]   // Bottom-right Y
                }
              })
            });
            
            if (overlapResponse.ok) {
              const overlapData = await overlapResponse.json();
              console.log('✅ Overlap detection result:', overlapData);
              
              if (overlapData.overlap_detected && overlapData.confidence > 0.3) {
                // Use detected overlap
                detectedOverlap = {
                  overlapX: overlapData.overlap_x,
                  overlapY: overlapData.overlap_y
                };
                console.log(`🎯 Overlap detected: ${overlapData.overlap_x.toFixed(2)}m × ${overlapData.overlap_y.toFixed(2)}m (confidence: ${overlapData.confidence.toFixed(2)})`);
              } else {
                console.log('ℹ️ No overlap detected or low confidence');
              }
            }
          } catch (err) {
            console.log('⚠️ Feature matching failed, using manual overlap:', err.message);
          }
        }
      }
      
      // Update camera with homography
      setCameras(cameras.map(c => 
        c.id === calibratingCamera.id 
          ? { 
              ...c, 
              calibrated: true, 
              homographyMatrix: data.homography_matrix,
              imagePoints: imagePoints,
              worldPoints: worldPoints,
              calibrationMethod: calibrationMode,
              originOffset: calibratingCamera.originOffset || { x: 0, y: 0 },
              overlapX: detectedOverlap.overlapX,
              overlapY: detectedOverlap.overlapY,
              overlapDetectionMethod: (detectedOverlap.overlapX > 0 && cameraIndex > 0) ? 'feature-matching' : 'manual'
            }
          : c
      ));
      
      setCalibratingCamera(null);
    } catch (err) {
      setError(`Calibration failed: ${err.message}`);
    }
  };

  const drawCalibrationCanvas = () => {
    if (!canvasRef.current || !calibratingCamera?.previewFrame) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      if (calibrationMode === 'auto') {
        // Draw vehicle bounding box
        if (calibratingCamera.vehicleBbox && calibratingCamera.vehicleBbox.length > 0) {
          const [topLeft, bottomRight] = calibratingCamera.vehicleBbox;
          
          if (topLeft) {
            // Draw first point
            ctx.beginPath();
            ctx.arc(topLeft[0], topLeft[1], 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#3b82f6';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
          }
          
          if (bottomRight) {
            // Draw second point
            ctx.beginPath();
            ctx.arc(bottomRight[0], bottomRight[1], 8, 0, 2 * Math.PI);
            ctx.fillStyle = '#3b82f6';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw bounding box
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 3;
            ctx.setLineDash([10, 5]);
            ctx.strokeRect(
              topLeft[0],
              topLeft[1],
              bottomRight[0] - topLeft[0],
              bottomRight[1] - topLeft[1]
            );
            ctx.setLineDash([]);
            
            // Draw label
            const vehicleSize = VEHICLE_SIZES[calibratingCamera.vehicleType];
            ctx.fillStyle = 'rgba(59, 130, 246, 0.9)';
            ctx.fillRect(topLeft[0], topLeft[1] - 30, 200, 25);
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 14px sans-serif';
            ctx.fillText(`${vehicleSize.name}`, topLeft[0] + 5, topLeft[1] - 10);
          }
        }
      } else {
        // Draw manual calibration points
        calibratingCamera.imagePoints.forEach((point, idx) => {
          ctx.beginPath();
          ctx.arc(point[0], point[1], 10, 0, 2 * Math.PI);
          ctx.fillStyle = '#10b981';
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 3;
          ctx.stroke();
          
          // Draw label
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 20px sans-serif';
          ctx.fillText(`${idx + 1}`, point[0] + 15, point[1] - 10);
        });
        
        // Draw lines
        if (calibratingCamera.imagePoints.length > 1) {
          ctx.beginPath();
          ctx.moveTo(calibratingCamera.imagePoints[0][0], calibratingCamera.imagePoints[0][1]);
          for (let i = 1; i < calibratingCamera.imagePoints.length; i++) {
            ctx.lineTo(calibratingCamera.imagePoints[i][0], calibratingCamera.imagePoints[i][1]);
          }
          if (calibratingCamera.imagePoints.length === 4) {
            ctx.closePath();
          }
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }
    };
    img.src = `data:image/jpeg;base64,${calibratingCamera.previewFrame}`;
  };

  useEffect(() => {
    if (calibratingCamera) {
      drawCalibrationCanvas();
    }
  }, [calibratingCamera]);

  const processMultiCamera = async () => {
    setProcessing(true);
    setError(null);
    
    try {
      // Prepare request
      const camerasData = cameras.map(c => ({
        camera_id: c.id,
        video_path: c.videoPath,
        local_path: c.videoPath,
        homography_matrix: c.homographyMatrix
      }));
      
      const response = await fetch(`${API_BASE_URL}/api/analyze/multi-camera`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cameras: camerasData,
          polygon_points: null,
          time_window: timeWindow,
          frame_idx: 0,
          use_sahi: useSahi
        })
      });
      
      if (!response.ok) throw new Error('Analysis failed');
      
      const data = await response.json();
      setResults(data);
      
    } catch (err) {
      setError(`Processing failed: ${err.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const downloadCSV = () => {
    window.open(`${API_BASE_URL}/api/download/csv`, '_blank');
  };

  const allCamerasCalibrated = cameras.length >= 2 && cameras.every(c => c.calibrated);
  const canAnalyze = allCamerasCalibrated && !processing;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <style>{`
        .card {
          background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .btn-primary {
          background: linear-gradient(135deg, #174D38 0%, #2D7A5E 100%);
          color: white;
          padding: 0.75rem 1.5rem;
          border-radius: 0.5rem;
          font-weight: 600;
          transition: all 0.2s;
        }
        .btn-primary:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(23, 77, 56, 0.3);
        }
        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>

      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <div className="col-span-3 space-y-6">
            {/* Camera Upload */}
            <div className="card rounded-lg p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">Camera Videos</h3>
              
              <input
                type="file"
                ref={fileInputRef}
                multiple
                accept="video/*"
                className="hidden"
                onChange={(e) => addCamera(e.target.files)}
              />
              
              <div className="space-y-2 mb-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition flex items-center justify-center gap-2"
                >
                  <Upload size={14} />
                  Upload Video Files
                </button>
                
                <button
                  onClick={() => setShowLocalVideos(!showLocalVideos)}
                  className="w-full px-3 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded text-xs font-medium transition flex items-center justify-center gap-2"
                >
                  <Camera size={14} />
                  Use Local Videos
                </button>
              </div>

              {/* Local Videos Dropdown */}
              {showLocalVideos && (
                <div className="mb-4 bg-gray-50 rounded-lg border border-gray-200 max-h-60 overflow-y-auto">
                  {localVideos.length > 0 ? (
                    <div className="p-2 space-y-1">
                      {localVideos.map((video) => (
                        <button
                          key={video}
                          onClick={() => useLocalVideo(video)}
                          className="w-full text-left px-3 py-2 text-xs text-gray-700 hover:bg-white rounded transition"
                        >
                          📹 {video}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="p-4 text-center text-xs text-gray-500">
                      No local videos found
                    </div>
                  )}
                </div>
              )}

              {/* Sequential Upload Warning */}
              <div className="bg-blue-50 rounded-lg p-3 mb-4 border border-blue-200">
                <p className="text-xs text-blue-900 font-medium mb-1">⚠️ Important: Sequential Upload</p>
                <p className="text-xs text-blue-700">
                  Add cameras in order (CAM1 → CAM2 → CAM3) for automatic coordinate alignment!
                </p>
              </div>

              <div className="text-xs text-gray-500 mb-4">
                Add 2-5 videos from different camera angles
              </div>

              {/* Camera List */}
              <div className="space-y-2">
                {cameras.map((camera) => (
                  <div key={camera.id} className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Camera size={14} className="text-[#174D38]" />
                        <span className="text-xs font-semibold text-gray-900">{camera.id}</span>
                      </div>
                      <button
                        onClick={() => removeCamera(camera.id)}
                        className="text-red-500 hover:text-red-700"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    <div className="text-xs text-gray-600 mb-2 truncate">{camera.name}</div>
                    
                    {/* Show overlap info if calibrated and has overlap */}
                    {camera.calibrated && (camera.overlapX > 0 || camera.overlapY > 0) && (
                      <div className={`text-xs rounded px-2 py-1 mb-2 ${
                        camera.overlapDetectionMethod === 'feature-matching' 
                          ? 'bg-green-50 text-green-700' 
                          : 'bg-purple-50 text-purple-700'
                      }`}>
                        {camera.overlapDetectionMethod === 'feature-matching' ? '🎯' : '🔗'} Overlap: {camera.overlapX.toFixed(1)}m × {camera.overlapY.toFixed(1)}m
                        {camera.overlapDetectionMethod === 'feature-matching' && ' (auto)'}
                      </div>
                    )}
                    
                    <div className="flex items-center justify-between">
                      <div className="text-xs">
                        {camera.uploading && (
                          <span className="text-blue-600 flex items-center gap-1">
                            <Loader size={12} className="animate-spin" /> Uploading...
                          </span>
                        )}
                        {camera.videoPath && !camera.uploading && (
                          <span className="text-gray-500">✓ Uploaded</span>
                        )}
                        {camera.error && (
                          <span className="text-red-500">✗ Failed</span>
                        )}
                      </div>
                      
                      {camera.videoPath && (
                        <button
                          onClick={() => openCalibration(camera)}
                          className={`text-xs px-2 py-1 rounded font-medium ${
                            camera.calibrated
                              ? 'bg-green-100 text-green-700'
                              : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                          }`}
                        >
                          {camera.calibrated ? (
                            <span className="flex items-center gap-1">
                              <Check size={12} /> Calibrated
                            </span>
                          ) : (
                            'Calibrate'
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                ))}
                
                {cameras.length === 0 && (
                  <div className="text-center py-8 text-gray-400">
                    <Camera size={24} className="mx-auto mb-2 opacity-50" />
                    <p className="text-xs">No cameras added</p>
                  </div>
                )}
              </div>
            </div>

            {/* Analysis Settings */}
            {cameras.length >= 2 && (
              <div className="card rounded-lg p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">Analysis Settings</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-xs font-medium text-gray-900 block mb-2">
                      Time Window (seconds)
                    </label>
                    <input
                      type="number"
                      value={timeWindow}
                      onChange={(e) => setTimeWindow(parseInt(e.target.value))}
                      min="10"
                      max="120"
                      className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded text-sm focus:ring-2 focus:ring-[#174D38]/20 focus:border-[#174D38] outline-none"
                    />
                    <div className="text-xs text-gray-500 mt-1">
                      Duration to track vehicles (10-120s)
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-medium text-gray-900">SAHI Detection</div>
                      <div className="text-xs text-gray-500 mt-0.5">Enhanced accuracy</div>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={useSahi}
                        onChange={(e) => setUseSahi(e.target.checked)}
                        className="sr-only peer"
                      />
                      <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-[#174D38]/20 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#174D38]"></div>
                    </label>
                  </div>

                  <button
                    onClick={processMultiCamera}
                    disabled={!canAnalyze}
                    className="w-full btn-primary flex items-center justify-center gap-2"
                  >
                    {processing ? (
                      <><Loader className="w-4 h-4 animate-spin" /> Analyzing...</>
                    ) : (
                      <><Car className="w-4 h-4" /> Run Multi-Camera Analysis</>
                    )}
                  </button>

                  {!allCamerasCalibrated && cameras.length >= 2 && (
                    <div className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded p-2">
                      ⚠️ All cameras must be calibrated before analysis
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Main Content */}
          <div className="col-span-9 space-y-6">
            {/* Error Display */}
            {error && (
              <div className="card rounded-lg p-4 border-l-4 border-red-500 bg-red-50">
                <div className="flex items-start gap-3">
                  <AlertCircle className="text-red-500 flex-shrink-0 mt-0.5" size={18} />
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-red-900">Error</p>
                    <p className="text-xs text-red-700 mt-1">{error}</p>
                  </div>
                  <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700">
                    <X size={16} />
                  </button>
                </div>
              </div>
            )}

            {/* Getting Started */}
            {cameras.length === 0 && (
              <div className="card rounded-lg p-8 text-center">
                <Camera size={48} className="mx-auto mb-4 text-gray-300" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Multi-Camera Vehicle Analytics</h3>
                <p className="text-sm text-gray-600 mb-6">
                  Analyze vehicles across multiple camera views with real-world coordinate fusion
                </p>
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="btn-primary inline-flex items-center gap-2"
                  >
                    <Upload size={16} />
                    Upload Videos
                  </button>
                  <button
                    onClick={() => setShowLocalVideos(true)}
                    className="px-6 py-3 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg font-semibold transition inline-flex items-center gap-2"
                  >
                    <Camera size={16} />
                    Use Local Videos
                  </button>
                </div>
              </div>
            )}

            {/* Camera Grid Preview */}
            {cameras.length > 0 && !results && (
              <div className="card rounded-lg p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">Camera Preview</h3>
                <div className={`grid gap-4 ${cameras.length === 1 ? 'grid-cols-1' : cameras.length === 2 ? 'grid-cols-2' : 'grid-cols-3'}`}>
                  {cameras.map((camera) => (
                    <div key={camera.id} className="bg-gray-900 rounded-lg overflow-hidden">
                      <div className="aspect-video relative">
                        {camera.previewFrame ? (
                          <img
                            src={`data:image/jpeg;base64,${camera.previewFrame}`}
                            alt={camera.id}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="flex items-center justify-center h-full text-gray-400">
                            {camera.uploading ? (
                              <Loader className="animate-spin" size={32} />
                            ) : (
                              <Camera size={32} />
                            )}
                          </div>
                        )}
                        <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-semibold">
                          {camera.id}
                        </div>
                        {camera.calibrated && (
                          <div className="absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded font-semibold flex items-center gap-1">
                            <Check size={12} /> Calibrated
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Results */}
            {results && (
              <div className="space-y-6">
                {/* Stats Cards */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="card rounded-lg p-5 border-l-4 border-blue-500">
                    <div className="text-xs font-medium text-gray-600 mb-1">Total Vehicles</div>
                    <div className="text-3xl font-bold text-gray-900">{results.total_vehicles}</div>
                  </div>
                  <div className="card rounded-lg p-5 border-l-4 border-green-500">
                    <div className="text-xs font-medium text-gray-600 mb-1">Cross-Camera Tracks</div>
                    <div className="text-3xl font-bold text-gray-900">{results.cross_camera_tracks}</div>
                  </div>
                  <div className="card rounded-lg p-5 border-l-4 border-purple-500">
                    <div className="text-xs font-medium text-gray-600 mb-1">Cameras Used</div>
                    <div className="text-3xl font-bold text-gray-900">{cameras.length}</div>
                  </div>
                </div>

                {/* Camera Results */}
                <div className="card rounded-lg p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-4">Camera-wise Detection Results</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {results.camera_results?.map((camResult) => (
                      <div key={camResult.camera_id} className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Camera size={16} className="text-[#174D38]" />
                            <span className="font-semibold text-gray-900">{camResult.camera_id}</span>
                          </div>
                          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded font-medium">
                            {camResult.vehicles_detected} vehicles
                          </span>
                        </div>
                        {camResult.annotated_frame && (
                          <img 
                            src={`data:image/jpeg;base64,${camResult.annotated_frame}`} 
                            alt={`${camResult.camera_id} detections`}
                            className="rounded-lg w-full border border-gray-300"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="card rounded-lg p-5">
                  <div className="flex gap-4">
                    <button
                      onClick={downloadCSV}
                      className="flex-1 btn-primary flex items-center justify-center gap-2"
                    >
                      <Download size={18} /> Download Fused CSV (Real-World Coordinates)
                    </button>
                    <button
                      onClick={() => {
                        setResults(null);
                        setCameras([]);
                      }}
                      className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg font-semibold transition"
                    >
                      New Analysis
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Calibration Modal */}
      {calibratingCamera && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-6 z-50">
          <div className="bg-white rounded-xl max-w-6xl w-full max-h-[90vh] overflow-auto shadow-2xl">
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
              <div>
                <h3 className="text-lg font-bold text-gray-900">Calibrate {calibratingCamera.id}</h3>
                <p className="text-xs text-gray-600 mt-0.5">
                  {calibrationMode === 'auto' 
                    ? 'Select a vehicle by clicking top-left and bottom-right corners' 
                    : 'Click 4 points that form a rectangle in the real world'}
                </p>
              </div>
              <button 
                onClick={() => setCalibratingCamera(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={24} />
              </button>
            </div>

            <div className="p-6">
              {/* Calibration Mode Selector */}
              <div className="mb-6 bg-gray-50 rounded-lg p-4">
                <label className="text-sm font-semibold text-gray-900 block mb-3">Calibration Method:</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => {
                      setCalibrationMode('auto');
                      setCalibratingCamera({
                        ...calibratingCamera,
                        imagePoints: [],
                        worldPoints: [],
                        vehicleBbox: null
                      });
                    }}
                    className={`px-4 py-3 rounded-lg font-semibold text-sm transition ${
                      calibrationMode === 'auto'
                        ? 'bg-blue-500 text-white shadow-lg'
                        : 'bg-white text-gray-700 border-2 border-gray-200 hover:border-blue-300'
                    }`}
                  >
                    <div className="flex items-center justify-center gap-2 mb-1">
                      <Camera size={18} />
                      Auto (Recommended)
                    </div>
                    <div className="text-xs opacity-90">Select a vehicle for scale</div>
                  </button>
                  <button
                    onClick={() => {
                      setCalibrationMode('manual');
                      setCalibratingCamera({
                        ...calibratingCamera,
                        imagePoints: [],
                        worldPoints: [],
                        vehicleBbox: null
                      });
                    }}
                    className={`px-4 py-3 rounded-lg font-semibold text-sm transition ${
                      calibrationMode === 'manual'
                        ? 'bg-green-500 text-white shadow-lg'
                        : 'bg-white text-gray-700 border-2 border-gray-200 hover:border-green-300'
                    }`}
                  >
                    <div className="flex items-center justify-center gap-2 mb-1">
                      <MapPin size={18} />
                      Manual
                    </div>
                    <div className="text-xs opacity-90">Enter exact coordinates</div>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                {/* Left: Canvas */}
                <div>
                  {calibrationMode === 'auto' ? (
                    <div className="bg-blue-50 rounded-lg p-3 mb-3 border border-blue-200">
                      <p className="text-xs text-blue-900 font-medium mb-2">📐 Auto Calibration:</p>
                      <ol className="text-xs text-blue-800 space-y-1 list-decimal list-inside">
                        <li>Select vehicle type below</li>
                        <li>Click <strong>top-left corner</strong> of a vehicle</li>
                        <li>Click <strong>bottom-right corner</strong> of the same vehicle</li>
                        <li>System calculates scale automatically!</li>
                      </ol>
                    </div>
                  ) : (
                    <div className="bg-green-50 rounded-lg p-3 mb-3 border border-green-200">
                      <p className="text-xs text-green-900 font-medium mb-2">📍 Manual Calibration:</p>
                      <ol className="text-xs text-green-800 space-y-1 list-decimal list-inside">
                        <li>Click 4 points forming a rectangle (e.g., road corners)</li>
                        <li>Enter real-world coordinates in meters</li>
                        <li>Click "Calculate Homography"</li>
                      </ol>
                    </div>
                  )}
                  
                  <div className="bg-black rounded-lg overflow-hidden border-2 border-gray-300">
                    <canvas
                      ref={canvasRef}
                      onClick={handleCanvasClick}
                      className="w-full h-auto cursor-crosshair"
                    />
                  </div>
                  
                  <div className="mt-3 flex items-center justify-between text-xs">
                    <span className="text-gray-600">
                      {calibrationMode === 'auto' ? (
                        <>Points: <span className="font-semibold text-blue-600">
                          {calibratingCamera.vehicleBbox?.length || 0}/2
                        </span></>
                      ) : (
                        <>Points: <span className="font-semibold text-green-600">
                          {calibratingCamera.imagePoints?.length || 0}/4
                        </span></>
                      )}
                    </span>
                    {((calibrationMode === 'auto' && calibratingCamera.vehicleBbox?.length > 0) ||
                      (calibrationMode === 'manual' && calibratingCamera.imagePoints?.length > 0)) && (
                      <button
                        onClick={() => setCalibratingCamera({
                          ...calibratingCamera,
                          imagePoints: [],
                          worldPoints: [],
                          vehicleBbox: null
                        })}
                        className="text-red-600 hover:text-red-700 font-medium"
                      >
                        Reset
                      </button>
                    )}
                  </div>
                </div>

                {/* Right: Parameters */}
                <div>
                  {calibrationMode === 'auto' ? (
                    /* Auto Calibration UI */
                    <div>
                      {/* Overlap Distance (for overlapping cameras) */}
                      {cameras.findIndex(c => c.id === calibratingCamera.id) > 0 && (
                        <div className="bg-purple-50 rounded-lg p-3 mb-4 border border-purple-200">
                          <p className="text-xs text-purple-900 font-medium mb-2">🔗 Camera Overlap (Optional)</p>
                          <p className="text-xs text-purple-700 mb-3">
                            The system will attempt to <strong>auto-detect overlap</strong> using feature matching. 
                            You can also manually specify overlap distance if needed.
                          </p>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <label className="text-xs text-gray-700 block mb-1">Overlap X (meters)</label>
                              <input
                                type="number"
                                step="1"
                                value={calibratingCamera.overlapX || 0}
                                onChange={(e) => setCalibratingCamera({
                                  ...calibratingCamera,
                                  overlapX: parseFloat(e.target.value) || 0
                                })}
                                className="w-full px-2 py-1.5 bg-white border border-purple-300 rounded text-sm focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500 outline-none"
                                placeholder="0 (auto-detect)"
                              />
                            </div>
                            <div>
                              <label className="text-xs text-gray-700 block mb-1">Overlap Y (meters)</label>
                              <input
                                type="number"
                                step="1"
                                value={calibratingCamera.overlapY || 0}
                                onChange={(e) => setCalibratingCamera({
                                  ...calibratingCamera,
                                  overlapY: parseFloat(e.target.value) || 0
                                })}
                                className="w-full px-2 py-1.5 bg-white border border-purple-300 rounded text-sm focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500 outline-none"
                                placeholder="0 (auto-detect)"
                              />
                            </div>
                          </div>
                          <p className="text-xs text-green-600 mt-2">
                            ✨ Leave as 0 to enable automatic detection via feature matching
                          </p>
                        </div>
                      )}
                      
                      {/* Origin Offset (for sequential/overlapping cameras) */}
                      <div className="bg-amber-50 rounded-lg p-3 mb-4 border border-amber-200">
                        <p className="text-xs text-amber-900 font-medium mb-2">🗺️ Coordinate System Origin (Auto-Calculated)</p>
                        <p className="text-xs text-amber-700 mb-3">
                          Origin offset is automatically calculated based on previous cameras' coverage.
                        </p>
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="text-xs text-gray-700 block mb-1">Offset X (meters)</label>
                            <input
                              type="number"
                              step="1"
                              value={calibratingCamera.originOffset?.x || 0}
                              onChange={(e) => setCalibratingCamera({
                                ...calibratingCamera,
                                originOffset: {
                                  ...calibratingCamera.originOffset,
                                  x: parseFloat(e.target.value) || 0
                                }
                              })}
                              className="w-full px-2 py-1.5 bg-white border border-amber-300 rounded text-sm focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 outline-none"
                              title="Auto-calculated, but you can override if needed"
                            />
                          </div>
                          <div>
                            <label className="text-xs text-gray-700 block mb-1">Offset Y (meters)</label>
                            <input
                              type="number"
                              step="1"
                              value={calibratingCamera.originOffset?.y || 0}
                              onChange={(e) => setCalibratingCamera({
                                ...calibratingCamera,
                                originOffset: {
                                  ...calibratingCamera.originOffset,
                                  y: parseFloat(e.target.value) || 0
                                }
                              })}
                              className="w-full px-2 py-1.5 bg-white border border-amber-300 rounded text-sm focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 outline-none"
                              title="Auto-calculated, but you can override if needed"
                            />
                          </div>
                        </div>
                        <p className="text-xs text-amber-600 mt-2">
                          💡 This is calculated from previous cameras. Calibrate cameras in sequential order!
                        </p>
                      </div>
                      
                      <div className="bg-gray-100 rounded-lg p-3 mb-4">
                        <p className="text-xs text-gray-700 font-medium">Vehicle Reference</p>
                        <p className="text-xs text-gray-600 mt-1">
                          Select the type of vehicle you'll mark on the image
                        </p>
                      </div>
                      
                      <label className="text-xs font-medium text-gray-900 block mb-2">
                        Vehicle Type:
                      </label>
                      <select
                        value={calibratingCamera.vehicleType || 'truck'}
                        onChange={(e) => setCalibratingCamera({
                          ...calibratingCamera,
                          vehicleType: e.target.value
                        })}
                        className="w-full px-3 py-2.5 bg-white border-2 border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none mb-4"
                      >
                        {Object.entries(VEHICLE_SIZES).map(([key, value]) => (
                          <option key={key} value={key}>
                            {value.name}
                          </option>
                        ))}
                      </select>

                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                        <p className="text-xs font-semibold text-blue-900 mb-2">Selected Vehicle:</p>
                        <div className="text-sm text-blue-800">
                          <div className="flex justify-between mb-1">
                            <span>Length:</span>
                            <span className="font-bold">
                              {VEHICLE_SIZES[calibratingCamera.vehicleType || 'truck'].length}m
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Width:</span>
                            <span className="font-bold">
                              {VEHICLE_SIZES[calibratingCamera.vehicleType || 'truck'].width}m
                            </span>
                          </div>
                        </div>
                      </div>

                      {calibratingCamera.vehicleBbox?.length === 2 && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                          <p className="text-xs font-semibold text-green-900 mb-2">✓ Vehicle Selected</p>
                          <p className="text-xs text-green-700">
                            Scale will be calculated automatically based on the selected vehicle size.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    /* Manual Calibration UI */
                    <div>
                      <div className="bg-gray-100 rounded-lg p-3 mb-3">
                        <p className="text-xs text-gray-700 font-medium">Real-World Coordinates (meters)</p>
                        <p className="text-xs text-gray-600 mt-1">Example: 50m × 30m road section</p>
                      </div>
                      
                      <div className="space-y-3 mb-4">
                        {[0, 1, 2, 3].map((idx) => (
                          <div key={idx} className={`rounded-lg p-3 border-2 ${
                            calibratingCamera.imagePoints[idx] 
                              ? 'bg-green-50 border-green-300' 
                              : 'bg-gray-50 border-gray-200'
                          }`}>
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs font-semibold text-gray-900">Point {idx + 1}</span>
                              {calibratingCamera.imagePoints[idx] && (
                                <span className="text-xs text-green-600 flex items-center gap-1">
                                  <Check size={12} /> Selected
                                </span>
                              )}
                            </div>
                            {calibratingCamera.imagePoints[idx] ? (
                              <div className="grid grid-cols-2 gap-2">
                                <div>
                                  <label className="text-xs text-gray-600 block mb-1">X (meters)</label>
                                  <input
                                    type="number"
                                    step="0.1"
                                    value={calibratingCamera.worldPoints[idx]?.[0] || 0}
                                    onChange={(e) => updateWorldPoint(idx, 'x', e.target.value)}
                                    className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded text-sm focus:ring-2 focus:ring-[#174D38]/20 focus:border-[#174D38] outline-none"
                                  />
                                </div>
                                <div>
                                  <label className="text-xs text-gray-600 block mb-1">Y (meters)</label>
                                  <input
                                    type="number"
                                    step="0.1"
                                    value={calibratingCamera.worldPoints[idx]?.[1] || 0}
                                    onChange={(e) => updateWorldPoint(idx, 'y', e.target.value)}
                                    className="w-full px-2 py-1.5 bg-white border border-gray-300 rounded text-sm focus:ring-2 focus:ring-[#174D38]/20 focus:border-[#174D38] outline-none"
                                  />
                                </div>
                              </div>
                            ) : (
                              <p className="text-xs text-gray-500 italic">Click on image to place this point</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <button
                    onClick={calculateHomography}
                    disabled={
                      (calibrationMode === 'auto' && (!calibratingCamera.vehicleBbox || calibratingCamera.vehicleBbox.length !== 2)) ||
                      (calibrationMode === 'manual' && calibratingCamera.imagePoints?.length !== 4)
                    }
                    className="w-full btn-primary flex items-center justify-center gap-2"
                  >
                    <MapPin size={16} />
                    Calculate Homography Matrix
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
