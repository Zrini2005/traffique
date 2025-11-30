import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, CheckCircle, Loader, Car, AlertCircle, Download } from 'lucide-react';

const AnalysisPage = () => {
  const [video, setVideo] = useState(null);
  const [videoMetadata, setVideoMetadata] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frameImage, setFrameImage] = useState(null);
  const [polygon, setPolygon] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('quick');
  const [useSahi, setUseSahi] = useState(true);
  const [localVideos, setLocalVideos] = useState([]);
  const [showLocalVideos, setShowLocalVideos] = useState(false);
  
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadLocalVideos();
  }, []);

  const loadLocalVideos = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/local-videos');
      const data = await response.json();
      setLocalVideos(data.videos || []);
    } catch (error) {
      console.error('Failed to load local videos:', error);
    }
  };

  const useLocalVideo = async (filename) => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:5000/api/use-local/${filename}`, {
        method: 'POST',
      });
      const data = await response.json();
      
      if (data.success) {
        setVideo(data.filename);
        setVideoMetadata(data.metadata);
        setCurrentFrame(Math.floor(data.metadata.total_frames / 2));
        await loadFrame(data.filename, Math.floor(data.metadata.total_frames / 2));
        setShowLocalVideos(false);
      }
    } catch (error) {
      console.error('Failed to use local video:', error);
      alert('Failed to load local video');
    } finally {
      setLoading(false);
    }
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (data.success) {
        setVideo(data.filename);
        setVideoMetadata(data.metadata);
        setCurrentFrame(Math.floor(data.metadata.total_frames / 2));
        await loadFrame(data.filename, Math.floor(data.metadata.total_frames / 2));
      }
    } catch (error) {
      console.error('Upload error:', error);
      alert('Failed to upload video');
    } finally {
      setLoading(false);
    }
  };

  const loadFrame = async (filename, frameIdx) => {
    try {
      const response = await fetch(`http://localhost:5000/api/frame/${filename}/${frameIdx}`);
      const data = await response.json();
      
      if (data.success) {
        setFrameImage(data.frame);
        drawCanvas(data.frame);
      }
    } catch (error) {
      console.error('Frame load error:', error);
    }
  };

  const drawCanvas = (imageData) => {
    const canvas = canvasRef.current;
    if (!canvas || !imageData) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      
      if (polygon.length > 0) {
        ctx.strokeStyle = '#174D38';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(23, 77, 56, 0.15)';
        
        ctx.beginPath();
        ctx.moveTo(polygon[0].x, polygon[0].y);
        polygon.forEach(point => ctx.lineTo(point.x, point.y));
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        polygon.forEach((point, idx) => {
          ctx.fillStyle = '#174D38';
          ctx.beginPath();
          ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
          ctx.fill();
          
          ctx.fillStyle = 'white';
          ctx.font = '11px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(idx + 1, point.x, point.y);
        });
      }
    };
    
    img.onerror = () => console.error('Failed to load image');
    const cleanImageData = imageData.replace(/\s/g, '');
    img.src = `data:image/jpeg;base64,${cleanImageData}`;
  };

  const handleCanvasClick = (event) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    setPolygon([...polygon, { x: Math.round(x), y: Math.round(y) }]);
  };

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === ' ' && isDrawing && polygon.length >= 3) {
        e.preventDefault();
        setIsDrawing(false);
      } else if (e.key === 'Escape') {
        setPolygon([]);
        setIsDrawing(false);
        if (frameImage) drawCanvas(frameImage);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isDrawing, polygon]);

  useEffect(() => {
    if (frameImage) drawCanvas(frameImage);
  }, [polygon]);

  const runAnalysis = async () => {
    if (!video || polygon.length < 3) {
      alert('Please upload a video and draw a region of interest');
      return;
    }

    setLoading(true);
    
    try {
      const endpoint = mode === 'quick' 
        ? 'http://localhost:5000/api/analyze/quick'
        : 'http://localhost:5000/api/analyze/full';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: video,
          frame_idx: currentFrame,
          polygon: polygon,
          use_sahi: useSahi,
          confidence: 0.20,
          time_window: 5, // For full mode
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setResults(data);
      } else {
        alert('Analysis failed: ' + data.error);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Failed to run analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">Traffic Analysis System</h1>
              <p className="text-sm text-gray-500 mt-0.5">AI-powered vehicle detection and tracking</p>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-xs px-2.5 py-1 bg-green-100 text-green-700 rounded-full font-medium">
                System Active
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <div className="col-span-3 space-y-6">
            {/* Video Source */}
            <div className="card rounded-lg p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">Video Source</h3>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                className="hidden"
              />
              
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                className="w-full btn-primary flex items-center justify-center gap-2 mb-3"
              >
                {loading && !video ? (
                  <><Loader className="w-4 h-4 animate-spin" /> Uploading...</>
                ) : (
                  <><Upload className="w-4 h-4" /> Upload File</>
                )}
              </button>

              <button
                onClick={() => setShowLocalVideos(!showLocalVideos)}
                className="w-full btn-secondary flex items-center justify-center gap-2 text-sm"
              >
                Local Videos ({localVideos.length})
              </button>

              {showLocalVideos && localVideos.length > 0 && (
                <div className="mt-3 max-h-48 overflow-y-auto space-y-2">
                  {localVideos.map((vid) => (
                    <button
                      key={vid.filename}
                      onClick={() => useLocalVideo(vid.filename)}
                      className="w-full p-2.5 bg-gray-50 hover:bg-gray-100 border border-gray-200 hover:border-gray-300 rounded text-left text-xs transition"
                    >
                      <div className="font-medium text-gray-900 truncate">{vid.filename}</div>
                      <div className="text-gray-500 mt-0.5">{vid.size_mb.toFixed(1)} MB • {vid.width}x{vid.height}</div>
                    </button>
                  ))}
                </div>
              )}

              {videoMetadata && (
                <div className="mt-4 pt-4 border-t border-gray-200 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Frames</span>
                    <span className="font-medium text-gray-900">{videoMetadata.total_frames.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">FPS</span>
                    <span className="font-medium text-gray-900">{videoMetadata.fps}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Resolution</span>
                    <span className="font-medium text-gray-900">{videoMetadata.width}x{videoMetadata.height}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Frame Control */}
            {video && (
              <div className="card rounded-lg p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">Frame Selection</h3>
                
                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-gray-600 mb-2 block">
                      Frame {currentFrame.toLocaleString()} / {videoMetadata?.total_frames.toLocaleString() || 0}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max={videoMetadata?.total_frames || 0}
                      value={currentFrame}
                      onChange={(e) => {
                        const frame = parseInt(e.target.value);
                        setCurrentFrame(frame);
                        loadFrame(video, frame);
                      }}
                      className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#174D38]"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => {
                        const newFrame = Math.max(0, currentFrame - 100);
                        setCurrentFrame(newFrame);
                        loadFrame(video, newFrame);
                      }}
                      className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition"
                    >
                      -100
                    </button>
                    <button
                      onClick={() => {
                        const newFrame = Math.min(videoMetadata.total_frames, currentFrame + 100);
                        setCurrentFrame(newFrame);
                        loadFrame(video, newFrame);
                      }}
                      className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition"
                    >
                      +100
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ROI Drawing */}
            {video && (
              <div className="card rounded-lg p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">Region of Interest</h3>
                
                {!isDrawing && polygon.length === 0 && (
                  <button
                    onClick={() => { setPolygon([]); setIsDrawing(true); }}
                    className="w-full btn-primary flex items-center justify-center gap-2"
                  >
                    <Play className="w-4 h-4" /> Draw Region
                  </button>
                )}

                {isDrawing && (
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded text-xs text-blue-900">
                      Click on frame to add points. Press <strong>Space</strong> when done.
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => { if(polygon.length >= 3) setIsDrawing(false); }}
                        className="px-3 py-2 bg-[#174D38] hover:bg-[#0d3025] text-white rounded text-xs font-medium transition"
                      >
                        <CheckCircle className="w-3.5 h-3.5 inline mr-1" /> Finish
                      </button>
                      <button
                        onClick={() => { setPolygon([]); setIsDrawing(false); if (frameImage) drawCanvas(frameImage); }}
                        className="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {!isDrawing && polygon.length > 0 && (
                  <div className="space-y-3">
                    <div className="p-3 bg-green-50 border border-green-200 rounded text-xs text-green-900 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      <span>{polygon.length} points defined</span>
                    </div>
                    <button
                      onClick={() => { setPolygon([]); setIsDrawing(true); if (frameImage) drawCanvas(frameImage); }}
                      className="w-full px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded text-xs font-medium transition"
                    >
                      Redraw Region
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Analysis Settings */}
            {video && polygon.length > 0 && (
              <div className="card rounded-lg p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">Analysis Settings</h3>
                
                <div className="space-y-4">
                  {/* Analysis Mode Selector */}
                  <div>
                    <div className="text-xs font-medium text-gray-900 mb-2">Analysis Mode</div>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setMode('quick')}
                        className={`px-3 py-2 text-xs font-medium rounded transition ${
                          mode === 'quick'
                            ? 'bg-[#174D38] text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        Quick Mode
                      </button>
                      <button
                        onClick={() => setMode('full')}
                        className={`px-3 py-2 text-xs font-medium rounded transition ${
                          mode === 'full'
                            ? 'bg-[#174D38] text-white'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        Full Mode
                      </button>
                    </div>
                    <div className="text-xs text-gray-500 mt-1.5">
                      {mode === 'quick' 
                        ? 'Single frame detection (~6s)' 
                        : 'Multi-frame tracking with velocity & CSV export (~60s)'}
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
                    onClick={runAnalysis}
                    disabled={loading}
                    className="w-full btn-primary flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <><Loader className="w-4 h-4 animate-spin" /> Analyzing...</>
                    ) : (
                      <><Car className="w-4 h-4" /> Run Analysis</>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Main Content */}
          <div className="col-span-9 space-y-6">
            {/* Video Frame */}
            <div className="card rounded-lg p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">Video Frame</h3>
              
              <div className="relative bg-gray-900 rounded overflow-hidden">
                {!video ? (
                  <div className="aspect-video flex items-center justify-center text-gray-400">
                    <div className="text-center">
                      <Upload className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm">Upload or select a video to begin</p>
                    </div>
                  </div>
                ) : (
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    className={`w-full h-auto ${isDrawing ? 'cursor-crosshair' : 'cursor-default'}`}
                    style={{ maxHeight: '600px' }}
                  />
                )}
              </div>
            </div>

            {/* Results */}
            {results && (
              <div className="card rounded-lg p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-4">
                  {mode === 'quick' ? 'Detection Results' : 'Tracking Analytics'}
                </h3>

                {mode === 'quick' ? (
                  <>
                    {/* Quick Mode Results */}
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      <div className="stat-card">
                        <div className="text-2xl font-semibold text-gray-900">{results.total_detections || 0}</div>
                        <div className="text-xs text-gray-500 mt-1">Total Detections</div>
                      </div>
                      <div className="stat-card">
                        <div className="text-2xl font-semibold text-[#174D38]">{results.polygon_detections || 0}</div>
                        <div className="text-xs text-gray-500 mt-1">In Region of Interest</div>
                      </div>
                    </div>

                    {results.visualization && (
                      <div className="mb-6">
                        <h4 className="text-xs font-semibold text-gray-700 mb-3">Annotated Frame</h4>
                        <div className="relative bg-black rounded overflow-hidden border border-gray-200">
                          <img 
                            src={`data:image/jpeg;base64,${results.visualization}`}
                            alt="Detection visualization"
                            className="w-full h-auto"
                          />
                        </div>
                        <div className="mt-2 text-xs text-gray-500 flex items-center gap-4">
                          <span className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-[#174D38] rounded"></div>
                            In ROI
                          </span>
                          <span className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-gray-400 rounded"></div>
                            Outside ROI
                          </span>
                        </div>
                      </div>
                    )}

                    {results.vehicles && results.vehicles.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-700 mb-3">Detected Vehicles ({results.vehicles.length})</h4>
                        <div className="max-h-80 overflow-y-auto space-y-2">
                          {results.vehicles.map((vehicle, idx) => (
                            <div key={idx} className="p-3 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded transition">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-medium text-gray-900">Vehicle #{vehicle.id || idx + 1}</span>
                                <span className="badge bg-[#174D38] text-white">{vehicle.class}</span>
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                                <div>Confidence: <span className="font-medium text-gray-900">{(vehicle.confidence * 100).toFixed(1)}%</span></div>
                                <div>Position: <span className="font-medium text-gray-900">({vehicle.center_x}, {vehicle.center_y})</span></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    {/* Full Mode Results */}
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="stat-card">
                        <div className="text-2xl font-semibold text-gray-900">{results.total_tracks || 0}</div>
                        <div className="text-xs text-gray-500 mt-1">Tracked Vehicles</div>
                      </div>
                      <div className="stat-card">
                        <div className="text-2xl font-semibold text-[#174D38]">{results.analytics?.length || 0}</div>
                        <div className="text-xs text-gray-500 mt-1">With Trajectories</div>
                      </div>
                      <div className="stat-card">
                        <div className="text-xs font-medium text-blue-600">CSV Ready</div>
                        <div className="text-xs text-gray-500 mt-1">
                          <a 
                            href="http://localhost:5000/api/download/csv" 
                            className="text-[#174D38] hover:underline"
                            download
                          >
                            Download
                          </a>
                        </div>
                      </div>
                    </div>

                    {results.annotated_image && (
                      <div className="mb-6">
                        <h4 className="text-xs font-semibold text-gray-700 mb-3">Trajectory Visualization</h4>
                        <div className="relative bg-black rounded overflow-hidden border border-gray-200">
                          <img 
                            src={`data:image/jpeg;base64,${results.annotated_image}`}
                            alt="Trajectory visualization"
                            className="w-full h-auto"
                          />
                        </div>
                        <div className="mt-2 text-xs text-gray-500 flex items-center gap-4">
                          <span className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-orange-500 rounded"></div>
                            Vehicle Trajectories
                          </span>
                          <span className="flex items-center gap-1.5">
                            <div className="w-3 h-3 bg-[#174D38] rounded-full"></div>
                            Current Position
                          </span>
                        </div>
                      </div>
                    )}

                    {results.analytics && results.analytics.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-700 mb-3">Vehicle Trajectories ({results.analytics.length})</h4>
                        <div className="max-h-96 overflow-y-auto space-y-2">
                          {results.analytics.map((vehicle, idx) => (
                            <div key={idx} className="p-3 bg-gray-50 hover:bg-gray-100 border border-gray-200 rounded transition">
                              <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-medium text-gray-900">Vehicle #{vehicle.vehicle_id}</span>
                                <span className="badge bg-[#174D38] text-white">{vehicle.class}</span>
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                                <div>Time in scene: <span className="font-medium text-gray-900">{vehicle.time_in_scene}s</span></div>
                                <div>Frames: <span className="font-medium text-gray-900">{vehicle.num_frames}</span></div>
                                <div>Avg Velocity: <span className="font-medium text-gray-900">{vehicle.avg_velocity_px_per_sec?.toFixed(1)} px/s</span></div>
                                <div>Distance: <span className="font-medium text-gray-900">{vehicle.total_distance_px?.toFixed(0)} px</span></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;
