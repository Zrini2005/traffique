# GPS Calibration & Video Sequencing Guide

## 📍 Converting to Latitude/Longitude (GPS Coordinates)

### Current System: Local Coordinates (Meters)
The system currently uses a **local coordinate system** where:
- Origin (0, 0) is an arbitrary reference point
- All coordinates are in **meters** (x, y)
- Works great for spatial fusion but doesn't give GPS locations

### To Get GPS Coordinates: Add Reference Point

You need to provide **at least one GPS reference point** during calibration.

---

## 🌍 Method 1: Single GPS Reference Point (Recommended)

### How It Works:
1. Choose a reference point in your scene (e.g., road intersection, building corner)
2. Get its GPS coordinates using Google Maps or GPS device
3. System converts local meters → GPS coordinates using simple offset

### Implementation:

#### Step 1: Add GPS Reference to Calibration Modal
```javascript
// In MultiCameraPage.jsx, add to calibration state:
const [gpsReference, setGpsReference] = useState({
  latitude: null,   // e.g., 13.0827° N
  longitude: null,  // e.g., 80.2707° E
  localX: 0,        // Origin point in local coordinate system
  localY: 0
});
```

#### Step 2: Modify Backend to Accept GPS Reference
```python
# In api_server.py, update /api/calibrate-camera endpoint:
@app.route('/api/calibrate-camera', methods=['POST'])
def calibrate_camera():
    data = request.json
    image_points = np.array(data['image_points'], dtype=np.float32)
    world_points = np.array(data['world_points'], dtype=np.float32)
    
    # NEW: Accept optional GPS reference
    gps_ref = data.get('gps_reference', None)  # {lat, lon, local_x, local_y}
    
    # Calculate homography
    H, status = cv2.findHomography(image_points, world_points, method=0)
    
    return jsonify({
        'success': True,
        'homography_matrix': H.tolist(),
        'gps_reference': gps_ref  # Store for later use
    })
```

#### Step 3: Convert Local Coords → GPS During Analysis
```python
def local_to_gps(local_x, local_y, gps_ref):
    """
    Convert local coordinates (meters) to GPS coordinates (lat/lon)
    
    Args:
        local_x, local_y: Position in meters from origin
        gps_ref: {'lat': 13.0827, 'lon': 80.2707, 'local_x': 0, 'local_y': 0}
    
    Returns:
        (latitude, longitude)
    """
    # Earth's radius in meters
    R = 6378137.0  
    
    # Offset from reference point
    dx = local_x - gps_ref['local_x']  # meters east
    dy = local_y - gps_ref['local_y']  # meters north
    
    # Convert meters to degrees
    # 1 degree latitude ≈ 111,320 meters (constant)
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters (varies with latitude)
    
    ref_lat = gps_ref['lat']
    ref_lon = gps_ref['lon']
    
    # Calculate new latitude (north/south)
    new_lat = ref_lat + (dy / 111320.0)
    
    # Calculate new longitude (east/west)
    new_lon = ref_lon + (dx / (111320.0 * np.cos(np.radians(ref_lat))))
    
    return new_lat, new_lon


# In analyze_multi_camera(), add GPS conversion:
if 'gps_reference' in camera and camera['gps_reference']:
    gps_ref = camera['gps_reference']
    start_lat, start_lon = local_to_gps(
        detection['world_x_start'], 
        detection['world_y_start'],
        gps_ref
    )
    end_lat, end_lon = local_to_gps(
        detection['world_x_end'],
        detection['world_y_end'],
        gps_ref
    )
    
    all_detections.append({
        # ... existing fields ...
        'gps_lat_start': float(start_lat),
        'gps_lon_start': float(start_lon),
        'gps_lat_end': float(end_lat),
        'gps_lon_end': float(end_lon)
    })
```

### Example Workflow:

```
1. User opens Google Maps satellite view of the road
2. Right-click on a road intersection → Copy coordinates
   → Gets: 13.0827° N, 80.2707° E
3. In calibration modal, user enters GPS reference:
   - Latitude: 13.0827
   - Longitude: 80.2707
   - Click intersection in video frame (marks local origin)
4. System calibrates using this reference
5. All detections now have GPS coordinates!
```

### CSV Output with GPS:
```csv
global_id,cameras_seen,gps_lat_start,gps_lon_start,gps_lat_end,gps_lon_end,world_x_start,world_y_start,velocity_mps
V0001,"['CAM1','CAM2']",13.0828,80.2708,13.0830,80.2712,25.4,10.85,12.5
V0002,"['CAM2','CAM3']",13.0826,80.2709,13.0829,80.2715,15.15,8.55,8.3
```

---

## 🌍 Method 2: Multiple GPS Points (More Accurate)

For better accuracy across large areas, use **4 GPS points** instead of 4 local meter points:

### Modified Calibration:
```javascript
// User clicks 4 points and enters GPS coordinates directly
const worldPoints = [
  [13.0827, 80.2707],  // GPS coords of point 1
  [13.0830, 80.2710],  // GPS coords of point 2
  [13.0832, 80.2708],  // GPS coords of point 3
  [13.0829, 80.2705]   // GPS coords of point 4
];

// Backend converts GPS → local meters for homography calculation
// Then stores GPS reference for reverse conversion
```

### Backend Conversion:
```python
def gps_to_local_meters(gps_points):
    """
    Convert GPS coordinates to local meters
    Uses first point as origin (0, 0)
    """
    origin = gps_points[0]
    local_points = []
    
    for point in gps_points:
        # Calculate distance from origin in meters
        dy = (point[0] - origin[0]) * 111320.0  # latitude difference
        dx = (point[1] - origin[1]) * 111320.0 * np.cos(np.radians(origin[0]))  # longitude difference
        local_points.append([dx, dy])
    
    return np.array(local_points, dtype=np.float32), origin
```

---

## 📹 Video Sequencing: Does Order Matter?

### Short Answer: **NO, order doesn't matter!**

The system uses **spatial fusion**, not sequential processing. Here's why:

### How Fusion Works:
```python
# System doesn't care about order
cameras = [
    {'video': 'D1F1.mp4', 'H': H1},  # Front camera
    {'video': 'D2F1.mp4', 'H': H2},  # Side camera  
    {'video': 'D3F1.mp4', 'H': H3}   # Rear camera
]

# Could be in ANY order:
cameras = [
    {'video': 'D3F1.mp4', 'H': H3},  # Rear first
    {'video': 'D1F1.mp4', 'H': H1},  # Front last
    {'video': 'D2F1.mp4', 'H': H2}   # Side middle
]

# Result is IDENTICAL!
```

### Why Order Doesn't Matter:

1. **Each camera processed independently**
   - Videos are analyzed separately (parallel processing possible)
   - Detection lists generated independently

2. **Fusion is spatial, not temporal**
   - System looks at **positions**, not time sequence
   - Groups detections within 2m radius (spatial clustering)
   - No "previous camera" or "next camera" logic

3. **Homography transforms to common space**
   - All coordinates transformed to same reference frame
   - Doesn't matter which camera processed first

### Example:
```
Scenario: 3 cameras recording the same road section

Process Order A:        Process Order B:        Process Order C:
1. Front camera         1. Rear camera          1. Side camera
2. Side camera          2. Side camera          2. Front camera
3. Rear camera          3. Front camera         3. Rear camera

Result in ALL cases:
- Vehicle V0001 detected at (25.4m, 10.8m)
- Seen by ['CAM1', 'CAM2', 'CAM3']
- Total distance: 45.3m
```

---

## ⏰ When Video Order/Timing WOULD Matter

### Current System: Ignores Time
The fusion algorithm currently uses **only spatial proximity**:
```python
# Current logic:
if distance_between_detections < 2.0:  # meters
    merge_into_same_vehicle()
# Time is ignored!
```

### Problem with This Approach:
```
Time: 10:00 AM - Camera 1 sees blue car at (25m, 10m)
Time: 10:05 AM - Camera 2 sees red car at (25m, 10m)
                  ↓
          Current system: MERGES THEM (wrong!)
          Reason: Same location, but 5 minutes apart
```

### Future Enhancement: Temporal Filtering
```python
def fuse_detections_with_time(detections, spatial_threshold=2.0, time_threshold=5.0):
    """
    Fuse detections using BOTH space and time
    
    Args:
        spatial_threshold: Distance in meters (default 2m)
        time_threshold: Time difference in seconds (default 5s)
    """
    for det1 in detections:
        for det2 in detections:
            # Spatial check
            distance = calculate_distance(det1, det2)
            
            # Temporal check (NEW)
            time_diff = abs(det1['timestamp'] - det2['timestamp'])
            
            # Merge only if BOTH conditions met
            if distance < spatial_threshold and time_diff < time_threshold:
                merge_detections(det1, det2)
```

### When You NEED Time Synchronization:
- ✅ **Overlapping camera views** (vehicles visible in multiple cameras simultaneously)
- ✅ **Real-time tracking** (live feed from multiple cameras)
- ✅ **High traffic density** (many vehicles at same location at different times)

### When You DON'T NEED Time Synchronization:
- ✅ **Non-overlapping cameras** (each camera sees different road section)
- ✅ **Low traffic** (unlikely two vehicles at exact same spot)
- ✅ **Offline analysis** (post-processing recorded videos)

---

## 🎯 Practical Recommendations

### For GPS Coordinates:

**Option 1: Quick & Simple**
- Use **Method 1** (single GPS reference point)
- Pick an easily identifiable landmark (intersection, building)
- Get GPS from Google Maps → Right-click → Copy coordinates
- ±5-10m accuracy is typical (good enough for most use cases)

**Option 2: High Accuracy**
- Use **Method 2** (4 GPS reference points)
- Use RTK GPS device for centimeter-level accuracy
- Measure 4 corners of your scene
- Best for large areas or precision requirements

### For Video Loading:

**You can add videos in ANY order:**
```
✅ Front → Side → Rear
✅ Rear → Front → Side  
✅ Side → Rear → Front
✅ All at once (batch upload)
```

**Just ensure:**
- ✅ Each camera is **calibrated** (homography calculated)
- ✅ Videos are **from the same time period** (if using temporal filtering)
- ✅ Overlapping cameras have **consistent coordinate systems** (use same GPS reference)

---

## 🔧 Implementation Priority

### Phase 1: Basic GPS Support (Quick Win)
```
1. Add GPS reference fields to calibration modal
2. Store GPS reference with homography matrix
3. Convert local coords → GPS in output CSV
4. Test with Google Maps coordinates
```

### Phase 2: Enhanced Fusion (If Needed)
```
1. Add timestamp extraction from videos
2. Implement temporal filtering in fuse_detections()
3. Add time synchronization warnings
4. Test with overlapping camera footage
```

### Phase 3: Advanced Features (Future)
```
1. GPS-based calibration (4 GPS points instead of manual measurement)
2. Automatic coordinate system alignment
3. Real-time GPS streaming integration
4. Map overlay visualization (trajectories on Google Maps)
```

---

## 📊 Example: Complete Workflow with GPS

### Setup:
```
Location: IIT Madras Main Gate
GPS Reference: 12.9915° N, 80.2336° E (gate entrance)
Videos: 3 drone cameras with overlapping views
```

### Calibration:
```javascript
// Camera 1 (Front view)
imagePoints: [[450, 320], [800, 320], [800, 580], [450, 580]]
gpsReference: {
  lat: 12.9915,
  lon: 80.2336,
  localX: 0,   // Origin at gate entrance
  localY: 0
}

// Camera 2 (Side view) - SAME GPS reference!
imagePoints: [[200, 400], [650, 400], [650, 700], [200, 700]]
gpsReference: {
  lat: 12.9915,  // Same reference point
  lon: 80.2336,
  localX: 0,
  localY: 0
}

// Camera 3 (Rear view) - SAME GPS reference!
imagePoints: [[350, 280], [920, 280], [920, 620], [350, 620]]
gpsReference: {
  lat: 12.9915,  // Same reference point
  lon: 80.2336,
  localX: 0,
  localY: 0
}
```

### Output CSV with GPS:
```csv
global_id,cameras_seen,gps_lat_start,gps_lon_start,gps_lat_end,gps_lon_end,velocity_mps,distance_m
V0001,"['CAM1','CAM2']",12.9915,80.2336,12.9917,80.2340,12.5,48.3
V0002,"['CAM2','CAM3']",12.9914,80.2337,12.9916,80.2342,8.3,52.1
V0003,"['CAM1']",12.9916,80.2335,12.9918,80.2338,15.2,35.7
```

### Visualization:
Import CSV into Google My Maps:
1. File → Import → Select CSV
2. Choose `gps_lat_start`, `gps_lon_start` as start marker
3. Choose `gps_lat_end`, `gps_lon_end` as end marker
4. See vehicle trajectories overlaid on satellite imagery! 🗺️

---

## ❓ FAQ

**Q: Do all cameras need the same GPS reference point?**
A: **YES!** All cameras must use the **same reference** to align coordinate systems.

**Q: Can I use my phone's GPS for reference coordinates?**
A: Yes! Install "GPS Status" app, stand at reference point, copy coordinates. Accuracy: ±3-5m.

**Q: What if I don't have GPS coordinates?**
A: Use local coordinates (current system). Works fine for spatial analysis, just no map overlay.

**Q: Does video order affect GPS coordinates?**
A: **NO!** GPS conversion is per-detection, independent of processing order.

**Q: Can I add GPS reference after calibration?**
A: Not currently. You'd need to recalibrate. Future feature: post-calibration GPS alignment.

**Q: How accurate is the GPS conversion?**
A: Depends on reference point accuracy:
- Google Maps: ±5-10m
- Consumer GPS: ±3-5m  
- RTK GPS: ±0.02m (2cm!)

---

## 🎓 Summary

### GPS Coordinates:
- ✅ **Need GPS reference point** during calibration
- ✅ **Use same reference** for all cameras
- ✅ **Get from Google Maps** (right-click → copy coordinates)
- ✅ **Converts local meters → lat/lon** automatically

### Video Sequencing:
- ✅ **Order doesn't matter** (spatial fusion, not sequential)
- ✅ **Add videos in any order** (front-side-rear or rear-front-side)
- ✅ **Process is parallel** (each camera independent)
- ✅ **Time sync optional** (only needed for high-density overlapping areas)

### Implementation Steps:
1. Add GPS reference fields to UI ✍️
2. Update backend to accept GPS reference 🔧
3. Implement `local_to_gps()` conversion 📐
4. Update CSV output with GPS columns 📊
5. Test with Google Maps overlay 🗺️
