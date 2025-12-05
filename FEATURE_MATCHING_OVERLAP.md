# 🎯 Feature Matching for Automatic Overlap Detection

## ✅ **Implemented: Option C**

The system now uses **computer vision feature matching** to automatically detect overlap between cameras!

---

## How It Works

### **Algorithm Overview:**

```
Step 1: Calibrate CAM1
  - User clicks vehicle bbox
  - System creates homography matrix H1
  - CAM1 coverage: 0-64m

Step 2: Calibrate CAM2
  - User clicks vehicle bbox
  - System calculates scale (pixels/meter)
  
Step 3: Automatic Overlap Detection 🎯
  a) Extract frame from CAM1 (frame 0)
  b) Extract frame from CAM2 (frame 0)
  c) Detect ORB features in both frames
  d) Match features using BFMatcher
  e) Keep top 30% matches (quality filter)
  
  f) For each matched feature pair:
     - Get CAM1 pixel position (x1, y1)
     - Transform to world coords using H1: (wx, wy)
     - Get CAM2 pixel position (x2, y2)
     - Calculate: cam2_origin = (wx, wy) - (x2, y2) / scale
  
  g) Take median of all calculated origins (robust to outliers)
  
  h) Calculate overlap:
     overlap_x = cam1_end_x - cam2_origin_x
     overlap_y = cam1_end_y - cam2_origin_y

Step 4: Apply detected overlap
  - If overlap > 0.5m and confidence > 0.3: Use detected value
  - Otherwise: Fall back to manual input (default 0)
```

---

## Technical Details

### **Backend: `/api/detect-overlap` Endpoint**

**Location:** `api_server.py` (lines 733-900)

**Input:**
```json
{
  "cam1_video": "D1F1_stab.mp4",
  "cam2_video": "D2F1_stab.mp4",
  "cam1_frame_idx": 0,
  "cam2_frame_idx": 0,
  "cam1_homography": [[...], [...], [...]],
  "cam2_scale": 25.5,
  "cam1_coverage": {
    "end_x": 64.0,
    "end_y": 36.0
  }
}
```

**Output:**
```json
{
  "overlap_detected": true,
  "overlap_x": 18.5,
  "overlap_y": 2.3,
  "confidence": 0.75,
  "num_matches": 42,
  "cam2_origin_x": 45.5,
  "cam2_origin_y": -2.3,
  "message": "Found 42 feature matches (confidence: 0.75)"
}
```

**Feature Detection:**
- **Algorithm:** ORB (Oriented FAST and Rotated BRIEF)
- **Why ORB?** Fast, patent-free, works well for real-time apps
- **Parameters:** 2000 max features
- **Matching:** BFMatcher with Hamming distance, cross-check enabled

**Quality Filters:**
1. Minimum 10 features per image
2. Keep top 30% of matches (sorted by distance)
3. Minimum 10 good matches required
4. Confidence threshold: 0.3

**Confidence Calculation:**
```python
match_confidence = min(1.0, num_matches / 50.0)
consistency_confidence = 1.0 / (1.0 + std_deviation_of_origins / 10.0)
overall_confidence = (match_confidence + consistency_confidence) / 2
```

---

### **Frontend Integration**

**Location:** `MultiCameraPage.jsx` (lines 335-393)

**Workflow:**
```javascript
// After calibration homography is calculated:
if (cameraIndex > 0 && calibrationMode === 'auto') {
  // Try feature matching
  const overlapResponse = await fetch('/api/detect-overlap', {...});
  
  if (overlapResponse.ok) {
    const data = await overlapResponse.json();
    
    if (data.overlap_detected && data.confidence > 0.3) {
      // Use auto-detected overlap ✅
      detectedOverlap = {
        overlapX: data.overlap_x,
        overlapY: data.overlap_y
      };
    } else {
      // Fall back to manual input
      detectedOverlap = {
        overlapX: calibratingCamera.overlapX || 0,
        overlapY: calibratingCamera.overlapY || 0
      };
    }
  }
}
```

**UI Indicators:**
- **Green badge (🎯 auto)**: Overlap detected via feature matching
- **Purple badge (🔗 manual)**: Manual input or no overlap
- **Display format**: "Overlap: 18.5m × 2.3m (auto)"

---

## Advantages

### **Why Feature Matching Works:**

✅ **No GPS needed** - Pure computer vision  
✅ **No manual measurement** - Fully automatic  
✅ **Scene-independent** - Works regardless of vehicle positions  
✅ **Robust** - Uses median to filter outliers  
✅ **Fast** - ORB is optimized for speed  
✅ **Confidence score** - Know when to trust the result

### **When It Works Best:**

- Overlapping cameras with 10-50% overlap
- Scenes with good texture (roads, buildings, trees)
- Consistent lighting between cameras
- Stable camera positions (not extreme angles)

### **When It Struggles:**

- No overlap (returns 0, which is correct)
- Very low texture scenes (empty sky, uniform surfaces)
- Extreme perspective differences
- Motion blur in frames

---

## Usage

### **For Users:**

```
1. Add CAM1, calibrate normally
   → System stores homography

2. Add CAM2
   → Leave overlap fields at 0
   → Calibrate normally
   
3. System automatically:
   ✨ Detects features in both videos
   ✨ Matches features
   ✨ Calculates overlap
   ✨ Shows result: "🎯 Overlap: 18.5m × 2.3m (auto)"

4. If auto-detection fails:
   → Manually enter overlap estimate
   → System uses manual value instead
```

### **Manual Override:**

If auto-detection gives wrong results, you can:
1. Recalibrate CAM2
2. Enter manual overlap values **before** calibrating
3. System will use your manual values instead of auto-detection

---

## Example Scenarios

### **Scenario 1: Good Overlap Detection**

```
CAM1: Highway section with trees, road markings
CAM2: Overlapping highway section with same trees

Feature matching finds:
- 45 matched features (trees, road edges, lane markers)
- Consistent origin calculation (low std deviation)
- Confidence: 0.82

Result: ✅ Overlap detected: 22.3m × 1.8m
```

### **Scenario 2: No Overlap**

```
CAM1: Highway 0-100m
CAM2: Highway 100-200m (edge-to-edge, no overlap)

Feature matching finds:
- 3 matched features (very different scenes)
- Below 10 matches threshold

Result: ℹ️ No overlap detected (0m × 0m)
```

### **Scenario 3: Failed Detection**

```
CAM1: Highway at noon
CAM2: Highway at dusk (different lighting)

Feature matching finds:
- 8 matched features (below 10 threshold)
- OR high standard deviation in origin calculation

Result: ⚠️ Low confidence, falling back to manual input
```

---

## Technical Trade-offs

### **Why ORB over SIFT/SURF?**

| Feature | ORB | SIFT | SURF |
|---------|-----|------|------|
| Speed | ⚡⚡⚡ Very Fast | 🐌 Slow | 🚀 Fast |
| Patent | ✅ Free | ❌ Patented | ❌ Patented |
| Accuracy | ✅ Good | ⭐ Excellent | ⭐ Excellent |
| Rotation | ✅ Yes | ✅ Yes | ✅ Yes |
| Scale | ✅ Yes | ✅ Yes | ✅ Yes |

**Decision:** ORB is fast enough, accurate enough, and patent-free!

### **Why Median over Mean?**

```python
# Outlier example:
origins = [45, 46, 44, 47, 45, 120]  # One bad match!

mean = 57.8  ❌ Skewed by outlier
median = 45.5  ✅ Robust to outlier
```

---

## Future Improvements

### **Potential Enhancements:**

1. **Use current calibration frame** instead of frame 0
   - More likely to show overlap zone
   - Better matches

2. **RANSAC for robustness**
   - Filter outlier matches before origin calculation
   - Higher accuracy

3. **Multi-frame matching**
   - Match features across multiple frames
   - Average results for stability

4. **Visual feedback**
   - Show matched features on frames
   - Let user verify matches

5. **Deep learning features**
   - Use SuperPoint or LoFTR
   - Better accuracy in difficult scenes

---

## Testing Checklist

- [ ] CAM1 + CAM2 with 20m overlap → Detects correctly
- [ ] CAM1 + CAM2 edge-to-edge → Returns 0
- [ ] CAM1 + CAM2 with gap → Returns 0
- [ ] Poor lighting conditions → Falls back to manual
- [ ] Low texture scene → Falls back to manual
- [ ] Manual override works
- [ ] Confidence score correlates with accuracy
- [ ] UI shows correct badge (🎯 vs 🔗)

---

## Summary

| Feature | Status | Notes |
|---------|--------|-------|
| **Feature Detection** | ✅ Implemented | ORB with 2000 features |
| **Feature Matching** | ✅ Implemented | BFMatcher with cross-check |
| **Overlap Calculation** | ✅ Implemented | Median-based robust estimation |
| **Confidence Score** | ✅ Implemented | Match count + consistency |
| **API Endpoint** | ✅ Implemented | `/api/detect-overlap` |
| **Frontend Integration** | ✅ Implemented | Auto-call after calibration |
| **Manual Fallback** | ✅ Implemented | If detection fails |
| **Visual Feedback** | ✅ Implemented | Green/purple badges |
| **Testing** | ⏸️ Pending | Need real drone videos |

**Bottom Line:** Fully automatic overlap detection using computer vision! No GPS, no manual measurement, just pure feature matching. 🎯
