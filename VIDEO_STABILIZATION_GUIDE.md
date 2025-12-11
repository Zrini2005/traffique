# Video Stabilization Guide

This directory contains two methods for stabilizing drone footage for traffic analysis.

## 📁 Files Created

1. **`stabilize_method1_stabilo.py`** - Stabilo-based stabilization
2. **`stabilize_method2_homography_ransac.py`** - Custom Homography + RANSAC

---

## 🎯 Method Comparison

### Method 1: Stabilo (VidStab Library)

**Best For:**
- Quick and easy stabilization
- General-purpose traffic video
- Users who want minimal configuration

**Pros:**
- ✅ Easy to use (one command)
- ✅ Automatic parameter tuning
- ✅ Built-in smoothing and border handling
- ✅ Auto-crop black borders
- ✅ Designed for long-duration videos

**Cons:**
- ❌ Less control over algorithm
- ❌ May not handle extreme perspective changes well
- ❌ Requires installing external library

**When to Use:**
- First-time stabilization
- Moderate drone movement
- You want quick results

---

### Method 2: Homography + RANSAC (Custom OpenCV)

**Best For:**
- Drone footage with altitude/tilt changes
- High precision requirements
- Advanced users who want control

**Pros:**
- ✅ Handles perspective transformations (8-DOF)
- ✅ RANSAC robustly ignores moving cars
- ✅ Full control over parameters
- ✅ No external dependencies (just OpenCV)
- ✅ Better for extreme camera motion

**Cons:**
- ❌ More complex to configure
- ❌ May require parameter tuning
- ❌ Slower processing

**When to Use:**
- Drone changes altitude significantly
- Method 1 results not satisfactory
- You need maximum control

---

## 🚀 Usage

### Method 1: Stabilo

```bash
# Basic usage (interactive)
python stabilize_method1_stabilo.py

# With video path
python stabilize_method1_stabilo.py /path/to/your/video.mp4

# With mask for road region
python stabilize_method1_stabilo.py /path/to/video.mp4
# Then choose 'y' when asked about creating a mask
```

**Output:** `video_stabilo_stabilized.mp4`

---

### Method 2: Homography + RANSAC

```bash
# Basic usage (interactive)
python stabilize_method2_homography_ransac.py

# With video path
python stabilize_method2_homography_ransac.py /path/to/your/video.mp4

# With custom parameters
python stabilize_method2_homography_ransac.py /path/to/video.mp4
# Then choose configuration option (1-4)
```

**Output:** `video_ransac_stabilized.mp4`

---

## 🎨 Creating Road Masks (Optional but Recommended)

Both methods support road masks to improve stabilization:

1. Run either script
2. Choose 'y' when asked about creating a mask
3. Interactive window will open:
   - **Click** points around the road area
   - **Avoid** areas with heavy car traffic
   - **Press SPACE** to finish polygon
   - **Press 'r'** to reset
   - **Press 'q'** to cancel

**Why Use Masks?**
- Focuses feature detection on static road surface
- Ignores moving vehicles
- Reduces false tracking
- Improves stabilization quality

---

## ⚙️ Configuration Guide

### Method 2 Configuration Options

#### Option 1: Default (Recommended for Most Cases)
```
smoothing_window = 30 frames
ransac_threshold = 5.0 pixels
```
- Balanced smoothing
- Good for moderate drone movement

#### Option 2: Aggressive Smoothing
```
smoothing_window = 50 frames
ransac_threshold = 10.0 pixels
```
- Very smooth output
- Best for shaky footage
- More aggressive outlier rejection

#### Option 3: Light Smoothing
```
smoothing_window = 15 frames
ransac_threshold = 3.0 pixels
```
- Preserves more original motion
- Good for already stable footage
- Faster processing

#### Option 4: Custom
- Full control over parameters
- Experiment to find optimal settings

---

## 📊 Expected Processing Times

For a **15-minute 1920x1080 video at 25fps** (~22,500 frames):

### Method 1 (Stabilo):
- **Estimate:** 30-60 minutes
- **Speed:** ~6-12 fps processing

### Method 2 (Homography + RANSAC):
- **Estimate:** 45-90 minutes
- **Speed:** ~4-8 fps processing

**Factors affecting speed:**
- Video resolution
- Processor speed
- Mask usage (slightly slower)
- Preview mode (significantly slower)

---

## 🔍 How to Compare Results

After running both methods, compare:

1. **Visual Quality:**
   - Open both videos side-by-side
   - Check for jitter/shake
   - Look at black borders (cropping)

2. **Tracking Stability:**
   - Watch a fixed point (road marking)
   - Should remain stationary

3. **Car Movement:**
   - Cars should move smoothly
   - No warping artifacts

4. **Border Handling:**
   - Method 1: Auto-crops black borders
   - Method 2: Shows black borders (can be cropped later)

---

## 🎯 Recommended Workflow

1. **Start with Method 1** (Stabilo):
   ```bash
   python stabilize_method1_stabilo.py your_video.mp4
   ```

2. **Review results:**
   - If good → Use it for traffic analysis
   - If not satisfactory → Try Method 2

3. **Try Method 2** with different configurations:
   ```bash
   python stabilize_method2_homography_ransac.py your_video.mp4
   ```
   - Start with Option 1 (Default)
   - If still shaky → Try Option 2 (Aggressive)

4. **Use mask** if needed:
   - If stabilization locks onto moving cars
   - If road has distinct features

---

## 📝 Technical Details

### Method 1: Stabilo Algorithm
- Uses optical flow between consecutive frames
- Builds trajectory of motion over time
- Smooths trajectory to remove jitter
- Applies inverse transform to stabilize

### Method 2: Homography + RANSAC Algorithm
- **Step 1:** Detect features (Shi-Tomasi corners)
- **Step 2:** Track features (Lucas-Kanade optical flow)
- **Step 3:** Estimate homography (RANSAC)
- **Step 4:** Smooth transform (temporal averaging)
- **Step 5:** Warp frame (perspective transform)

**Key Difference:** Method 2 uses 8-DOF homography (handles perspective changes) vs Method 1's 6-DOF affine (translation, rotation, scale).

---

## 🐛 Troubleshooting

### Stabilo fails to install:
```bash
pip install --upgrade vidstab
```

### OpenCV errors:
```bash
pip install --upgrade opencv-python opencv-contrib-python
```

### Out of memory:
- Process shorter clips
- Reduce video resolution first
- Close other applications

### Poor stabilization:
- Create a road mask
- Adjust RANSAC threshold (Method 2)
- Try different smoothing window (Method 2)

### Black borders too large:
- Method 1: Already auto-cropped
- Method 2: Crop manually or adjust stabilization strength

---

## 💡 Tips for Best Results

1. **Video Quality:**
   - Higher resolution = better feature tracking
   - Good lighting conditions help
   - Avoid motion blur

2. **Drone Flight:**
   - Smooth movements better than jerky
   - Consistent altitude helps
   - Avoid rapid rotation

3. **Processing:**
   - Use SSD for faster I/O
   - Process during off-hours (takes time)
   - Monitor first 30 seconds to check quality

4. **Masks:**
   - Draw masks generously (include full road)
   - Exclude intersections with many cars
   - Save mask for reuse on similar videos

---

## 📦 Output Files

After processing, you'll have:

```
your_video.mp4                          # Original
your_video_stabilo_stabilized.mp4       # Method 1 output
your_video_ransac_stabilized.mp4        # Method 2 output
your_video_road_mask.png                # Mask (if created)
```

Use the stabilized videos with your existing traffic analysis system:
```bash
# Use with your main analysis
python api_server.py
# Then upload stabilized video in the web interface
```

---

## 🎓 Summary

- **Method 1 (Stabilo):** Quick, easy, good for most cases
- **Method 2 (RANSAC):** Advanced, precise, handles perspective changes
- **Use masks:** For best quality on busy roads
- **Try both:** Compare results to find best for your footage

**Recommended:** Start with Method 1, upgrade to Method 2 if needed.
