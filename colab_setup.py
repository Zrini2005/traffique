#!/usr/bin/env python3
"""
Setup script for running traffic analysis in Google Colab
"""

# Install required packages
print("Installing required packages...")
print("Run these commands in your Colab notebook:")
print("""
# Cell 1: Install dependencies
!pip install ultralytics opencv-python-headless numpy torch sahi huggingface_hub flask flask-cors

# Cell 2: Clone repository (if needed)
!git clone https://github.com/SJK2150/traffique.git
%cd traffique

# Cell 3: Upload your video to Colab
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print(f"Video uploaded: {video_path}")

# Cell 4: Run analysis
from interactive_analytics import InteractiveVehicleAnalytics
import json

# Initialize analyzer
analyzer = InteractiveVehicleAnalytics(
    video_path=video_path,
    use_sahi=True,  # Enable SAHI for better accuracy
    confidence_threshold=0.25
)

# Analyze video (full tracking mode)
results = analyzer.analyze_full_video(
    roi_points=None,  # No ROI - analyze entire frame
    start_frame=0,
    end_frame=None,  # Analyze all frames
    save_every=25  # Save every 25th frame for speed
)

# Display results
print(f"\\nVehicles detected: {len(results['vehicles'])}")
print(f"\\nSample vehicle data:")
for vid, vdata in list(results['vehicles'].items())[:3]:
    print(f"  {vid}: {vdata['class']} - {vdata['frames_tracked']} frames")

# Save results
with open('vehicle_analytics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\\nResults saved to vehicle_analytics.json")

# Cell 5: Download results
from google.colab import files
files.download('vehicle_analytics.json')
if 'output/vehicle_analytics.csv' in !ls output/:
    files.download('output/vehicle_analytics.csv')
""")
