#!/usr/bin/env python3
"""
PRECISION TEST - Maximum accuracy tracking for long-duration vehicles
"""

import subprocess
import sys

print("="*70)
print("🎯 PRECISION TRACKING TEST - LAST CHANCE")
print("="*70)

print("\n⚙️  AGGRESSIVE PRECISION SETTINGS:")
print("   • Confidence: 0.5 (HIGHEST quality detections only)")
print("   • Min trajectory: 50 frames (only long-duration vehicles)")
print("   • Smoothing: MINIMAL (preserve exact positions)")
print("   • Tracking: STRICT (min_iou=0.45, max_age=40)")
print("   • Spatial threshold: 40px (prevent jumping)")
print("   • SAHI overlap: 0.4 (maximum consistency)")
print("   • Detection filter: 600px min area (larger vehicles only)")

print("\n🎬 Processing 400 frames starting at 4500...")
print("   Target: Vehicles appearing throughout entire duration")
print("   Expected: <10-15 high-quality trajectories")
print("   Goal: <5px RMSE on best vehicles")

cmd = [
    sys.executable,
    "generate_and_compare_properly.py",
    "--video", "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4",
    "--gt", "/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv",
    "--frames", "400",
    "--roi", "road",
    "--start-frame", "4500",
    "--best", "10",
    "--worst", "3"
]

print("\n" + "="*70)
print("STARTING PRECISION TRACKING...")
print("="*70 + "\n")

try:
    subprocess.run(cmd, check=True)
    print("\n" + "="*70)
    print("✅ PRECISION TEST COMPLETE")
    print("="*70)
    print("\n📋 CHECK RESULTS:")
    print("   1. Vehicle count should be LOW (10-20 high-quality tracks)")
    print("   2. RMSE should be <10px (ideally <5px on best)")
    print("   3. Trajectory overlay should show PERFECT alignment")
    print("   4. No vehicle jumping - check single vehicle videos")
except subprocess.CalledProcessError:
    print("\n❌ TEST FAILED")
    sys.exit(1)
