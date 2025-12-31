#!/usr/bin/env python3
"""
Quick test script to validate trajectory accuracy fixes
Processes a short segment and reports key metrics
"""

import subprocess
import sys
from pathlib import Path

def run_test():
    print("="*70)
    print("TRAJECTORY ACCURACY FIX - QUICK TEST")
    print("="*70)
    
    # Test parameters
    video = "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4"
    gt = "/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv"
    frames = 200
    
    print(f"\n📹 Video: D2F1_stab.mp4")
    print(f"📊 Ground Truth: D2F1_lclF (1).csv")
    print(f"🎬 Frames: {frames}")
    print(f"\n🔧 New Settings Applied:")
    print(f"   • Confidence: 0.45 (was 0.3)")
    print(f"   • min_iou: 0.50 (was 0.40)")
    print(f"   • max_age: 12 (was 20)")
    print(f"   • Spatial threshold: 50px (was 70px)")
    print(f"   • Smoothing: REDUCED (3x lighter)")
    print(f"   • SAHI overlap: 0.3 (was 0.1)")
    print(f"   • Quality filtering: ADDED")
    
    print("\n" + "="*70)
    print("Running trajectory generation and comparison...")
    print("="*70 + "\n")
    
    # Run the comparison script
    cmd = [
        sys.executable,
        "generate_and_compare_properly.py",
        "--video", video,
        "--gt", gt,
        "--frames", str(frames),
        "--roi", "road",
        "--best", "5",
        "--worst", "5"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print("\n" + "="*70)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\n📋 Check the comparison_XXX folder for:")
        print("   1. trajectories_overlay.mp4 - Visual inspection")
        print("   2. comprehensive_statistics.csv - RMSE metrics")
        print("   3. best_*.mp4 - Verify trajectory accuracy")
        print("   4. worst_*.mp4 - Check problematic cases")
        
        print("\n🎯 Expected Improvements:")
        print("   • RMSE: Should be 12-18px (was 25-35px)")
        print("   • Y-jitter ratio: Should be 1.0-1.3x (was 2.0-3.0x)")
        print("   • Match rate: Should be >75% (was ~50%)")
        print("   • No vehicle jumping in overlay videos")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except FileNotFoundError:
        print(f"\n❌ ERROR: generate_and_compare_properly.py not found")
        return 1

if __name__ == "__main__":
    sys.exit(run_test())
