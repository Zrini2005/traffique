#!/usr/bin/env python3
"""
Method 1: Stabilo-based Video Stabilization
Uses the Stabilo library designed for traffic monitoring scenarios.
Handles drift over long durations by locking to static background.
"""

import cv2
import numpy as np
from pathlib import Path
import time
import sys

def install_stabilo():
    """Install Stabilo if not available"""
    try:
        import vidstab
        print("✓ Stabilo (vidstab) already installed")
        return True
    except ImportError:
        print("Installing Stabilo (vidstab)...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "vidstab"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Stabilo installed successfully")
            return True
        else:
            print(f"✗ Failed to install Stabilo: {result.stderr}")
            return False

def stabilize_with_stabilo(input_video_path, output_video_path, mask_path=None):
    """
    Stabilize video using Stabilo library
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output stabilized video
        mask_path: Optional path to mask image (white=road, black=ignore cars)
    """
    from vidstab import VidStab
    
    print("\n" + "="*70)
    print("METHOD 1: STABILO-BASED STABILIZATION")
    print("="*70)
    print(f"Input:  {input_video_path}")
    print(f"Output: {output_video_path}")
    if mask_path:
        print(f"Mask:   {mask_path}")
    print()
    
    # Get video info
    cap = cv2.VideoCapture(str(input_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    cap.release()
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration_sec/60:.2f} minutes")
    print()
    
    # Load mask if provided
    layer_func = None
    if mask_path and Path(mask_path).exists():
        print(f"Loading mask from {mask_path}...")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Resize mask to match video
            mask = cv2.resize(mask, (width, height))
            # Create layer function that applies mask
            def layer_func(frame):
                return cv2.bitwise_and(frame, frame, mask=mask)
            print("✓ Mask loaded and will be applied")
        else:
            print("⚠ Failed to load mask, proceeding without it")
    
    # Initialize Stabilo
    print("\nInitializing Stabilo stabilizer...")
    stabilizer = VidStab()
    
    # Configure stabilizer for traffic monitoring
    # Key parameters:
    # - smoothing_window: Larger = smoother but more lag (default: 30)
    # - max_frames: Process all frames for long videos
    # - border_type: How to handle black borders
    # - border_size: Auto-crop to remove borders
    
    start_time = time.time()
    
    print("\nStarting stabilization...")
    print("This will take a while for long videos...")
    print("Progress will be shown below:")
    print("-" * 70)
    
    try:
        # Stabilize the video
        stabilizer.stabilize(
            input_path=str(input_video_path),
            output_path=str(output_video_path),
            smoothing_window=50,  # Increased for traffic (longer smoothing)
            max_frames=total_frames,  # Set to total frames (workaround for vidstab bug)
            border_type='black',  # Keep black borders visible
            border_size='auto',  # Auto-crop borders
            layer_func=layer_func,  # Apply mask if provided
            playback=False  # Don't show preview (faster)
        )
        
        elapsed_time = time.time() - start_time
        
        print("-" * 70)
        print("\n✓ STABILIZATION COMPLETE!")
        print(f"Time taken: {elapsed_time/60:.2f} minutes")
        print(f"Output saved to: {output_video_path}")
        
        # Calculate processing speed
        fps_processed = total_frames / elapsed_time
        print(f"Processing speed: {fps_processed:.2f} fps")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR during stabilization: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_road_mask_interactive(video_path, output_mask_path):
    """
    Interactive tool to create a mask for the road region.
    User draws a polygon around the road to focus stabilization on it.
    """
    print("\n" + "="*70)
    print("INTERACTIVE MASK CREATOR")
    print("="*70)
    print("Instructions:")
    print("  1. Click points to draw a polygon around the ROAD area")
    print("  2. Press SPACE to finish the polygon")
    print("  3. Press 'r' to reset and start over")
    print("  4. Press 'q' to quit without saving")
    print("="*70)
    print()
    
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Failed to read first frame")
        return None
    
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, mask, frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw point
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # Draw lines between points
            if len(points) > 1:
                cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
            cv2.imshow('Create Road Mask', frame)
    
    cv2.namedWindow('Create Road Mask')
    cv2.setMouseCallback('Create Road Mask', mouse_callback)
    cv2.imshow('Create Road Mask', frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - finish polygon
            if len(points) >= 3:
                # Close polygon
                cv2.line(frame, points[-1], points[0], (0, 255, 0), 2)
                # Fill mask
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                cv2.imshow('Create Road Mask', frame)
                cv2.waitKey(500)
                break
            else:
                print("Need at least 3 points to create a polygon")
        
        elif key == ord('r'):  # Reset
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            mask = np.zeros((height, width), dtype=np.uint8)
            points = []
            cv2.imshow('Create Road Mask', frame)
        
        elif key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    # Save mask
    cv2.imwrite(str(output_mask_path), mask)
    print(f"\n✓ Mask saved to: {output_mask_path}")
    
    # Show preview
    preview = cv2.addWeighted(frame, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow('Mask Preview (Press any key)', preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return output_mask_path

def main():
    """Main function"""
    print("\n" + "="*70)
    print("VIDEO STABILIZATION - METHOD 1: STABILO")
    print("="*70)
    print()
    
    # Check/install Stabilo
    if not install_stabilo():
        print("Cannot proceed without Stabilo. Exiting.")
        return
    
    # Get input video path
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        input_video = input("Enter path to input video (or drag & drop): ").strip().strip('"\'')
    
    input_path = Path(input_video)
    if not input_path.exists():
        print(f"✗ Video file not found: {input_path}")
        return
    
    # Generate output path
    output_path = input_path.parent / f"{input_path.stem}_stabilo_stabilized.mp4"
    
    # Ask about mask
    print("\nDo you want to create a mask for the road region?")
    print("(Recommended: helps focus on road, ignores moving cars)")
    use_mask = input("Create mask? (y/n) [n]: ").strip().lower()
    
    mask_path = None
    if use_mask == 'y':
        mask_path = input_path.parent / f"{input_path.stem}_road_mask.png"
        created_mask = create_road_mask_interactive(input_path, mask_path)
        if created_mask is None:
            print("Proceeding without mask...")
            mask_path = None
    
    # Stabilize
    success = stabilize_with_stabilo(input_path, output_path, mask_path)
    
    if success:
        print("\n" + "="*70)
        print("✓ ALL DONE!")
        print("="*70)
        print(f"\nStabilized video: {output_path}")
        if mask_path and Path(mask_path).exists():
            print(f"Road mask: {mask_path}")
        print("\nYou can now use this stabilized video for traffic analysis.")
    else:
        print("\n✗ Stabilization failed. Check error messages above.")

if __name__ == "__main__":
    main()
