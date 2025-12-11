#!/usr/bin/env python3
"""
Method 2: Custom Homography + RANSAC Video Stabilization
Uses OpenCV's feature detection and RANSAC to robustly stabilize video
by focusing on static background (road) and ignoring moving objects (cars).
"""

import cv2
import numpy as np
from pathlib import Path
import time
import sys
from collections import deque

class HomographyStabilizer:
    """
    Custom video stabilizer using Homography + RANSAC.
    Designed to handle drone footage with perspective changes.
    """
    
    def __init__(self, smoothing_window=30, ransac_threshold=5.0):
        """
        Initialize stabilizer
        
        Args:
            smoothing_window: Number of frames to average transforms over
            ransac_threshold: RANSAC reprojection threshold (higher = more aggressive outlier removal)
        """
        self.smoothing_window = smoothing_window
        self.ransac_threshold = ransac_threshold
        self.transforms = deque(maxlen=smoothing_window)
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=500,  # Maximum number of features to detect
            qualityLevel=0.01,  # Quality threshold for features
            minDistance=30,  # Minimum distance between features
            blockSize=7  # Size of neighborhood for feature detection
        )
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),  # Window size for optical flow
            maxLevel=3,  # Pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        print("Initialized Homography + RANSAC Stabilizer")
        print(f"  Smoothing window: {smoothing_window} frames")
        print(f"  RANSAC threshold: {ransac_threshold}")
    
    def detect_features(self, frame, mask=None):
        """
        Detect good features to track in the frame.
        
        Args:
            frame: Input frame (grayscale)
            mask: Optional mask (255=detect, 0=ignore)
        
        Returns:
            Array of feature points
        """
        features = cv2.goodFeaturesToTrack(
            frame,
            mask=mask,
            **self.feature_params
        )
        return features
    
    def track_features(self, prev_frame, curr_frame, prev_points):
        """
        Track features from previous frame to current frame using optical flow.
        
        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            prev_points: Features from previous frame
        
        Returns:
            Tuple of (good_prev_points, good_curr_points, status)
        """
        if prev_points is None or len(prev_points) == 0:
            return None, None, None
        
        # Calculate optical flow
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            curr_frame,
            prev_points,
            None,
            **self.lk_params
        )
        
        if curr_points is None:
            return None, None, None
        
        # Keep only good points (status=1)
        status = status.reshape(-1)
        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]
        
        return good_prev, good_curr, status
    
    def estimate_homography(self, prev_points, curr_points):
        """
        Estimate homography transformation using RANSAC.
        
        Args:
            prev_points: Feature points from previous frame
            curr_points: Corresponding points in current frame
        
        Returns:
            Homography matrix (3x3) or None if failed
        """
        if prev_points is None or curr_points is None:
            return None
        
        if len(prev_points) < 4:
            return None
        
        # Estimate homography with RANSAC
        # RANSAC will ignore outliers (moving cars)
        H, mask = cv2.findHomography(
            curr_points,
            prev_points,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold
        )
        
        # Count inliers
        if mask is not None:
            inliers = np.sum(mask)
            total = len(mask)
            inlier_ratio = inliers / total if total > 0 else 0
            
            # Require at least 30% inliers for valid homography
            if inlier_ratio < 0.3:
                return None
        
        return H
    
    def smooth_transform(self, H):
        """
        Smooth the transformation over multiple frames to reduce jitter.
        
        Args:
            H: Current homography matrix
        
        Returns:
            Smoothed homography matrix
        """
        if H is None:
            return None
        
        self.transforms.append(H)
        
        if len(self.transforms) == 0:
            return H
        
        # Average all transforms in the window
        # This creates smooth motion by averaging out rapid changes
        avg_H = np.mean(self.transforms, axis=0)
        
        return avg_H
    
    def stabilize_video(self, input_path, output_path, mask_path=None, preview=False):
        """
        Stabilize entire video using homography + RANSAC.
        
        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video
            mask_path: Optional path to mask image (white=road, black=ignore)
            preview: Show preview window during processing
        """
        print("\n" + "="*70)
        print("METHOD 2: HOMOGRAPHY + RANSAC STABILIZATION")
        print("="*70)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        if mask_path:
            print(f"Mask:   {mask_path}")
        print()
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        print(f"Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps/60:.2f} minutes")
        print()
        
        # Load mask if provided
        mask = None
        if mask_path and Path(mask_path).exists():
            print(f"Loading mask from {mask_path}...")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (width, height))
                print("✓ Mask loaded and will be applied")
            else:
                print("⚠ Failed to load mask, proceeding without it")
        
        # Create video writer
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            print("✗ Failed to read first frame")
            cap.release()
            return False
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_points = self.detect_features(prev_gray, mask)
        
        # Write first frame as-is (reference frame)
        out.write(prev_frame)
        
        # Cumulative transform (to stabilize relative to first frame)
        cumulative_H = np.eye(3, dtype=np.float32)
        
        # Statistics
        start_time = time.time()
        frames_processed = 1
        frames_with_good_tracking = 0
        
        print("Starting stabilization...")
        print("This will take a while for long videos...")
        print("-" * 70)
        
        try:
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Track features
                good_prev, good_curr, status = self.track_features(
                    prev_gray, curr_gray, prev_points
                )
                
                # Estimate homography with RANSAC
                H = self.estimate_homography(good_prev, good_curr)
                
                if H is not None:
                    # Update cumulative transform
                    cumulative_H = cumulative_H @ H
                    
                    # Smooth the transform
                    smoothed_H = self.smooth_transform(cumulative_H)
                    
                    # Warp current frame to stabilize it
                    stabilized = cv2.warpPerspective(
                        curr_frame,
                        smoothed_H,
                        (width, height),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0)
                    )
                    
                    out.write(stabilized)
                    frames_with_good_tracking += 1
                    
                    if preview:
                        # Show side-by-side comparison
                        comparison = np.hstack([curr_frame, stabilized])
                        comparison = cv2.resize(comparison, (0, 0), fx=0.5, fy=0.5)
                        cv2.imshow('Original | Stabilized', comparison)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    # No good homography found, write original frame
                    out.write(curr_frame)
                
                # Detect new features for next iteration
                prev_points = self.detect_features(curr_gray, mask)
                prev_gray = curr_gray
                
                frames_processed += 1
                
                # Progress update every 30 frames
                if frames_processed % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frames_processed / elapsed
                    eta_sec = (total_frames - frames_processed) / fps_processed
                    tracking_rate = (frames_with_good_tracking / frames_processed) * 100
                    
                    print(f"  Frame {frames_processed}/{total_frames} "
                          f"({frames_processed/total_frames*100:.1f}%) | "
                          f"Speed: {fps_processed:.1f} fps | "
                          f"ETA: {eta_sec/60:.1f} min | "
                          f"Tracking: {tracking_rate:.1f}%")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
        
        finally:
            cap.release()
            out.release()
            if preview:
                cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        
        print("-" * 70)
        print("\n✓ STABILIZATION COMPLETE!")
        print(f"Frames processed: {frames_processed}/{total_frames}")
        print(f"Successful tracking: {frames_with_good_tracking} frames ({frames_with_good_tracking/frames_processed*100:.1f}%)")
        print(f"Time taken: {elapsed_time/60:.2f} minutes")
        print(f"Processing speed: {frames_processed/elapsed_time:.2f} fps")
        print(f"Output saved to: {output_path}")
        
        return True

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
    print("  2. Avoid areas with moving cars if possible")
    print("  3. Press SPACE to finish the polygon")
    print("  4. Press 'r' to reset and start over")
    print("  5. Press 'q' to quit without saving")
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
    display_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, mask, display_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Redraw
            display_frame = frame.copy()
            # Draw all points and lines
            for i, pt in enumerate(points):
                cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display_frame, points[i-1], points[i], (0, 255, 0), 2)
            cv2.imshow('Create Road Mask', display_frame)
    
    cv2.namedWindow('Create Road Mask')
    cv2.setMouseCallback('Create Road Mask', mouse_callback)
    cv2.imshow('Create Road Mask', display_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - finish polygon
            if len(points) >= 3:
                # Close polygon
                display_frame = frame.copy()
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                cv2.fillPoly(mask, [pts], 255)
                
                # Show preview with mask overlay
                mask_overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                preview = cv2.addWeighted(display_frame, 0.7, mask_overlay, 0.3, 0)
                cv2.imshow('Create Road Mask', preview)
                cv2.waitKey(500)
                break
            else:
                print("Need at least 3 points to create a polygon")
        
        elif key == ord('r'):  # Reset
            points = []
            mask = np.zeros((height, width), dtype=np.uint8)
            display_frame = frame.copy()
            cv2.imshow('Create Road Mask', display_frame)
        
        elif key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    # Save mask
    cv2.imwrite(str(output_mask_path), mask)
    print(f"\n✓ Mask saved to: {output_mask_path}")
    
    return output_mask_path

def main():
    """Main function"""
    print("\n" + "="*70)
    print("VIDEO STABILIZATION - METHOD 2: HOMOGRAPHY + RANSAC")
    print("="*70)
    print()
    print("This method uses:")
    print("  • Good Features to Track (Shi-Tomasi corners)")
    print("  • Lucas-Kanade Optical Flow")
    print("  • Homography estimation with RANSAC")
    print("  • Temporal smoothing to reduce jitter")
    print()
    
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
    output_path = input_path.parent / f"{input_path.stem}_ransac_stabilized.mp4"
    
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
    
    # Configuration
    print("\nStabilizer Configuration:")
    print("  1. Default (smoothing_window=30, ransac_threshold=5.0)")
    print("  2. Aggressive smoothing (smoothing_window=50, ransac_threshold=10.0)")
    print("  3. Light smoothing (smoothing_window=15, ransac_threshold=3.0)")
    print("  4. Custom")
    
    config_choice = input("\nChoice [1]: ").strip() or '1'
    
    if config_choice == '2':
        smoothing_window = 50
        ransac_threshold = 10.0
    elif config_choice == '3':
        smoothing_window = 15
        ransac_threshold = 3.0
    elif config_choice == '4':
        smoothing_window = int(input("Smoothing window (frames) [30]: ").strip() or '30')
        ransac_threshold = float(input("RANSAC threshold [5.0]: ").strip() or '5.0')
    else:
        smoothing_window = 30
        ransac_threshold = 5.0
    
    # Preview option
    show_preview = input("\nShow preview during processing? (slower) (y/n) [n]: ").strip().lower()
    preview = show_preview == 'y'
    
    # Create stabilizer
    stabilizer = HomographyStabilizer(
        smoothing_window=smoothing_window,
        ransac_threshold=ransac_threshold
    )
    
    # Stabilize
    success = stabilizer.stabilize_video(
        input_path,
        output_path,
        mask_path,
        preview=preview
    )
    
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
