"""
Generate trajectories using:
1. Low Confidence (0.15) -> To see the scooters
2. High Resolution (1920) -> To make the scooters visible to the model

This does NOT use 'sticky' tracking. It relies purely on better detection visibility.
"""

from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import cv2

def generate_tracks_lowconf_highres(
    video_path="uploads\\D2F1_stab_10sec.mp4",
    model_path= r"C:\Users\srini\Downloads\visdrone_finetuned.pt",
    output_path="trajectories_final_ig.csv",
    conf_thresh=0.15,
    img_size=1920, # The High Res Magic
    frames_to_process=200 # FULL VIDEO
):
    print(f"🚀 Starting Tracking: LowConf({conf_thresh}) + HighRes({img_size})...")
    print(f"   Region: Ground Truth Lane Only (Y=899-1065)")
    print(f"   Model: {model_path}")
    
    model = YOLO(model_path)
    
    tracks = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Total Frames: {total_frames}")
    
    pbar = tqdm(total=min(total_frames, frames_to_process))
    frame_idx = 0
    
    try:
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret: break
            
            # Run Tracking
            results = model.track(
                frame, 
                persist=True, 
                conf=conf_thresh, 
                imgsz=img_size,  # Force High Res
                verbose=False,
                tracker="botsort.yaml" # Default tracker
            )[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # FILTER: Only keep detections in the Ground Truth Lane
                    if 899 <= cy <= 1065:
                        tracks.append({
                            "Frame": frame_idx,
                            "VehicleID": int(track_id),
                            "Class": int(cls),
                            "X_pixel": cx,
                            "Y_pixel": cy,
                            "Time": frame_idx / 30.0
                        })
                    
            frame_idx += 1
            pbar.update(1)
            
            # Explicit Logging for User
            if frame_idx % 50 == 0:
                # Use tqdm.write so it doesn't break the progress bar layout
                tqdm.write(f"   Processed {frame_idx}/{total_frames} frames. ({len(tracks)} detections so far)")
                
            # Intermediate Save (Safety)
            if frame_idx % 500 == 0 and tracks:
                pd.DataFrame(tracks).to_csv("trajectories_final_full_PARTIAL.csv", index=False)
                tqdm.write(f"   [Saved backup: trajectories_final_full_PARTIAL.csv]")
            
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        pbar.close()
    
    if tracks:
        df = pd.DataFrame(tracks)
        df.to_csv(output_path, index=False)
        print(f"\n✅ DONE! Saved final detections to {output_path}", flush=True)
        print(f"   Total Unique IDs: {df['VehicleID'].nunique()}", flush=True)
        print(f"   Total Frames: {frame_idx}", flush=True)
    else:
        print("\n❌ No detections found!")

if __name__ == "__main__":
    generate_tracks_lowconf_highres()
