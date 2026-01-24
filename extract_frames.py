import cv2
import os
from pathlib import Path
from insightface.app import FaceAnalysis
import numpy as np
from tqdm import tqdm

def extract_frames_with_faces(
    base_dir="emotion_dataset",
    fps_sample=2,  # Extract 2 frames per second
    min_face_size=100,
    output_size=(512, 512)
):
    base_path = Path(base_dir)
    raw_video_path = base_path / "raw_videos"
    frames_path = base_path / "frames"
    
    # Initialize face detector
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    for emotion_dir in raw_video_path.iterdir():
        if not emotion_dir.is_dir():
            continue
        
        emotion = emotion_dir.name
        print(f"\nProcessing {emotion} videos...")
        
        for video_file in emotion_dir.glob("*.mp4"):
            video_id = video_file.stem
            output_dir = frames_path / emotion / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            cap = cv2.VideoCapture(str(video_file))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / fps_sample)
            
            frame_count = 0
            saved_count = 0
            
            pbar = tqdm(desc=f"  {video_id}", unit="frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Detect faces
                    faces = face_app.get(frame)
                    
                    for face_idx, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        
                        # Check face size
                        if (x2 - x1) < min_face_size or (y2 - y1) < min_face_size:
                            continue
                        
                        # Expand bbox for context
                        margin = int((x2 - x1) * 0.3)
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(frame.shape[1], x2 + margin)
                        y2 = min(frame.shape[0], y2 + margin)
                        
                        # Crop and resize
                        face_img = frame[y1:y2, x1:x2]
                        face_img = cv2.resize(face_img, output_size)
                        
                        # Save with face embedding for later clustering
                        frame_name = f"frame_{saved_count:05d}_face_{face_idx}.jpg"
                        cv2.imwrite(str(output_dir / frame_name), face_img)
                        
                        # Save embedding for ID clustering
                        emb_name = f"frame_{saved_count:05d}_face_{face_idx}.npy"
                        np.save(str(output_dir / emb_name), face.embedding)
                        
                        saved_count += 1
                
                frame_count += 1
                pbar.update(1)
            
            cap.release()
            pbar.close()
            print(f"    Saved {saved_count} face frames")

if __name__ == "__main__":
    extract_frames_with_faces()