"""
Enhanced Frame Extraction with Quality Filtering

Features:
1. Fixed sampling: N frames per segment (front/middle/end)
2. Quality assessment: sharpness, frontality, identity strength
3. Data size control: stops when reaching target size (default 1GB)
4. Saves quality metadata for downstream filtering
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from tqdm import tqdm
from dataclasses import asdict
from typing import Optional

from quality_filter import (
    assess_face_quality,
    is_quality_acceptable,
    compute_background_hash,
    FaceQualityScore
)

# ============== Configuration ==============

# Sampling strategy
FRAMES_PER_SEGMENT = 30      # Frames to sample per segment
N_SEGMENTS = 3               # Number of segments (front/middle/end)
TOTAL_SAMPLES_PER_VIDEO = FRAMES_PER_SEGMENT * N_SEGMENTS  # 90

# Quality thresholds (STRICT)
MIN_SHARPNESS = 100          # Laplacian variance threshold
MIN_FRONTALITY = 0.6         # Frontality score 0-1
MIN_IDENTITY_SCORE = 0.6     # Detection confidence
MAX_YAW = 30                 # Maximum yaw angle (degrees)
MAX_PITCH = 25               # Maximum pitch angle (degrees)

# Face detection
MIN_FACE_SIZE = 100          # Minimum face size in pixels
OUTPUT_SIZE = (512, 512)     # Output face image size

# Data size control
MAX_DATA_SIZE_GB = 1.0       # Stop when reaching this size
MAX_FACES_PER_VIDEO = 50     # Maximum faces to save per video

# ============================================


def get_directory_size_gb(path: Path) -> float:
    """Get total size of directory in GB"""
    total = 0
    for file in path.rglob('*'):
        if file.is_file():
            total += file.stat().st_size
    return total / (1024 ** 3)


def compute_sample_indices(total_frames: int, n_segments: int, frames_per_segment: int) -> list:
    """
    Compute frame indices to sample from video.

    Divides video into n_segments and samples frames_per_segment from each.

    Args:
        total_frames: Total frames in video
        n_segments: Number of segments (e.g., 3 for front/middle/end)
        frames_per_segment: Frames to sample per segment

    Returns:
        List of frame indices to sample
    """
    if total_frames < n_segments * frames_per_segment:
        # Video too short, sample uniformly
        step = max(1, total_frames // (n_segments * frames_per_segment))
        return list(range(0, total_frames, step))

    indices = []
    segment_size = total_frames // n_segments

    for seg in range(n_segments):
        seg_start = seg * segment_size
        seg_end = seg_start + segment_size if seg < n_segments - 1 else total_frames

        # Sample uniformly within segment
        seg_frames = seg_end - seg_start
        step = max(1, seg_frames // frames_per_segment)

        for i in range(frames_per_segment):
            idx = seg_start + i * step
            if idx < total_frames:
                indices.append(idx)

    return sorted(set(indices))


def extract_frames_with_quality(
    base_dir: str = "emotion_dataset",
    max_size_gb: float = MAX_DATA_SIZE_GB,
    frames_per_segment: int = FRAMES_PER_SEGMENT,
    n_segments: int = N_SEGMENTS
):
    """
    Extract frames with quality filtering and size control.

    Args:
        base_dir: Base directory for dataset
        max_size_gb: Maximum data size in GB before stopping
        frames_per_segment: Frames to sample per segment
        n_segments: Number of segments per video
    """
    base_path = Path(base_dir)
    raw_video_path = base_path / "raw_videos"
    frames_path = base_path / "frames"
    metadata_path = base_path / "metadata"

    metadata_path.mkdir(parents=True, exist_ok=True)

    # Initialize face detector
    print("Initializing face detector...")
    face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Track statistics
    stats = {
        'total_frames_processed': 0,
        'total_faces_detected': 0,
        'faces_passed_quality': 0,
        'faces_failed_quality': 0,
        'videos_processed': 0,
        'videos_skipped_size_limit': 0,
        'quality_failures': {
            'sharpness': 0,
            'frontality': 0,
            'identity': 0,
            'yaw': 0,
            'pitch': 0
        }
    }

    # Quality metadata for all saved faces
    all_quality_data = {}

    # Process each emotion directory
    for emotion_dir in sorted(raw_video_path.iterdir()):
        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        print(f"\n{'='*50}")
        print(f"Processing {emotion.upper()} videos...")
        print(f"{'='*50}")

        for video_file in sorted(emotion_dir.glob("*.mp4")):
            # Check data size limit
            current_size = get_directory_size_gb(frames_path)
            if current_size >= max_size_gb:
                print(f"\n[STOP] Data size limit reached: {current_size:.2f}GB >= {max_size_gb}GB")
                stats['videos_skipped_size_limit'] += 1
                break

            video_id = video_file.stem
            output_dir = frames_path / emotion / video_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(str(video_file))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames == 0:
                print(f"  [SKIP] {video_id} - no frames")
                cap.release()
                continue

            # Compute sample indices
            sample_indices = compute_sample_indices(total_frames, n_segments, frames_per_segment)

            print(f"\n  Processing: {video_id}")
            print(f"    Total frames: {total_frames}, FPS: {video_fps:.1f}")
            print(f"    Sampling {len(sample_indices)} frames from {n_segments} segments")

            # Track per-video stats
            video_faces_saved = 0
            video_quality_data = []

            # Process sampled frames
            pbar = tqdm(sample_indices, desc=f"    {video_id}", unit="frame")

            for frame_idx in pbar:
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                stats['total_frames_processed'] += 1

                # Detect faces
                faces = face_app.get(frame)

                for face_idx, face in enumerate(faces):
                    stats['total_faces_detected'] += 1

                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox

                    # Check face size
                    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                        continue

                    # Expand bbox for context
                    margin = int((x2 - x1) * 0.3)
                    x1_exp = max(0, x1 - margin)
                    y1_exp = max(0, y1 - margin)
                    x2_exp = min(frame.shape[1], x2 + margin)
                    y2_exp = min(frame.shape[0], y2 + margin)

                    # Crop face
                    face_img = frame[y1_exp:y2_exp, x1_exp:x2_exp]

                    # Assess quality
                    quality_score = assess_face_quality(face_img, face, frame.shape)

                    # Check quality thresholds
                    is_acceptable = True
                    if quality_score.sharpness < MIN_SHARPNESS:
                        stats['quality_failures']['sharpness'] += 1
                        is_acceptable = False
                    if quality_score.frontality < MIN_FRONTALITY:
                        stats['quality_failures']['frontality'] += 1
                        is_acceptable = False
                    if quality_score.identity_strength < MIN_IDENTITY_SCORE:
                        stats['quality_failures']['identity'] += 1
                        is_acceptable = False
                    if abs(quality_score.yaw) > MAX_YAW:
                        stats['quality_failures']['yaw'] += 1
                        is_acceptable = False
                    if abs(quality_score.pitch) > MAX_PITCH:
                        stats['quality_failures']['pitch'] += 1
                        is_acceptable = False

                    if not is_acceptable:
                        stats['faces_failed_quality'] += 1
                        continue

                    # Passed quality check
                    stats['faces_passed_quality'] += 1

                    # Check per-video limit
                    if video_faces_saved >= MAX_FACES_PER_VIDEO:
                        break

                    # Resize to output size
                    face_img_resized = cv2.resize(face_img, OUTPUT_SIZE)

                    # Compute background hash for diversity tracking
                    bg_hash = compute_background_hash(frame, (x1, y1, x2, y2))

                    # Save face image
                    frame_name = f"frame_{frame_idx:05d}_face_{face_idx}.jpg"
                    cv2.imwrite(str(output_dir / frame_name), face_img_resized)

                    # Save embedding
                    emb_name = f"frame_{frame_idx:05d}_face_{face_idx}.npy"
                    np.save(str(output_dir / emb_name), face.embedding)

                    # Record quality data
                    quality_entry = {
                        'filename': frame_name,
                        'frame_index': frame_idx,
                        'face_index': face_idx,
                        'sharpness': quality_score.sharpness,
                        'frontality': quality_score.frontality,
                        'identity_strength': quality_score.identity_strength,
                        'face_ratio': quality_score.face_ratio,
                        'yaw': quality_score.yaw,
                        'pitch': quality_score.pitch,
                        'roll': quality_score.roll,
                        'composite_score': quality_score.composite,
                        'bg_hash': bg_hash.tolist()
                    }
                    video_quality_data.append(quality_entry)

                    video_faces_saved += 1

                # Update progress
                pbar.set_postfix({'saved': video_faces_saved})

            cap.release()
            pbar.close()

            # Save video quality metadata
            if video_quality_data:
                quality_key = f"{emotion}/{video_id}"
                all_quality_data[quality_key] = video_quality_data

            stats['videos_processed'] += 1
            print(f"    Saved {video_faces_saved} faces (passed quality filter)")

        # Check size limit after each emotion
        if get_directory_size_gb(frames_path) >= max_size_gb:
            print(f"\n[STOP] Data size limit reached")
            break

    # Save all quality metadata
    quality_metadata_file = metadata_path / "quality_scores.json"
    with open(quality_metadata_file, 'w') as f:
        json.dump(all_quality_data, f, indent=2)

    # Save statistics
    stats_file = metadata_path / "extraction_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Videos processed: {stats['videos_processed']}")
    print(f"Total frames processed: {stats['total_frames_processed']}")
    print(f"Total faces detected: {stats['total_faces_detected']}")
    print(f"Faces passed quality: {stats['faces_passed_quality']}")
    print(f"Faces failed quality: {stats['faces_failed_quality']}")
    print(f"\nQuality failure breakdown:")
    for reason, count in stats['quality_failures'].items():
        print(f"  - {reason}: {count}")
    print(f"\nFinal data size: {get_directory_size_gb(frames_path):.2f} GB")
    print(f"Quality metadata saved to: {quality_metadata_file}")


if __name__ == "__main__":
    extract_frames_with_quality()
