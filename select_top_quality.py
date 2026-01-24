"""
Select Top Quality Faces per Person

After identity clustering, this script selects:
1. Top 10 highest quality faces per person (for identity anchors)
2. 5 temporal sequence frames per person (for expression disentanglement)

Input: processed/ directory with clustered identities + quality_scores.json
Output: final/ directory with selected faces organized by person
"""

import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

# Configuration
TOP_K_QUALITY = 10           # Top quality faces per person
TEMPORAL_FRAMES = 5          # Temporal sequence frames per person
BG_SIMILARITY_THRESHOLD = 0.7  # Background similarity threshold for diversity


def load_quality_scores(metadata_path: Path) -> Dict:
    """Load quality scores from metadata file"""
    quality_file = metadata_path / "quality_scores.json"
    if not quality_file.exists():
        print(f"[WARN] Quality scores file not found: {quality_file}")
        return {}

    with open(quality_file, 'r') as f:
        return json.load(f)


def load_identity_map(metadata_path: Path) -> Dict:
    """Load identity clustering results"""
    identity_file = metadata_path / "identity_map.json"
    if not identity_file.exists():
        print(f"[ERROR] Identity map not found: {identity_file}")
        return {}

    with open(identity_file, 'r') as f:
        return json.load(f)


def background_similarity(hash1: List[int], hash2: List[int]) -> float:
    """Compute similarity between two background hashes"""
    if len(hash1) != len(hash2):
        return 0.0
    matches = sum(1 for a, b in zip(hash1, hash2) if a == b)
    return matches / len(hash1)


def get_face_quality_info(
    image_path: str,
    quality_data: Dict
) -> Optional[Dict]:
    """
    Find quality info for a face image from quality_scores.json

    Args:
        image_path: Path to face image (e.g., emotion_dataset/frames/happy/video_id/frame_00001_face_0.jpg)
        quality_data: Loaded quality_scores.json data

    Returns:
        Quality info dict or None if not found
    """
    path = Path(image_path)

    # Extract emotion/video_id key
    # Path format: .../frames/emotion/video_id/frame_xxxxx_face_x.jpg
    parts = path.parts
    try:
        frames_idx = parts.index('frames')
        emotion = parts[frames_idx + 1]
        video_id = parts[frames_idx + 2]
        filename = path.name
    except (ValueError, IndexError):
        return None

    key = f"{emotion}/{video_id}"
    if key not in quality_data:
        return None

    # Find matching filename
    for entry in quality_data[key]:
        if entry['filename'] == filename:
            return entry

    return None


def select_top_quality_faces(
    faces: List[Dict],
    k: int = TOP_K_QUALITY,
    ensure_diversity: bool = False,
    bg_threshold: float = BG_SIMILARITY_THRESHOLD
) -> List[Dict]:
    """
    Select top-k faces by quality score.

    Args:
        faces: List of face dicts with quality info
        k: Number of faces to select
        ensure_diversity: If True, ensure background diversity
        bg_threshold: Background similarity threshold

    Returns:
        List of selected face dicts
    """
    # Sort by composite score (descending)
    sorted_faces = sorted(faces, key=lambda x: x.get('composite_score', 0), reverse=True)

    if not ensure_diversity:
        return sorted_faces[:k]

    # Greedy selection with diversity
    selected = []
    for face in sorted_faces:
        if len(selected) >= k:
            break

        # Check background similarity
        is_diverse = True
        face_hash = face.get('bg_hash')

        if face_hash:
            for sel in selected:
                sel_hash = sel.get('bg_hash')
                if sel_hash:
                    sim = background_similarity(face_hash, sel_hash)
                    if sim > bg_threshold:
                        is_diverse = False
                        break

        if is_diverse:
            selected.append(face)

    # Fill remaining slots if needed
    if len(selected) < k:
        remaining = [f for f in sorted_faces if f not in selected]
        selected.extend(remaining[:k - len(selected)])

    return selected


def select_temporal_sequence(
    faces: List[Dict],
    n_frames: int = TEMPORAL_FRAMES
) -> List[Dict]:
    """
    Select temporal sequence frames (consecutive frames for expression disentanglement).

    Strategy: Find the best quality cluster of consecutive frames.

    Args:
        faces: List of face dicts with frame_index
        n_frames: Number of frames to select

    Returns:
        List of selected face dicts
    """
    if len(faces) <= n_frames:
        return sorted(faces, key=lambda x: x.get('frame_index', 0))

    # Sort by frame index
    sorted_by_time = sorted(faces, key=lambda x: x.get('frame_index', 0))

    # Find best window of n_frames consecutive
    best_window = None
    best_score = -1

    for i in range(len(sorted_by_time) - n_frames + 1):
        window = sorted_by_time[i:i + n_frames]
        avg_score = np.mean([f.get('composite_score', 0) for f in window])

        if avg_score > best_score:
            best_score = avg_score
            best_window = window

    return best_window if best_window else sorted_by_time[:n_frames]


def select_top_quality_per_person(
    base_dir: str = "emotion_dataset",
    top_k: int = TOP_K_QUALITY,
    temporal_k: int = TEMPORAL_FRAMES
):
    """
    Main function to select top quality faces for each person.

    Args:
        base_dir: Base directory for dataset
        top_k: Number of top quality faces per person
        temporal_k: Number of temporal sequence frames per person
    """
    base_path = Path(base_dir)
    processed_path = base_path / "processed"
    metadata_path = base_path / "metadata"
    final_path = base_path / "final"

    if not processed_path.exists():
        print(f"[ERROR] Processed directory not found: {processed_path}")
        print("Please run cluster_identities.py first.")
        return

    # Load quality scores
    print("Loading quality scores...")
    quality_data = load_quality_scores(metadata_path)

    # Load identity map
    print("Loading identity map...")
    identity_map = load_identity_map(metadata_path)

    if not identity_map:
        print("[ERROR] No identity map found. Please run cluster_identities.py first.")
        return

    # Statistics
    stats = {
        'total_persons': 0,
        'total_faces_input': 0,
        'total_high_quality_selected': 0,
        'total_temporal_selected': 0,
        'persons_with_quality_data': 0
    }

    # Selection results
    selection_results = {}

    # Process each person
    print(f"\nSelecting faces for each person...")
    print(f"  - Top {top_k} high quality faces")
    print(f"  - {temporal_k} temporal sequence frames")

    for person_id, emotions_data in identity_map.items():
        stats['total_persons'] += 1

        # Collect all faces for this person with quality info
        person_faces = []

        for emotion, image_paths in emotions_data.items():
            for img_path in image_paths:
                stats['total_faces_input'] += 1

                # Get quality info
                quality_info = get_face_quality_info(img_path, quality_data)

                face_entry = {
                    'path': img_path,
                    'emotion': emotion,
                    'person_id': person_id
                }

                if quality_info:
                    face_entry.update(quality_info)
                else:
                    # Default quality score if not found
                    face_entry['composite_score'] = 0.5
                    face_entry['frame_index'] = 0

                person_faces.append(face_entry)

        if not person_faces:
            continue

        # Check if we have quality data
        has_quality = any('sharpness' in f for f in person_faces)
        if has_quality:
            stats['persons_with_quality_data'] += 1

        # Select top quality faces
        top_quality = select_top_quality_faces(person_faces, k=top_k, ensure_diversity=False)

        # Select temporal sequence
        temporal_seq = select_temporal_sequence(person_faces, n_frames=temporal_k)

        # Create output directories
        person_final_dir = final_path / person_id
        high_quality_dir = person_final_dir / "high_quality"
        temporal_dir = person_final_dir / "temporal_sequence"

        high_quality_dir.mkdir(parents=True, exist_ok=True)
        temporal_dir.mkdir(parents=True, exist_ok=True)

        # Copy high quality faces
        for idx, face in enumerate(top_quality):
            src_path = Path(face['path'])
            if src_path.exists():
                emotion = face.get('emotion', 'unknown')
                dst_name = f"{idx:02d}_{emotion}_{src_path.name}"
                shutil.copy2(src_path, high_quality_dir / dst_name)

                # Also copy embedding if exists
                emb_src = src_path.with_suffix('.npy')
                if emb_src.exists():
                    shutil.copy2(emb_src, high_quality_dir / dst_name.replace('.jpg', '.npy'))

                stats['total_high_quality_selected'] += 1

        # Copy temporal sequence faces
        for idx, face in enumerate(temporal_seq):
            src_path = Path(face['path'])
            if src_path.exists():
                emotion = face.get('emotion', 'unknown')
                frame_idx = face.get('frame_index', 0)
                dst_name = f"{idx:02d}_f{frame_idx:05d}_{emotion}_{src_path.name}"
                shutil.copy2(src_path, temporal_dir / dst_name)

                # Also copy embedding if exists
                emb_src = src_path.with_suffix('.npy')
                if emb_src.exists():
                    shutil.copy2(emb_src, temporal_dir / dst_name.replace('.jpg', '.npy'))

                stats['total_temporal_selected'] += 1

        # Record selection results
        selection_results[person_id] = {
            'total_faces': len(person_faces),
            'high_quality_selected': len(top_quality),
            'temporal_selected': len(temporal_seq),
            'high_quality_files': [f['path'] for f in top_quality],
            'temporal_files': [f['path'] for f in temporal_seq]
        }

        print(f"  {person_id}: {len(person_faces)} faces -> {len(top_quality)} HQ + {len(temporal_seq)} temporal")

    # Save selection results
    results_file = metadata_path / "selection_results.json"
    with open(results_file, 'w') as f:
        json.dump(selection_results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("SELECTION COMPLETE")
    print(f"{'='*50}")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Total input faces: {stats['total_faces_input']}")
    print(f"Persons with quality data: {stats['persons_with_quality_data']}")
    print(f"High quality selected: {stats['total_high_quality_selected']}")
    print(f"Temporal sequence selected: {stats['total_temporal_selected']}")
    print(f"\nOutput directory: {final_path}")
    print(f"Selection results saved to: {results_file}")


if __name__ == "__main__":
    select_top_quality_per_person()
