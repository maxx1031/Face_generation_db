"""
Face Quality Assessment Module

Evaluates face images based on:
1. Sharpness (Laplacian variance)
2. Frontality (yaw, pitch, roll angles)
3. Identity strength (detection confidence)
4. Face ratio (face area / image area)

References:
- InstantID: uses InsightFace antelopev2 for face analysis
- PuLID: contrastive alignment for ID customization
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class FaceQualityScore:
    """Quality scores for a face image"""
    sharpness: float          # Laplacian variance (higher = sharper)
    frontality: float         # 0-1, higher = more frontal
    identity_strength: float  # Detection confidence 0-1
    face_ratio: float         # Face area / total area

    # Raw pose values
    yaw: float    # Left-right rotation (-90 to 90)
    pitch: float  # Up-down rotation (-90 to 90)
    roll: float   # Tilt rotation (-90 to 90)

    # Composite score
    composite: float = 0.0

    def compute_composite(
        self,
        w_sharpness: float = 0.30,
        w_frontality: float = 0.35,
        w_identity: float = 0.25,
        w_face_ratio: float = 0.10
    ) -> float:
        """Compute weighted composite score"""
        # Normalize sharpness to 0-1 range (typical values 0-1000+)
        norm_sharpness = min(self.sharpness / 500.0, 1.0)

        self.composite = (
            w_sharpness * norm_sharpness +
            w_frontality * self.frontality +
            w_identity * self.identity_strength +
            w_face_ratio * min(self.face_ratio * 5, 1.0)  # Normalize face ratio
        )
        return self.composite


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute image sharpness using Laplacian variance.
    Higher values indicate sharper images.

    Args:
        image: BGR or grayscale image

    Returns:
        Laplacian variance score
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return float(variance)


def compute_frontality(yaw: float, pitch: float, roll: float) -> float:
    """
    Compute frontality score from pose angles.

    Args:
        yaw: Left-right rotation (-90 to 90 degrees)
        pitch: Up-down rotation (-90 to 90 degrees)
        roll: Tilt rotation (-90 to 90 degrees)

    Returns:
        Frontality score 0-1 (1 = perfectly frontal)
    """
    # Weight yaw more heavily (side profile is worse for identity)
    weighted_deviation = (
        abs(yaw) * 1.5 +   # Yaw is most important
        abs(pitch) * 1.0 +  # Pitch moderately important
        abs(roll) * 0.5     # Roll least important
    )

    # Normalize: max weighted deviation = 1.5*90 + 1.0*90 + 0.5*90 = 270
    max_deviation = 270.0
    frontality = 1.0 - (weighted_deviation / max_deviation)

    return max(0.0, frontality)


def assess_face_quality(
    face_image: np.ndarray,
    face_info: dict,
    image_shape: Tuple[int, int, int]
) -> FaceQualityScore:
    """
    Assess the quality of a detected face.

    Args:
        face_image: Cropped face image (BGR)
        face_info: InsightFace detection result containing:
            - bbox: [x1, y1, x2, y2]
            - det_score: detection confidence
            - pose: [pitch, yaw, roll] in degrees
            - embedding: face embedding vector
        image_shape: Original frame shape (H, W, C)

    Returns:
        FaceQualityScore with all metrics
    """
    # 1. Sharpness
    sharpness = compute_sharpness(face_image)

    # 2. Pose / Frontality
    # InsightFace pose format: [pitch, yaw, roll]
    if hasattr(face_info, 'pose') and face_info.pose is not None:
        pitch, yaw, roll = face_info.pose
    else:
        # Default to frontal if pose not available
        pitch, yaw, roll = 0.0, 0.0, 0.0

    frontality = compute_frontality(yaw, pitch, roll)

    # 3. Identity strength (detection confidence)
    if hasattr(face_info, 'det_score'):
        identity_strength = float(face_info.det_score)
    else:
        identity_strength = 0.5  # Default

    # 4. Face ratio
    bbox = face_info.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
    image_area = image_shape[0] * image_shape[1]
    face_ratio = face_area / image_area if image_area > 0 else 0.0

    # Create score object
    score = FaceQualityScore(
        sharpness=sharpness,
        frontality=frontality,
        identity_strength=identity_strength,
        face_ratio=face_ratio,
        yaw=yaw,
        pitch=pitch,
        roll=roll
    )

    # Compute composite
    score.compute_composite()

    return score


def is_quality_acceptable(
    score: FaceQualityScore,
    min_sharpness: float = 50.0,
    min_frontality: float = 0.5,
    min_identity: float = 0.5,
    max_yaw: float = 45.0,
    max_pitch: float = 30.0
) -> bool:
    """
    Check if face quality meets minimum thresholds.

    Args:
        score: FaceQualityScore object
        min_sharpness: Minimum Laplacian variance
        min_frontality: Minimum frontality score (0-1)
        min_identity: Minimum detection confidence
        max_yaw: Maximum allowed yaw angle (degrees)
        max_pitch: Maximum allowed pitch angle (degrees)

    Returns:
        True if quality is acceptable
    """
    if score.sharpness < min_sharpness:
        return False

    if score.frontality < min_frontality:
        return False

    if score.identity_strength < min_identity:
        return False

    if abs(score.yaw) > max_yaw:
        return False

    if abs(score.pitch) > max_pitch:
        return False

    return True


def compute_background_hash(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    hash_size: int = 8
) -> np.ndarray:
    """
    Compute a perceptual hash of the background region (excluding face).
    Used for ensuring background diversity.

    Args:
        frame: Full frame image
        face_bbox: Face bounding box [x1, y1, x2, y2]
        hash_size: Size of the hash (default 8x8)

    Returns:
        Binary hash array
    """
    x1, y1, x2, y2 = face_bbox

    # Create mask for background
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    mask[y1:y2, x1:x2] = 0  # Exclude face region

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply mask
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    # Resize for hashing
    resized = cv2.resize(gray_masked, (hash_size, hash_size), interpolation=cv2.INTER_AREA)

    # Compute difference hash
    avg = resized.mean()
    hash_array = (resized > avg).astype(np.uint8).flatten()

    return hash_array


def background_similarity(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """
    Compute similarity between two background hashes.

    Returns:
        Similarity score 0-1 (1 = identical backgrounds)
    """
    if hash1.shape != hash2.shape:
        return 0.0

    matches = np.sum(hash1 == hash2)
    similarity = matches / len(hash1)

    return similarity


class QualityTracker:
    """
    Track quality scores for multiple faces and select top-k.
    Supports both quality-based and diversity-based selection.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.faces: List[dict] = []  # List of {score, path, embedding, bg_hash, ...}

    def add_face(
        self,
        score: FaceQualityScore,
        image_path: str,
        embedding: np.ndarray,
        bg_hash: Optional[np.ndarray] = None,
        frame_index: int = 0,
        metadata: Optional[dict] = None
    ):
        """Add a face to the tracker"""
        entry = {
            'score': score,
            'composite': score.composite,
            'path': image_path,
            'embedding': embedding,
            'bg_hash': bg_hash,
            'frame_index': frame_index,
            'metadata': metadata or {}
        }
        self.faces.append(entry)

    def get_top_quality(self, k: Optional[int] = None) -> List[dict]:
        """Get top-k faces by quality score"""
        k = k or self.top_k
        sorted_faces = sorted(self.faces, key=lambda x: x['composite'], reverse=True)
        return sorted_faces[:k]

    def get_diverse_selection(
        self,
        k: Optional[int] = None,
        bg_similarity_threshold: float = 0.8
    ) -> List[dict]:
        """
        Get top-k faces ensuring background diversity.

        Greedy selection: pick highest quality, then pick next highest
        that has different background, etc.
        """
        k = k or self.top_k
        sorted_faces = sorted(self.faces, key=lambda x: x['composite'], reverse=True)

        selected = []
        for face in sorted_faces:
            if len(selected) >= k:
                break

            # Check background similarity with already selected
            is_diverse = True
            if face['bg_hash'] is not None:
                for sel in selected:
                    if sel['bg_hash'] is not None:
                        sim = background_similarity(face['bg_hash'], sel['bg_hash'])
                        if sim > bg_similarity_threshold:
                            is_diverse = False
                            break

            if is_diverse:
                selected.append(face)

        # If not enough diverse faces, fill with remaining top quality
        if len(selected) < k:
            remaining = [f for f in sorted_faces if f not in selected]
            selected.extend(remaining[:k - len(selected)])

        return selected

    def get_temporal_clusters(
        self,
        n_clusters: int = 3,
        frames_per_cluster: int = 3
    ) -> List[List[dict]]:
        """
        Group faces into temporal clusters (front/middle/end of video).
        Returns top-quality faces from each cluster.

        Useful for: same background, different expressions (temporal continuity)
        """
        if not self.faces:
            return []

        # Sort by frame index
        sorted_by_time = sorted(self.faces, key=lambda x: x['frame_index'])

        # Divide into clusters
        total = len(sorted_by_time)
        cluster_size = max(1, total // n_clusters)

        clusters = []
        for i in range(n_clusters):
            start = i * cluster_size
            end = start + cluster_size if i < n_clusters - 1 else total
            cluster_faces = sorted_by_time[start:end]

            # Sort by quality within cluster
            cluster_sorted = sorted(cluster_faces, key=lambda x: x['composite'], reverse=True)
            clusters.append(cluster_sorted[:frames_per_cluster])

        return clusters

    def clear(self):
        """Clear all tracked faces"""
        self.faces = []


# Sharpness thresholds for reference
SHARPNESS_THRESHOLDS = {
    'very_blurry': 20,
    'blurry': 50,
    'acceptable': 100,
    'sharp': 200,
    'very_sharp': 500
}

# Frontality thresholds
FRONTALITY_THRESHOLDS = {
    'profile': 0.3,       # Side view
    'three_quarter': 0.5, # 3/4 view
    'near_frontal': 0.7,  # Almost frontal
    'frontal': 0.85       # Full frontal
}

# Preset configurations
QUALITY_PRESETS = {
    'strict': {
        'min_sharpness': 100,
        'min_frontality': 0.6,
        'min_identity': 0.6,
        'max_yaw': 30,
        'max_pitch': 25,
        'description': 'Strict: high quality frontal faces only'
    },
    'moderate': {
        'min_sharpness': 50,
        'min_frontality': 0.5,
        'min_identity': 0.5,
        'max_yaw': 45,
        'max_pitch': 30,
        'description': 'Moderate: balanced quality and quantity'
    },
    'relaxed': {
        'min_sharpness': 30,
        'min_frontality': 0.4,
        'min_identity': 0.4,
        'max_yaw': 60,
        'max_pitch': 45,
        'description': 'Relaxed: more faces, lower quality threshold'
    }
}


def get_preset(name: str = 'strict') -> dict:
    """Get a quality threshold preset by name"""
    return QUALITY_PRESETS.get(name, QUALITY_PRESETS['strict'])
