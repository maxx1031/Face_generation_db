import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
import shutil
import json

def cluster_by_identity(base_dir="emotion_dataset", similarity_threshold=0.6):
    """
    Cluster faces by identity using embeddings.
    This helps separate multi-ID videos automatically.
    """
    base_path = Path(base_dir)
    frames_path = base_path / "frames"
    processed_path = base_path / "processed"
    
    # Collect all embeddings
    all_embeddings = []
    all_paths = []
    all_emotions = []
    
    for emotion_dir in frames_path.iterdir():
        if not emotion_dir.is_dir():
            continue
        
        emotion = emotion_dir.name
        
        for video_dir in emotion_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            for emb_file in video_dir.glob("*.npy"):
                embedding = np.load(str(emb_file))
                img_path = emb_file.with_suffix(".jpg")
                
                if img_path.exists():
                    all_embeddings.append(embedding)
                    all_paths.append(img_path)
                    all_emotions.append(emotion)
    
    if not all_embeddings:
        print("No embeddings found!")
        return
    
    # Cluster using DBSCAN
    embeddings_array = np.array(all_embeddings)
    embeddings_normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Convert similarity to distance
    clustering = DBSCAN(
        eps=1 - similarity_threshold,
        min_samples=3,
        metric='cosine'
    ).fit(embeddings_normalized)
    
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print(f"Found {n_clusters} unique identities")
    
    # Organize by person ID
    identity_map = {}
    
    for idx, (label, img_path, emotion) in enumerate(zip(labels, all_paths, all_emotions)):
        if label == -1:  # Noise
            continue
        
        person_id = f"person_{label:04d}"
        
        if person_id not in identity_map:
            identity_map[person_id] = {}
        
        if emotion not in identity_map[person_id]:
            identity_map[person_id][emotion] = []
        
        identity_map[person_id][emotion].append(str(img_path))
    
    # Copy to processed folder
    for person_id, emotions in identity_map.items():
        for emotion, paths in emotions.items():
            person_emotion_dir = processed_path / person_id / emotion
            person_emotion_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, src_path in enumerate(paths):
                dst_path = person_emotion_dir / f"{idx:05d}.jpg"
                shutil.copy(src_path, dst_path)
    
    # Save identity map
    metadata_path = base_path / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    with open(metadata_path / "identity_map.json", "w") as f:
        json.dump(identity_map, f, indent=2)
    
    # Print summary
    print("\nDataset Summary:")
    print("-" * 50)
    for person_id, emotions in identity_map.items():
        emotion_counts = {e: len(p) for e, p in emotions.items()}
        print(f"{person_id}: {emotion_counts}")

if __name__ == "__main__":
    cluster_by_identity()