import pandas as pd
from pathlib import Path
import random
import json
from itertools import combinations

def prepare_training_pairs(base_dir="emotion_dataset", min_emotions_per_id=2):
    """
    Prepare training pairs for emotion editing.
    
    For Step A (Pseudo-Data Generation), we need:
    - Identity anchor set A (high-quality reference photos)
    - Source image b (input image)
    - Target emotion e
    - Ground truth (same person with target emotion)
    """
    base_path = Path(base_dir)
    metadata_path = base_path / "metadata"
    
    df = pd.read_csv(metadata_path / "labels.csv")
    
    training_pairs = []
    
    # Group by person
    for person_id, person_df in df.groupby("person_id"):
        emotions_available = person_df["emotion"].unique()
        
        # Skip if person doesn't have multiple emotions
        if len(emotions_available) < min_emotions_per_id:
            continue
        
        # For each emotion pair, create training samples
        for source_emotion, target_emotion in combinations(emotions_available, 2):
            source_images = person_df[person_df["emotion"] == source_emotion]["image_path"].tolist()
            target_images = person_df[person_df["emotion"] == target_emotion]["image_path"].tolist()
            
            # Create pairs
            for src_img in source_images[:5]:  # Limit to avoid explosion
                for tgt_img in target_images[:5]:
                    # Select anchor images (other images of same person)
                    anchor_candidates = person_df[
                        ~person_df["image_path"].isin([src_img, tgt_img])
                    ]["image_path"].tolist()
                    
                    # Sample anchor set A
                    anchor_set = random.sample(
                        anchor_candidates, 
                        min(5, len(anchor_candidates))
                    )
                    
                    training_pairs.append({
                        "person_id": person_id,
                        "source_image": src_img,
                        "source_emotion": source_emotion,
                        "target_image": tgt_img,  # Ground truth
                        "target_emotion": target_emotion,
                        "anchor_set": anchor_set,
                    })
    
    # Save training pairs
    with open(metadata_path / "training_pairs.json", "w") as f:
        json.dump(training_pairs, f, indent=2)
    
    print(f"Generated {len(training_pairs)} training pairs")
    
    # Also create a simplified CSV version
    pairs_df = pd.DataFrame([
        {
            "person_id": p["person_id"],
            "source_image": p["source_image"],
            "source_emotion": p["source_emotion"],
            "target_image": p["target_image"],
            "target_emotion": p["target_emotion"],
            "num_anchors": len(p["anchor_set"]),
        }
        for p in training_pairs
    ])
    pairs_df.to_csv(metadata_path / "training_pairs.csv", index=False)
    
    return training_pairs

if __name__ == "__main__":
    prepare_training_pairs()