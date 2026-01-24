import pandas as pd
from pathlib import Path
import json

def generate_labels_csv(base_dir="emotion_dataset"):
    """
    Generate a comprehensive labels CSV for the dataset.
    This aligns with the video-ID-emotion multilabel sets needed for Step A.
    """
    base_path = Path(base_dir)
    processed_path = base_path / "processed"
    metadata_path = base_path / "metadata"
    
    records = []
    
    for person_dir in processed_path.iterdir():
        if not person_dir.is_dir():
            continue
        
        person_id = person_dir.name
        
        for emotion_dir in person_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            
            emotion = emotion_dir.name
            
            for img_file in emotion_dir.glob("*.jpg"):
                records.append({
                    "person_id": person_id,
                    "emotion": emotion,
                    "image_path": str(img_file.relative_to(base_path)),
                    "absolute_path": str(img_file),
                    "is_anchor_candidate": True,  # Can be used in identity anchor set A
                })
    
    df = pd.DataFrame(records)
    
    # Save CSV
    df.to_csv(metadata_path / "labels.csv", index=False)
    
    # Generate summary statistics
    summary = df.groupby(["person_id", "emotion"]).size().unstack(fill_value=0)
    summary.to_csv(metadata_path / "summary.csv")
    
    print("\nDataset Statistics:")
    print("=" * 60)
    print(f"Total images: {len(df)}")
    print(f"Unique identities: {df['person_id'].nunique()}")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print(f"\nPer-identity emotion coverage:")
    print(summary)
    
    return df

if __name__ == "__main__":
    generate_labels_csv()