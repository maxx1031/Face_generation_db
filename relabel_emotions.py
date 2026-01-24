import cv2
import numpy as np
from pathlib import Path
from deepface import DeepFace
import shutil
from tqdm import tqdm
import json

def relabel_by_detected_emotion(base_dir="sample_dataset", confidence_threshold=0.3):
    """
    Re-label faces by detected emotion using DeepFace model.
    This replaces the video-level emotion label with per-frame detected emotion.
    """
    base_path = Path(base_dir)
    processed_path = base_path / "processed"
    relabeled_path = base_path / "relabeled"
    metadata_path = base_path / "metadata"

    # Emotion mapping
    emotion_map = {}
    stats = {
        'happy': 0, 'sad': 0, 'angry': 0, 'fear': 0,
        'surprise': 0, 'neutral': 0, 'disgust': 0, 'skipped': 0
    }

    # Process each person
    for person_dir in tqdm(list(processed_path.iterdir()), desc="Processing persons"):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        emotion_map[person_id] = {}

        # Collect all images for this person
        for emotion_dir in person_dir.iterdir():
            if not emotion_dir.is_dir():
                continue

            for img_file in emotion_dir.glob("*.jpg"):
                try:
                    # Analyze emotion using DeepFace
                    result = DeepFace.analyze(
                        str(img_file),
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )

                    if isinstance(result, list):
                        result = result[0]

                    # Get dominant emotion
                    dominant_emotion = result['dominant_emotion']
                    confidence = result['emotion'][dominant_emotion] / 100.0

                    if confidence < confidence_threshold:
                        stats['skipped'] += 1
                        continue

                    # Track for this person
                    if dominant_emotion not in emotion_map[person_id]:
                        emotion_map[person_id][dominant_emotion] = []

                    emotion_map[person_id][dominant_emotion].append({
                        'src': str(img_file),
                        'confidence': float(confidence)
                    })
                    stats[dominant_emotion] += 1

                except Exception as e:
                    stats['skipped'] += 1
                    continue

    # Copy to relabeled folder
    print("\nCopying to relabeled folder...")
    for person_id, emotions in tqdm(emotion_map.items()):
        for emotion, files in emotions.items():
            out_dir = relabeled_path / person_id / emotion
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, file_info in enumerate(files):
                dst = out_dir / f"{idx:05d}.jpg"
                shutil.copy(file_info['src'], dst)

    # Save emotion map
    metadata_path.mkdir(parents=True, exist_ok=True)
    with open(metadata_path / "relabeled_map.json", "w") as f:
        json.dump(emotion_map, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Emotion Detection Summary")
    print("="*60)
    print(f"Total detected: {sum(v for k,v in stats.items() if k != 'skipped')}")
    print(f"Skipped (low confidence): {stats['skipped']}")
    print("\nEmotion distribution:")
    for emotion, count in sorted(stats.items(), key=lambda x: -x[1]):
        if emotion != 'skipped' and count > 0:
            print(f"  {emotion}: {count}")

    # Per-person summary
    print("\nPer-person emotion coverage:")
    print("-"*60)
    multi_emotion_persons = 0
    for person_id, emotions in emotion_map.items():
        if len(emotions) >= 2:
            multi_emotion_persons += 1
        emotion_counts = {e: len(f) for e, f in emotions.items()}
        if emotion_counts:
            print(f"{person_id}: {emotion_counts}")

    print(f"\nPersons with 2+ emotions: {multi_emotion_persons}/{len(emotion_map)}")

    return emotion_map

if __name__ == "__main__":
    relabel_by_detected_emotion()
