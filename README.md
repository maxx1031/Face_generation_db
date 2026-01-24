# Face Generation Dataset Pipeline

A Python pipeline for generating high-quality face datasets from videos. Designed for training face generation and emotion synthesis models.

## Features

- **Resolution Pre-check**: Automatically skips videos below 720p
- **Smart Frame Sampling**: Extracts 90 frames per video (front/middle/end sections)
- **Quality Filtering**: Strict filtering based on sharpness, frontality, and identity consistency
- **Identity Clustering**: Groups faces by person using InsightFace embeddings
- **Top Quality Selection**: Selects top 10 high-quality + 5 temporal sequence frames per person
- **Size Control**: Configurable dataset size limit (default: 1GB)

## Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Frontality (yaw)** | \|yaw\| < 30° | Maximum left-right head rotation |
| **Frontality (pitch)** | \|pitch\| < 25° | Maximum up-down head rotation |
| **Sharpness** | Laplacian > 100 | Minimum sharpness threshold |
| **Detection Confidence** | > 0.6 | Minimum face detection confidence |
| **Resolution** | >= 720p | Minimum video resolution (pre-download check) |
| **Frame Sampling** | 90 frames | 30 front + 30 middle + 30 end per video |
| **Target Data Size** | 1GB | Pipeline stops when reaching this limit |
| **Per-Person Selection** | Top 10 + 5 temporal | High quality anchors + temporal sequence |

## Processing Flow

```
Video URL
    │
    ▼
[Resolution Check] ─── < 720p ──→ Skip
    │
    │ >= 720p
    ▼
[Download Video]
    │
    ▼
[Fixed Sampling: 90 frames] ─── front 30 / middle 30 / end 30
    │
    ▼
[Face Detection + Quality Scoring]
    │
    ▼
[Filter] ─── blurry / side-face / low-confidence ──→ Discard
    │
    ▼
[Save Candidate Frames + Embeddings]
    │
    ▼
[Monitor Data Size] ─── >= 1GB ──→ Stop
    │
    ▼
[Identity Clustering]
    │
    ▼
[Per-Person Selection] ─── Top 10 HQ + 5 Temporal
    │
    ▼
Done
```

## Pipeline Steps

1. **Download Videos** - Fetches videos from YouTube with resolution check
2. **Extract Frames** - Samples frames and applies quality filtering
3. **Cluster Identities** - Groups extracted faces by person
4. **Select Top Quality** - Picks best frames per identity

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- yt-dlp
- OpenCV
- InsightFace
- scikit-learn
- pandas
- numpy
- tqdm
- FER (Facial Expression Recognition)
- ONNX Runtime

## Usage

Run the full pipeline:

```bash
./run_pipeline_v2.sh
```

Or run individual steps:

```bash
python download_videos.py
python extract_frames_v2.py
python cluster_identities.py
python select_top_quality.py
```

## Output Structure

```
emotion_dataset/
├── raw_videos/           # Downloaded videos
├── frames/               # Extracted face frames
├── processed/            # Clustered by person
├── final/                # Selected faces
│   └── person_XXXX/
│       ├── high_quality/       # Top 10 quality frames
│       └── temporal_sequence/  # 5 consecutive frames
└── metadata/
    ├── quality_scores.json
    ├── identity_map.json
    └── selection_results.json
```

## Emotion Categories

The pipeline supports the following emotion labels:
- Neutral
- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust

## License

MIT
