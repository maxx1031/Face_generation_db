#!/bin/bash

# =============================================================================
# Face Data Generation Pipeline v2
# =============================================================================
#
# Features:
# - Resolution pre-check before downloading (skip < 720p)
# - Fixed sampling: 90 frames per video (front/middle/end)
# - Strict quality filtering (sharpness, frontality, identity)
# - 1GB data size limit
# - Per-person selection: Top 10 quality + 5 temporal frames
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "Face Data Generation Pipeline v2"
echo "=============================================="
echo ""

# Step 1: Download videos (with resolution check)
echo "[Step 1/4] Downloading videos..."
echo "  - Checking resolution before download"
echo "  - Skipping videos < 720p"
python download_videos.py
echo ""

# Step 2: Extract frames with quality filtering
echo "[Step 2/4] Extracting frames with quality filtering..."
echo "  - Fixed sampling: 90 frames per video"
echo "  - Strict quality thresholds"
echo "  - Target size: 1GB"
python extract_frames_v2.py
echo ""

# Step 3: Cluster identities
echo "[Step 3/4] Clustering identities..."
python cluster_identities.py
echo ""

# Step 4: Select top quality per person
echo "[Step 4/4] Selecting top quality faces..."
echo "  - Top 10 high quality faces per person"
echo "  - 5 temporal sequence frames per person"
python select_top_quality.py
echo ""

echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Output structure:"
echo "  emotion_dataset/"
echo "  ├── raw_videos/      # Downloaded videos"
echo "  ├── frames/          # Extracted face frames"
echo "  ├── processed/       # Clustered by person"
echo "  ├── final/           # Selected faces"
echo "  │   └── person_XXXX/"
echo "  │       ├── high_quality/      # Top 10"
echo "  │       └── temporal_sequence/ # 5 frames"
echo "  └── metadata/"
echo "      ├── quality_scores.json"
echo "      ├── identity_map.json"
echo "      └── selection_results.json"
