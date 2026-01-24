#!/bin/bash

echo "========================================"
echo "Emotion Dataset Creation Pipeline"
echo "========================================"

# Step 1: Download videos
echo "[1/5] Downloading videos..."
python3 download_videos.py

# Step 2: Extract frames with face detection
echo "[2/5] Extracting frames..."
python3 extract_frames.py

# Step 3: Cluster by identity
echo "[3/5] Clustering identities..."
python3 cluster_identities.py

# Step 4: Generate labels
echo "[4/5] Generating labels..."
python3 generate_labels.py

# Step 5: Prepare training pairs
echo "[5/5] Preparing training pairs..."
python3 prepare_training_pairs.py

echo "========================================"
echo "Pipeline complete!"
echo "========================================"