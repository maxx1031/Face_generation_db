import os
import yt_dlp
import json
from pathlib import Path

# Minimum resolution requirement (skip videos below this)
MIN_RESOLUTION = 720

# Your video list with emotions
VIDEO_MANIFEST = {
    "neutral": [
        {"url": "https://www.youtube.com/watch?v=1uIBRC3xMEM", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=oQkQv1ULyzs", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=qhZFtomQaxw", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=ooouJuoeQ1c", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=OqpumEZmxa0", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=7tW66KxBOaQ", "notes": ""},
    ],
    "happy": [
        {"url": "https://www.youtube.com/watch?v=PVvCLgoeN1E", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=PFBrK7pbqCU", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=JDQcnQpxOj8", "notes": "multi-ID"},
        {"url": "https://www.youtube.com/watch?v=EDEeyk50nDQ", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=V4cpZlFESeA", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=Jygs4gYq0Gs", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=HTSejaU_TrM", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=u6HRhOttQG4", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=LdwClcDvVao", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=17Km-P0IJZY", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=QaFqZQ6JQhs", "notes": ""},
    ],
    "sad": [
        {"url": "https://www.youtube.com/watch?v=ZchRHfDklTk", "notes": "start at 1:39"},
        {"url": "https://www.youtube.com/watch?v=RW0a7Xn8hf8", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=QeWifFsvr8o", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=vUuAbRVVZwA", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=j_a_zvQOrIE", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=gjuizikJ2bk", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=IPXeiS1tzr4", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=1BimnTWx1b8", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=wpQ4R1jlFHs", "notes": ""},
    ],
    "angry": [
        {"url": "https://www.youtube.com/watch?v=2dD7upKpLks", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=aDJgv1iARPg", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=FFlejx9vekI", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=hXgLWvxgxqM", "notes": "after 1:54"},
        {"url": "https://www.youtube.com/watch?v=z_PLcTH6TEI", "notes": "perfect"},
    ],
    "fear": [
        {"url": "https://www.youtube.com/watch?v=J90JKBCDzSs", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=R1k_1p3mitg", "notes": ""},
    ],
    "surprise": [
        {"url": "https://www.youtube.com/watch?v=0wLgQQGob1U", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=jBvjrK-pgrA", "notes": ""},
        {"url": "https://www.youtube.com/watch?v=ar_o_qS68oA", "notes": ""},
    ],
}

def check_video_resolution(url: str) -> tuple[int, bool]:
    """
    Check video's maximum available resolution before downloading.

    Args:
        url: YouTube video URL

    Returns:
        (max_height, is_acceptable): Maximum height in pixels and whether it meets MIN_RESOLUTION
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Get all available formats
            formats = info.get('formats', [])

            # Find maximum height
            max_height = 0
            for fmt in formats:
                height = fmt.get('height')
                if height and isinstance(height, int):
                    max_height = max(max_height, height)

            is_acceptable = max_height >= MIN_RESOLUTION
            return max_height, is_acceptable

    except Exception as e:
        print(f"    [WARN] Could not check resolution: {e}")
        # Default to acceptable if check fails (will try to download)
        return 0, True


def download_videos(base_dir="emotion_dataset", check_resolution=True, best_quality=False, force_redownload=False):
    """
    Download videos from YouTube.

    Args:
        base_dir: Base directory for dataset
        check_resolution: Check resolution before downloading (skip < 720p)
        best_quality: Download best available quality (ignore 720p limit)
        force_redownload: Re-download even if file exists
    """
    base_path = Path(base_dir)
    raw_video_path = base_path / "raw_videos"
    metadata_path = base_path / "metadata"

    # Create directories
    for emotion in VIDEO_MANIFEST.keys():
        (raw_video_path / emotion).mkdir(parents=True, exist_ok=True)
    metadata_path.mkdir(parents=True, exist_ok=True)

    # Save manifest
    with open(metadata_path / "video_manifest.json", "w") as f:
        json.dump(VIDEO_MANIFEST, f, indent=2)
    
    # Download options
    if best_quality:
        # Best available quality (no height limit)
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
        }
        print("[MODE] Downloading BEST QUALITY (no resolution limit)")
    else:
        # Limited to 720p
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]',
            'merge_output_format': 'mp4',
        }
        print("[MODE] Downloading up to 720p")

    for emotion, videos in VIDEO_MANIFEST.items():
        print(f"\n{'='*50}")
        print(f"Downloading {emotion.upper()} videos...")
        print(f"{'='*50}")

        for idx, video_info in enumerate(videos):
            url = video_info["url"]
            video_id = url.split("v=")[-1].split("&")[0]
            output_path = raw_video_path / emotion / f"{video_id}.mp4"

            if output_path.exists() and not force_redownload:
                print(f"  [SKIP] {video_id} already exists")
                continue

            if output_path.exists() and force_redownload:
                print(f"  [REDOWNLOAD] {video_id} - removing old file...")
                output_path.unlink()

            # Check resolution before downloading
            if check_resolution and not best_quality:
                print(f"  [CHECK] {video_id} - checking resolution...")
                max_height, is_acceptable = check_video_resolution(url)

                if not is_acceptable:
                    print(f"  [SKIP] {video_id} - resolution too low ({max_height}p < {MIN_RESOLUTION}p)")
                    continue
                else:
                    print(f"  [OK] {video_id} - resolution acceptable ({max_height}p)")

            ydl_opts['outtmpl'] = str(output_path).replace('.mp4', '.%(ext)s')
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                print(f"  [OK] {video_id}")
            except Exception as e:
                print(f"  [ERROR] {video_id}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download YouTube videos')
    parser.add_argument('--best-quality', action='store_true', help='Download best available quality')
    parser.add_argument('--force', action='store_true', help='Force re-download existing files')
    parser.add_argument('--no-check', action='store_true', help='Skip resolution check')
    args = parser.parse_args()

    download_videos(
        check_resolution=not args.no_check,
        best_quality=args.best_quality,
        force_redownload=args.force
    )