
import openneuro
import os
from pathlib import Path

# Settings
DATASET_ID = 'ds004889'
TARGET_DIR = Path(__file__).parent / 'datasets'

print(f"Downloading full dataset from {DATASET_ID}...")
print(f"Target directory: {TARGET_DIR}")

# Download
openneuro.download(
    dataset=DATASET_ID,
    target_dir=str(TARGET_DIR),
    # include=include_patterns,
    verify_hash=False
)

print("Download complete!")
