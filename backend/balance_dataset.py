"""
Dataset Balancer - Creates balanced copy of chest_xray dataset
This solves the 3:1 imbalance problem by undersampling majority class
"""

import os
import shutil
from pathlib import Path
import random

print("="*80)
print("DATASET BALANCER - Creating Balanced Training Set")
print("="*80)

# Paths
SOURCE_BASE = Path('..') / 'chest_xray'
TARGET_BASE = Path('..') / 'chest_xray_balanced'

# Create target directories
print("\n📁 Creating balanced dataset structure...")
for split in ['train', 'val', 'test']:
    for category in ['NORMAL', 'PNEUMONIA']:
        target_dir = TARGET_BASE / split / category
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {target_dir}")

# Function to copy files
def copy_files(source_dir, target_dir, count=None, seed=42):
    """Copy files from source to target, optionally limiting count"""
    source_files = list(source_dir.glob('*.jpeg')) + list(source_dir.glob('*.jpg'))
    
    # Filter out .DS_Store and other non-images
    source_files = [f for f in source_files if f.suffix.lower() in ['.jpeg', '.jpg', '.png']]
    
    if count and count < len(source_files):
        # Undersample
        random.seed(seed)
        source_files = random.sample(source_files, count)
        print(f"   Undersampled {len(source_files)} from {len(list(source_dir.glob('*.jpeg')))}")
    
    for file in source_files:
        shutil.copy2(file, target_dir / file.name)
    
    return len(source_files)

# Balance TRAINING set
print("\n⚖️  Balancing TRAINING set...")

train_normal_source = SOURCE_BASE / 'train' / 'NORMAL'
train_pneumonia_source = SOURCE_BASE / 'train' / 'PNEUMONIA'

train_normal_count = len(list(train_normal_source.glob('*.jpeg'))) + len(list(train_normal_source.glob('*.jpg')))
train_pneumonia_count = len(list(train_pneumonia_source.glob('*.jpeg'))) + len(list(train_pneumonia_source.glob('*.jpg')))

print(f"\n   Original counts:")
print(f"   NORMAL: {train_normal_count}")
print(f"   PNEUMONIA: {train_pneumonia_count}")
print(f"   Ratio: 1:{train_pneumonia_count/train_normal_count:.2f}")

# Strategy: Match both to the NORMAL count (undersample pneumonia)
target_count = train_normal_count

print(f"\n   Target count: {target_count} for each class")

# Copy all NORMAL (minority class)
copied_normal = copy_files(
    train_normal_source,
    TARGET_BASE / 'train' / 'NORMAL',
    count=None
)

# Undersample PNEUMONIA to match NORMAL
copied_pneumonia = copy_files(
    train_pneumonia_source,
    TARGET_BASE / 'train' / 'PNEUMONIA',
    count=target_count
)

print(f"\n   ✅ Balanced training set:")
print(f"   NORMAL: {copied_normal}")
print(f"   PNEUMONIA: {copied_pneumonia}")
print(f"   New ratio: 1:{copied_pneumonia/copied_normal:.2f}")

# Copy VALIDATION and TEST as-is (small, don't need balancing)
print("\n📋 Copying VALIDATION set (as-is)...")
val_normal = copy_files(
    SOURCE_BASE / 'val' / 'NORMAL',
    TARGET_BASE / 'val' / 'NORMAL'
)
val_pneumonia = copy_files(
    SOURCE_BASE / 'val' / 'PNEUMONIA',
    TARGET_BASE / 'val' / 'PNEUMONIA'
)
print(f"   NORMAL: {val_normal}, PNEUMONIA: {val_pneumonia}")

print("\n📋 Copying TEST set (as-is)...")
test_normal = copy_files(
    SOURCE_BASE / 'test' / 'NORMAL',
    TARGET_BASE / 'test' / 'NORMAL'
)
test_pneumonia = copy_files(
    SOURCE_BASE / 'test' / 'PNEUMONIA',
    TARGET_BASE / 'test' / 'PNEUMONIA'
)
print(f"   NORMAL: {test_normal}, PNEUMONIA: {test_pneumonia}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

total_original = train_normal_count + train_pneumonia_count
total_balanced = copied_normal + copied_pneumonia

print(f"\nTraining Set:")
print(f"   Original: {train_normal_count + train_pneumonia_count:,} images (imbalanced)")
print(f"   Balanced: {copied_normal + copied_pneumonia:,} images (balanced 1:1)")
print(f"   Removed: {total_original - total_balanced:,} pneumonia images")

print(f"\nValidation Set: {val_normal + val_pneumonia} images")
print(f"Test Set: {test_normal + test_pneumonia} images")

print(f"\nTotal dataset size: {total_balanced + val_normal + val_pneumonia + test_normal + test_pneumonia:,} images")

# Calculate disk space
import shutil as sh
total, used, free = sh.disk_usage(TARGET_BASE)
size_mb = sum(f.stat().st_size for f in TARGET_BASE.rglob('*') if f.is_file()) / (1024**2)

print(f"\nBalanced dataset size: {size_mb:.1f} MB")
print(f"Location: {TARGET_BASE.absolute()}")

print("\n" + "="*80)
print("✅ BALANCED DATASET CREATED!")
print("="*80)
print("\nNext step: python training_script_balanced.py")
print("="*80)