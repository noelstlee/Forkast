#!/usr/bin/env python3
"""
Cleanup Intermediate Preprocessing Files

This script removes intermediate files that are no longer needed after preprocessing.
It keeps only the final files required for model training.

Files to KEEP:
    - data/processed/ga/biz_ga.parquet (business metadata)
    - data/processed/ga/xgboost_data/* (all XGBoost training files)
    - data/processed/ga/lstm_data/category_*.parquet (category LSTM files)
    - data/processed/ga/lstm_data/business_*.parquet (business LSTM files)
    - data/processed/ga/lstm_data/*_vocab.json (vocabularies)
    - data/processed/ga/lstm_data/biz_ga.parquet (business metadata copy)

Files to REMOVE (intermediate):
    - data/processed/ga/reviews_ga.parquet
    - data/processed/ga/user_sequences_ga.parquet
    - data/processed/ga/user_sequences_filtered_ga.parquet
    - data/processed/ga/pairs_ga.parquet
    - data/processed/ga/pairs_filtered_ga.parquet
    - data/processed/ga/features_ga.parquet
    - data/processed/ga/lstm_data/train.parquet
    - data/processed/ga/lstm_data/val.parquet
    - data/processed/ga/lstm_data/test.parquet

Author: Team 15
Date: October 2025
"""

from pathlib import Path
import shutil

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def main():
    """Remove intermediate preprocessing files."""
    
    print("=" * 80)
    print("CLEANUP INTERMEDIATE PREPROCESSING FILES")
    print("=" * 80)
    
    base_dir = Path("data/processed/ga")
    
    # Files to remove
    intermediate_files = [
        base_dir / "reviews_ga.parquet",
        base_dir / "user_sequences_ga.parquet",
        base_dir / "user_sequences_filtered_ga.parquet",
        base_dir / "pairs_ga.parquet",
        base_dir / "pairs_filtered_ga.parquet",
        base_dir / "features_ga.parquet",
        base_dir / "lstm_data" / "train.parquet",
        base_dir / "lstm_data" / "val.parquet",
        base_dir / "lstm_data" / "test.parquet",
    ]
    
    # Calculate total size to be freed
    total_size = 0
    files_to_remove = []
    
    print("\n📁 Scanning for intermediate files...")
    for file_path in intermediate_files:
        if file_path.exists():
            size_mb = get_file_size_mb(file_path)
            total_size += size_mb
            files_to_remove.append((file_path, size_mb))
            print(f"  Found: {file_path.name} ({size_mb:.1f} MB)")
    
    if not files_to_remove:
        print("\n✓ No intermediate files found. Already clean!")
        return
    
    print(f"\n💾 Total space to be freed: {total_size:.1f} MB ({total_size / 1024:.2f} GB)")
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will permanently delete intermediate files.")
    print("   Final training files will be preserved.")
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Cleanup cancelled.")
        return
    
    # Remove files
    print("\n🗑️  Removing intermediate files...")
    removed_count = 0
    removed_size = 0
    
    for file_path, size_mb in files_to_remove:
        try:
            file_path.unlink()
            removed_count += 1
            removed_size += size_mb
            print(f"  ✓ Removed: {file_path.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Failed to remove {file_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"\n✓ Removed {removed_count} files")
    print(f"✓ Freed {removed_size:.1f} MB ({removed_size / 1024:.2f} GB)")
    
    # List remaining files
    print("\n📁 Remaining files for training:")
    
    # XGBoost files
    xgb_dir = base_dir / "xgboost_data"
    if xgb_dir.exists():
        print("\n  XGBoost Data:")
        for file_path in sorted(xgb_dir.glob("*.parquet")):
            size_mb = get_file_size_mb(file_path)
            print(f"    ✓ {file_path.name} ({size_mb:.1f} MB)")
    
    # LSTM files
    lstm_dir = base_dir / "lstm_data"
    if lstm_dir.exists():
        print("\n  LSTM Data:")
        for file_path in sorted(lstm_dir.glob("category_*.parquet")) + sorted(lstm_dir.glob("business_*.parquet")):
            size_mb = get_file_size_mb(file_path)
            print(f"    ✓ {file_path.name} ({size_mb:.1f} MB)")
        
        print("\n  Vocabularies:")
        for file_path in sorted(lstm_dir.glob("*_vocab.json")):
            size_kb = file_path.stat().st_size / 1024
            print(f"    ✓ {file_path.name} ({size_kb:.1f} KB)")
        
        # Business metadata
        biz_file = lstm_dir / "biz_ga.parquet"
        if biz_file.exists():
            size_mb = get_file_size_mb(biz_file)
            print(f"    ✓ {biz_file.name} ({size_mb:.1f} MB)")
    
    # Root business metadata
    biz_file = base_dir / "biz_ga.parquet"
    if biz_file.exists():
        size_mb = get_file_size_mb(biz_file)
        print(f"\n  Business Metadata:")
        print(f"    ✓ {biz_file.name} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 80)
    print("✓✓✓ CLEANUP COMPLETE ✓✓✓")
    print("=" * 80)
    print("\nYour data is now optimized for model training! 🚀")


if __name__ == "__main__":
    main()

