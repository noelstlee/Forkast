#!/usr/bin/env python3
"""
XGBoost Preprocessing Pipeline - End-to-End

This script runs the complete preprocessing pipeline for XGBoost training:
    Phase A1: Data Ingestion & Normalization
    Phase A2: User Sequence Derivation (pairs only)
    Phase A2.5: Data Quality Filtering
    Phase A3: Feature Engineering
    Phase A4: Temporal Data Splitting

Input:
    - data/raw/meta-Georgia.json
    - data/raw/review-Georgia.json

Output:
    - data/processed/ga/xgboost_data/train.parquet
    - data/processed/ga/xgboost_data/val.parquet
    - data/processed/ga/xgboost_data/test.parquet
    - data/processed/ga/xgboost_data/biz_ga.parquet

Author: Team 15
Date: October 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all preprocessing modules
from src.data import ingest, sequences, filter_quality, features, split_data

def main():
    """Run complete XGBoost preprocessing pipeline."""
    
    print("=" * 80)
    print("XGBOOST PREPROCESSING PIPELINE - END-TO-END")
    print("=" * 80)
    print("\nThis will run phases A1 → A2 → A2.5 → A3 → A4")
    print("Estimated time: ~83 minutes\n")
    
    # Phase A1: Data Ingestion & Normalization
    print("\n" + "=" * 80)
    print("PHASE A1: DATA INGESTION & NORMALIZATION")
    print("=" * 80)
    ingest.main()
    
    # Phase A2: User Sequence Derivation (pairs only)
    print("\n" + "=" * 80)
    print("PHASE A2: USER SEQUENCE DERIVATION")
    print("=" * 80)
    sequences.main()
    
    # Phase A2.5: Data Quality Filtering
    print("\n" + "=" * 80)
    print("PHASE A2.5: DATA QUALITY FILTERING")
    print("=" * 80)
    filter_quality.main()
    
    # Phase A3: Feature Engineering
    print("\n" + "=" * 80)
    print("PHASE A3: FEATURE ENGINEERING")
    print("=" * 80)
    features.main()
    
    # Phase A4: Temporal Data Splitting (XGBoost only)
    print("\n" + "=" * 80)
    print("PHASE A4: TEMPORAL DATA SPLITTING (XGBOOST)")
    print("=" * 80)
    split_data.split_xgboost_only()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓✓✓ XGBOOST PREPROCESSING COMPLETE ✓✓✓")
    print("=" * 80)
    
    # Check output files
    xgb_dir = Path("data/processed/ga/xgboost_data")
    if xgb_dir.exists():
        print("\n📁 Output files:")
        for file in sorted(xgb_dir.glob("*.parquet")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file.name} ({size_mb:.1f} MB)")
    
    print("\n🚀 Ready for Phase B1: XGBoost Training")
    print("   Run: python src/models/xgboost_ranker.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

