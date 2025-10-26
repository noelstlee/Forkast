#!/usr/bin/env python3
"""
LSTM Preprocessing Pipeline - End-to-End

This script runs the complete preprocessing pipeline for LSTM training:
    Phase A1: Data Ingestion & Normalization
    Phase A2: User Sequence Derivation (sequences only)
    Phase A2.5: Data Quality Filtering (sequences only)
    Phase A4: Temporal Data Splitting (LSTM only)
    Phase A5: LSTM-Specific Preprocessing

Input:
    - data/raw/meta-Georgia.json
    - data/raw/review-Georgia.json

Output:
    - data/processed/ga/lstm_data/category_train.parquet
    - data/processed/ga/lstm_data/category_val.parquet
    - data/processed/ga/lstm_data/category_test.parquet
    - data/processed/ga/lstm_data/business_train.parquet
    - data/processed/ga/lstm_data/business_val.parquet
    - data/processed/ga/lstm_data/business_test.parquet
    - data/processed/ga/lstm_data/category_vocab.json
    - data/processed/ga/lstm_data/business_vocab.json
    - data/processed/ga/lstm_data/biz_ga.parquet

Author: Team 15
Date: October 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all preprocessing modules
from src.data import ingest, sequences, filter_quality, split_data, lstm_preprocessing

def main():
    """Run complete LSTM preprocessing pipeline."""
    
    print("=" * 80)
    print("LSTM PREPROCESSING PIPELINE - END-TO-END")
    print("=" * 80)
    print("\nThis will run phases A1 → A2 → A2.5 → A4 → A5")
    print("Estimated time: ~50 minutes\n")
    
    # Phase A1: Data Ingestion & Normalization
    print("\n" + "=" * 80)
    print("PHASE A1: DATA INGESTION & NORMALIZATION")
    print("=" * 80)
    ingest.main()
    
    # Phase A2: User Sequence Derivation (sequences only)
    print("\n" + "=" * 80)
    print("PHASE A2: USER SEQUENCE DERIVATION")
    print("=" * 80)
    sequences.main()
    
    # Phase A2.5: Data Quality Filtering (sequences only)
    print("\n" + "=" * 80)
    print("PHASE A2.5: DATA QUALITY FILTERING (SEQUENCES)")
    print("=" * 80)
    filter_quality.filter_sequences_only()
    
    # Phase A4: Temporal Data Splitting (LSTM only)
    print("\n" + "=" * 80)
    print("PHASE A4: TEMPORAL DATA SPLITTING (LSTM)")
    print("=" * 80)
    split_data.split_lstm_only()
    
    # Phase A5: LSTM-Specific Preprocessing
    print("\n" + "=" * 80)
    print("PHASE A5: LSTM-SPECIFIC PREPROCESSING")
    print("=" * 80)
    lstm_preprocessing.main()
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓✓✓ LSTM PREPROCESSING COMPLETE ✓✓✓")
    print("=" * 80)
    
    # Check output files
    lstm_dir = Path("data/processed/ga/lstm_data")
    if lstm_dir.exists():
        print("\n📁 Output files:")
        for file in sorted(lstm_dir.glob("*.parquet")) + sorted(lstm_dir.glob("*.json")):
            size_mb = file.stat().st_size / (1024 * 1024)
            if size_mb < 0.1:
                print(f"  ✓ {file.name} (<0.1 MB)")
            else:
                print(f"  ✓ {file.name} ({size_mb:.1f} MB)")
    
    print("\n🚀 Ready for Phase B2: LSTM Training")
    print("   Run: python src/models/lstm_predictor.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

