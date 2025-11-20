import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import ingest, sequences
from src.data.filter_quality import filter_sequences_only
from src.data import split_data_atlanta, lstm_preprocessing


def main():
    """Run complete LSTM preprocessing pipeline with Atlanta-specific splitting."""
    
    print("=" * 80)
    print("LSTM PREPROCESSING PIPELINE (ATLANTA-SPECIFIC)")
    print("=" * 80)
    print("\nThis will run:")
    print("  Phase A1: Ingest raw data")
    print("  Phase A2: Generate sequences")
    print("  Phase A2.5: Filter sequences by quality")
    print("  Phase A4: Split data (Atlanta-specific)")
    print("  Phase A5: LSTM preprocessing (windowing + vocabularies + class weights)")
    print("\nEstimated time: 10-15 minutes")
    print("=" * 80)
    
    try:
        # Phase A1: Ingest
        print("\n\n")
        print("▶ PHASE A1: INGESTING DATA")
        print("=" * 80)
        ingest.main()
        
        # Phase A2: Sequences
        print("\n\n")
        print("▶ PHASE A2: GENERATING SEQUENCES")
        print("=" * 80)
        sequences.main()
        
        # Phase A2.5: Filter
        print("\n\n")
        print("▶ PHASE A2.5: FILTERING SEQUENCES")
        print("=" * 80)
        filter_sequences_only()
        
        # Phase A4: Split (Atlanta-specific)
        print("\n\n")
        print("▶ PHASE A4: SPLITTING DATA (ATLANTA-SPECIFIC)")
        print("=" * 80)
        split_data_atlanta.main()
        
        # Phase A5: LSTM preprocessing
        print("\n\n")
        print("▶ PHASE A5: LSTM PREPROCESSING")
        print("=" * 80)
        lstm_preprocessing.main()
        
        # Success
        print("\n\n")
        print("=" * 80)
        print("✓✓✓ ALL PHASES COMPLETE ✓✓✓")
        print("=" * 80)
        print("\n📁 Output directory: data/processed/ga/lstm_data/")
        print("\n📊 Generated files:")
        print("  ├── train.parquet, val.parquet, test.parquet (raw splits)")
        print("  ├── category_train/val/test.parquet (windowed)")
        print("  ├── business_train/val/test.parquet (windowed)")
        print("  ├── category_vocab.json, business_vocab.json")
        print("  ├── category_class_weights.json (for imbalance handling)")
        print("  ├── atlanta_business_ids.json")
        print("  └── biz_ga.parquet")
        print("\n🚀 Next steps:")
        print("  1. Commit and push lstm_data/ to Git")
        print("  2. Pull on your VM with GPU access")
        print("  3. Run training notebooks:")
        print("     - notebooks/lstm_business_training.ipynb")
        print("     - notebooks/lstm_category_training.ipynb")
        print("=" * 80)
        
    except Exception as e:
        print("\n\n")
        print("=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPipeline failed. Please fix the error and try again.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

