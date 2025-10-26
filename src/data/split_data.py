"""
Phase A4: Temporal Data Splitting
Splits datasets chronologically into train/validation/test sets.
Organizes data into xgboost_data/ and lstm_data/ directories.
"""

import polars as pl
from pathlib import Path
import shutil


def create_directories(base_dir: Path):
    """Create xgboost_data and lstm_data directories."""
    xgb_dir = base_dir / "xgboost_data"
    lstm_dir = base_dir / "lstm_data"
    
    xgb_dir.mkdir(exist_ok=True)
    lstm_dir.mkdir(exist_ok=True)
    
    return xgb_dir, lstm_dir


def temporal_split_xgboost(features_df: pl.DataFrame, 
                           train_ratio: float = 0.70,
                           val_ratio: float = 0.15) -> tuple:
    """
    Split XGBoost features chronologically by timestamp.
    
    Args:
        features_df: Features DataFrame with src_ts
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n" + "=" * 80)
    print("SPLITTING XGBOOST DATA (TEMPORAL)")
    print("=" * 80)
    
    # Sort by source timestamp
    print("\n[1/4] Sorting by timestamp...")
    features_df = features_df.sort("src_ts")
    
    total_samples = len(features_df)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    print(f"\n[2/4] Calculating split indices...")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Train size: {train_size:,} ({train_ratio*100:.1f}%)")
    print(f"  Val size: {val_size:,} ({val_ratio*100:.1f}%)")
    print(f"  Test size: {total_samples - train_size - val_size:,} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    # Split
    print(f"\n[3/4] Splitting data...")
    train_df = features_df[:train_size]
    val_df = features_df[train_size:train_size + val_size]
    test_df = features_df[train_size + val_size:]
    
    # Print temporal ranges
    print(f"\n[4/4] Temporal ranges:")
    print(f"  Train: {train_df['src_ts'].min()} to {train_df['src_ts'].max()}")
    print(f"  Val:   {val_df['src_ts'].min()} to {val_df['src_ts'].max()}")
    print(f"  Test:  {test_df['src_ts'].min()} to {test_df['src_ts'].max()}")
    
    # Check label distribution
    print(f"\n  Label distribution:")
    print(f"  Train - Pos: {(train_df['label'] == 1).sum():,}, Neg: {(train_df['label'] == 0).sum():,}")
    print(f"  Val   - Pos: {(val_df['label'] == 1).sum():,}, Neg: {(val_df['label'] == 0).sum():,}")
    print(f"  Test  - Pos: {(test_df['label'] == 1).sum():,}, Neg: {(test_df['label'] == 0).sum():,}")
    
    return train_df, val_df, test_df


def temporal_split_lstm(sequences_df: pl.DataFrame,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15) -> tuple:
    """
    Split LSTM sequences chronologically by user's last visit timestamp.
    
    Args:
        sequences_df: User sequences DataFrame
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n" + "=" * 80)
    print("SPLITTING LSTM DATA (TEMPORAL BY USER)")
    print("=" * 80)
    
    # Get last timestamp per user
    print("\n[1/5] Finding last visit timestamp per user...")
    user_last_ts = sequences_df.group_by("user_id").agg([
        pl.col("ts").max().alias("last_ts")
    ])
    
    # Sort users by their last visit
    print("\n[2/5] Sorting users by last visit...")
    user_last_ts = user_last_ts.sort("last_ts")
    
    total_users = len(user_last_ts)
    train_users = int(total_users * train_ratio)
    val_users = int(total_users * val_ratio)
    
    print(f"\n[3/5] Calculating split indices...")
    print(f"  Total users: {total_users:,}")
    print(f"  Train users: {train_users:,} ({train_ratio*100:.1f}%)")
    print(f"  Val users: {val_users:,} ({val_ratio*100:.1f}%)")
    print(f"  Test users: {total_users - train_users - val_users:,} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    # Split users
    train_user_ids = set(user_last_ts[:train_users]["user_id"].to_list())
    val_user_ids = set(user_last_ts[train_users:train_users + val_users]["user_id"].to_list())
    test_user_ids = set(user_last_ts[train_users + val_users:]["user_id"].to_list())
    
    # Split sequences
    print(f"\n[4/5] Splitting sequences by user...")
    train_df = sequences_df.filter(pl.col("user_id").is_in(list(train_user_ids)))
    val_df = sequences_df.filter(pl.col("user_id").is_in(list(val_user_ids)))
    test_df = sequences_df.filter(pl.col("user_id").is_in(list(test_user_ids)))
    
    # Print statistics
    print(f"\n[5/5] Split statistics:")
    print(f"  Train: {len(train_df):,} visits from {train_df['user_id'].n_unique():,} users")
    print(f"  Val:   {len(val_df):,} visits from {val_df['user_id'].n_unique():,} users")
    print(f"  Test:  {len(test_df):,} visits from {test_df['user_id'].n_unique():,} users")
    
    # Print temporal ranges
    print(f"\n  Temporal ranges:")
    print(f"  Train: {train_df['ts'].min()} to {train_df['ts'].max()}")
    print(f"  Val:   {val_df['ts'].min()} to {val_df['ts'].max()}")
    print(f"  Test:  {test_df['ts'].min()} to {test_df['ts'].max()}")
    
    return train_df, val_df, test_df


def save_xgboost_splits(train_df: pl.DataFrame, val_df: pl.DataFrame, 
                        test_df: pl.DataFrame, output_dir: Path):
    """Save XGBoost train/val/test splits."""
    print("\n" + "=" * 80)
    print("SAVING XGBOOST SPLITS")
    print("=" * 80)
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    print(f"\n[1/3] Saving training data...")
    train_df.write_parquet(train_path, compression="snappy")
    print(f"  {train_path}")
    print(f"  Size: {train_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n[2/3] Saving validation data...")
    val_df.write_parquet(val_path, compression="snappy")
    print(f"  {val_path}")
    print(f"  Size: {val_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n[3/3] Saving test data...")
    test_df.write_parquet(test_path, compression="snappy")
    print(f"  {test_path}")
    print(f"  Size: {test_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n✓ XGBoost splits saved to {output_dir}")


def save_lstm_splits(train_df: pl.DataFrame, val_df: pl.DataFrame,
                     test_df: pl.DataFrame, output_dir: Path):
    """Save LSTM train/val/test splits."""
    print("\n" + "=" * 80)
    print("SAVING LSTM SPLITS")
    print("=" * 80)
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    print(f"\n[1/3] Saving training sequences...")
    train_df.write_parquet(train_path, compression="snappy")
    print(f"  {train_path}")
    print(f"  Size: {train_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n[2/3] Saving validation sequences...")
    val_df.write_parquet(val_path, compression="snappy")
    print(f"  {val_path}")
    print(f"  Size: {val_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n[3/3] Saving test sequences...")
    test_df.write_parquet(test_path, compression="snappy")
    print(f"  {test_path}")
    print(f"  Size: {test_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n✓ LSTM splits saved to {output_dir}")


def copy_business_data(source_path: Path, xgb_dir: Path, lstm_dir: Path):
    """Copy business metadata to both directories."""
    print("\n" + "=" * 80)
    print("COPYING BUSINESS METADATA")
    print("=" * 80)
    
    xgb_biz_path = xgb_dir / "biz_ga.parquet"
    lstm_biz_path = lstm_dir / "biz_ga.parquet"
    
    print(f"\n[1/2] Copying to xgboost_data/...")
    shutil.copy2(source_path, xgb_biz_path)
    print(f"  {xgb_biz_path}")
    print(f"  Size: {xgb_biz_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n[2/2] Copying to lstm_data/...")
    shutil.copy2(source_path, lstm_biz_path)
    print(f"  {lstm_biz_path}")
    print(f"  Size: {lstm_biz_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n✓ Business metadata copied to both directories")


def print_final_summary(xgb_dir: Path, lstm_dir: Path):
    """Print final summary of data organization."""
    print("\n" + "=" * 80)
    print("FINAL DATA ORGANIZATION")
    print("=" * 80)
    
    print(f"\n📁 XGBoost Data ({xgb_dir}):")
    print(f"   ├── train.parquet     (70% of data, chronological)")
    print(f"   ├── val.parquet       (15% of data, chronological)")
    print(f"   ├── test.parquet      (15% of data, chronological)")
    print(f"   └── biz_ga.parquet    (business metadata)")
    
    print(f"\n📁 LSTM Data ({lstm_dir}):")
    print(f"   ├── train.parquet     (70% of users by last visit)")
    print(f"   ├── val.parquet       (15% of users by last visit)")
    print(f"   ├── test.parquet      (15% of users by last visit)")
    print(f"   └── biz_ga.parquet    (business metadata)")
    
    # Calculate total sizes
    xgb_size = sum(f.stat().st_size for f in xgb_dir.glob("*.parquet")) / 1024 / 1024
    lstm_size = sum(f.stat().st_size for f in lstm_dir.glob("*.parquet")) / 1024 / 1024
    
    print(f"\n📊 Storage:")
    print(f"   XGBoost: {xgb_size:.1f} MB")
    print(f"   LSTM: {lstm_size:.1f} MB")
    print(f"   Total: {xgb_size + lstm_size:.1f} MB")


def main():
    """Main data splitting pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "data" / "processed" / "ga"
    
    features_input = processed_dir / "features_ga.parquet"
    sequences_input = processed_dir / "user_sequences_filtered_ga.parquet"
    biz_input = processed_dir / "biz_ga.parquet"
    
    print("=" * 80)
    print("PHASE A4: TEMPORAL DATA SPLITTING")
    print("=" * 80)
    print(f"\nInputs:")
    print(f"  - {features_input}")
    print(f"  - {sequences_input}")
    print(f"  - {biz_input}")
    
    # Create directories
    print(f"\nCreating output directories...")
    xgb_dir, lstm_dir = create_directories(processed_dir)
    print(f"  ✓ {xgb_dir}")
    print(f"  ✓ {lstm_dir}")
    
    # Load data
    print(f"\n[Loading data...]")
    features_df = pl.read_parquet(features_input)
    sequences_df = pl.read_parquet(sequences_input)
    print(f"  Loaded {len(features_df):,} XGBoost samples")
    print(f"  Loaded {len(sequences_df):,} LSTM sequences")
    
    # Split XGBoost data (chronological by timestamp)
    xgb_train, xgb_val, xgb_test = temporal_split_xgboost(
        features_df, 
        train_ratio=0.70, 
        val_ratio=0.15
    )
    
    # Split LSTM data (chronological by user's last visit)
    lstm_train, lstm_val, lstm_test = temporal_split_lstm(
        sequences_df,
        train_ratio=0.70,
        val_ratio=0.15
    )
    
    # Save splits
    save_xgboost_splits(xgb_train, xgb_val, xgb_test, xgb_dir)
    save_lstm_splits(lstm_train, lstm_val, lstm_test, lstm_dir)
    
    # Copy business metadata
    copy_business_data(biz_input, xgb_dir, lstm_dir)
    
    # Print final summary
    print_final_summary(xgb_dir, lstm_dir)
    
    print("\n✓✓✓ PHASE A4 COMPLETE ✓✓✓\n")


def split_xgboost_only():
    """Split only XGBoost data (for consolidated pipeline)."""
    print("\n[Loading data...]")
    xgboost_df = pl.read_parquet(PROCESSED_DIR / "features_ga.parquet")
    biz_df = pl.read_parquet(PROCESSED_DIR / "biz_ga.parquet")
    
    print(f"  XGBoost: {len(xgboost_df):,} samples")
    
    # Create directories
    XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Split XGBoost data
    print("\n[Splitting XGBoost data...]")
    xgboost_train, xgboost_val, xgboost_test = split_xgboost_data(xgboost_df)
    
    # Save XGBoost splits
    print("\n[Saving XGBoost splits...]")
    xgboost_train.write_parquet(XGBOOST_DIR / "train.parquet")
    xgboost_val.write_parquet(XGBOOST_DIR / "val.parquet")
    xgboost_test.write_parquet(XGBOOST_DIR / "test.parquet")
    print(f"  ✓ Saved train.parquet ({len(xgboost_train):,} samples)")
    print(f"  ✓ Saved val.parquet ({len(xgboost_val):,} samples)")
    print(f"  ✓ Saved test.parquet ({len(xgboost_test):,} samples)")
    
    # Copy business metadata
    print("\n[Copying business metadata...]")
    biz_df.write_parquet(XGBOOST_DIR / "biz_ga.parquet")
    print(f"  ✓ Copied biz_ga.parquet to xgboost_data/")


def split_lstm_only():
    """Split only LSTM data (for consolidated pipeline)."""
    print("\n[Loading data...]")
    lstm_sequences_df = pl.read_parquet(PROCESSED_DIR / "user_sequences_filtered_ga.parquet")
    biz_df = pl.read_parquet(PROCESSED_DIR / "biz_ga.parquet")
    
    print(f"  LSTM: {len(lstm_sequences_df):,} visits from {lstm_sequences_df['user_id'].n_unique():,} users")
    
    # Create directories
    LSTM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Split LSTM data
    print("\n[Splitting LSTM data...]")
    lstm_train, lstm_val, lstm_test = split_lstm_data(lstm_sequences_df)
    
    # Save LSTM splits
    print("\n[Saving LSTM splits...]")
    lstm_train.write_parquet(LSTM_DIR / "train.parquet")
    lstm_val.write_parquet(LSTM_DIR / "val.parquet")
    lstm_test.write_parquet(LSTM_DIR / "test.parquet")
    print(f"  ✓ Saved train.parquet ({len(lstm_train):,} visits, {lstm_train['user_id'].n_unique():,} users)")
    print(f"  ✓ Saved val.parquet ({len(lstm_val):,} visits, {lstm_val['user_id'].n_unique():,} users)")
    print(f"  ✓ Saved test.parquet ({len(lstm_test):,} visits, {lstm_test['user_id'].n_unique():,} users)")
    
    # Copy business metadata
    print("\n[Copying business metadata...]")
    biz_df.write_parquet(LSTM_DIR / "biz_ga.parquet")
    print(f"  ✓ Copied biz_ga.parquet to lstm_data/")


if __name__ == "__main__":
    main()

