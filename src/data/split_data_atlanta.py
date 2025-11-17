"""
Phase A4: Atlanta-Specific Data Splitting for LSTM

This script implements a special splitting strategy for the business LSTM model:
- Filters out ALL users who visited Atlanta businesses from training/validation
- Uses Atlanta users as the test set (for final inference/visualization)
- Ensures zero data leakage for the venue manager dashboard use case

For category sequences, uses standard temporal split (no Atlanta filtering needed).

Input:
    - data/processed/ga/user_sequences_ga.parquet
    - data/processed/ga/biz_ga.parquet

Output:
    - data/processed/ga/lstm_data/train.parquet (non-Atlanta users)
    - data/processed/ga/lstm_data/val.parquet (non-Atlanta users)
    - data/processed/ga/lstm_data/test.parquet (Atlanta users only)
    - data/processed/ga/lstm_data/atlanta_business_ids.json
    - data/processed/ga/lstm_data/biz_ga.parquet
"""

import polars as pl
import json
from pathlib import Path

# Atlanta geographic bounds
ATLANTA_BOUNDS = {
    "lat_min": 33.6,
    "lat_max": 34.0,
    "lon_min": -84.6,
    "lon_max": -84.2,
}


def identify_atlanta_businesses(biz_df: pl.DataFrame) -> set:
    """
    Identify all businesses within Atlanta geographic bounds.
    
    Args:
        biz_df: Business metadata DataFrame
        
    Returns:
        Set of Atlanta business IDs
    """
    print("\n[1/7] Identifying Atlanta businesses...")
    
    atlanta_biz = biz_df.filter(
        (pl.col("lat") >= ATLANTA_BOUNDS["lat_min"]) &
        (pl.col("lat") <= ATLANTA_BOUNDS["lat_max"]) &
        (pl.col("lon") >= ATLANTA_BOUNDS["lon_min"]) &
        (pl.col("lon") <= ATLANTA_BOUNDS["lon_max"])
    )
    
    atlanta_biz_ids = set(atlanta_biz["gmap_id"])
    
    print(f"  Atlanta businesses: {len(atlanta_biz_ids):,} / {len(biz_df):,} ({len(atlanta_biz_ids)/len(biz_df)*100:.1f}%)")
    print(f"  Bounds: lat [{ATLANTA_BOUNDS['lat_min']}, {ATLANTA_BOUNDS['lat_max']}]")
    print(f"          lon [{ATLANTA_BOUNDS['lon_min']}, {ATLANTA_BOUNDS['lon_max']}]")
    
    return atlanta_biz_ids


def split_users_by_atlanta(sequences_df: pl.DataFrame, atlanta_biz_ids: set) -> tuple:
    """
    Split users into Atlanta and non-Atlanta groups.
    
    Args:
        sequences_df: User sequences DataFrame
        atlanta_biz_ids: Set of Atlanta business IDs
        
    Returns:
        Tuple of (non_atlanta_users, atlanta_users)
    """
    print("\n[2/7] Splitting users by Atlanta visits...")
    
    # Find all users who visited Atlanta
    atlanta_visits = sequences_df.filter(pl.col("gmap_id").is_in(atlanta_biz_ids))
    atlanta_users = set(atlanta_visits["user_id"].unique())
    
    # Get non-Atlanta users
    all_users = set(sequences_df["user_id"].unique())
    non_atlanta_users = all_users - atlanta_users
    
    # Get visit counts
    atlanta_visit_count = len(atlanta_visits)
    non_atlanta_visit_count = len(sequences_df.filter(pl.col("user_id").is_in(non_atlanta_users)))
    
    print(f"\n  Atlanta users (test set):")
    print(f"    Users: {len(atlanta_users):,} ({len(atlanta_users)/len(all_users)*100:.1f}%)")
    print(f"    Visits: {atlanta_visit_count:,} ({atlanta_visit_count/len(sequences_df)*100:.1f}%)")
    
    print(f"\n  Non-Atlanta users (train/val):")
    print(f"    Users: {len(non_atlanta_users):,} ({len(non_atlanta_users)/len(all_users)*100:.1f}%)")
    print(f"    Visits: {non_atlanta_visit_count:,} ({non_atlanta_visit_count/len(sequences_df)*100:.1f}%)")
    
    return non_atlanta_users, atlanta_users


def temporal_split_non_atlanta(sequences_df: pl.DataFrame, 
                                non_atlanta_users: set,
                                train_ratio: float = 0.85) -> tuple:
    """
    Split non-Atlanta users temporally into train/val sets.
    
    Args:
        sequences_df: User sequences DataFrame
        non_atlanta_users: Set of non-Atlanta user IDs
        train_ratio: Proportion for training (default 0.85, rest goes to val)
        
    Returns:
        Tuple of (train_df, val_df)
    """
    print("\n[3/7] Splitting non-Atlanta users into train/val...")
    
    # Filter to non-Atlanta users only
    non_atlanta_df = sequences_df.filter(pl.col("user_id").is_in(non_atlanta_users))
    
    # Get last timestamp per user for temporal ordering
    user_last_ts = non_atlanta_df.group_by("user_id").agg([
        pl.col("ts").max().alias("last_ts")
    ]).sort("last_ts")
    
    # Split by time
    n_train = int(len(user_last_ts) * train_ratio)
    train_user_ids = set(user_last_ts[:n_train]["user_id"])
    val_user_ids = set(user_last_ts[n_train:]["user_id"])
    
    # Create splits
    train_df = non_atlanta_df.filter(pl.col("user_id").is_in(train_user_ids))
    val_df = non_atlanta_df.filter(pl.col("user_id").is_in(val_user_ids))
    
    print(f"  Train: {len(train_df):,} visits from {len(train_user_ids):,} users ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_df):,} visits from {len(val_user_ids):,} users ({(1-train_ratio)*100:.0f}%)")
    
    # Print temporal ranges
    print(f"\n  Temporal ranges:")
    print(f"    Train: {train_df['ts'].min()} to {train_df['ts'].max()}")
    print(f"    Val:   {val_df['ts'].min()} to {val_df['ts'].max()}")
    
    return train_df, val_df


def create_test_set(sequences_df: pl.DataFrame, atlanta_users: set) -> pl.DataFrame:
    """
    Create test set from Atlanta users.
    
    Args:
        sequences_df: User sequences DataFrame
        atlanta_users: Set of Atlanta user IDs
        
    Returns:
        Test DataFrame
    """
    print("\n[4/7] Creating test set from Atlanta users...")
    
    test_df = sequences_df.filter(pl.col("user_id").is_in(atlanta_users))
    
    print(f"  Test: {len(test_df):,} visits from {len(atlanta_users):,} users")
    print(f"  Temporal range: {test_df['ts'].min()} to {test_df['ts'].max()}")
    
    return test_df


def save_splits(train_df: pl.DataFrame, val_df: pl.DataFrame, 
                test_df: pl.DataFrame, output_dir: Path):
    """Save train/val/test splits."""
    print("\n[5/7] Saving splits...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.write_parquet(train_path, compression="snappy")
    print(f"  ✓ {train_path.name} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    val_df.write_parquet(val_path, compression="snappy")
    print(f"  ✓ {val_path.name} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    test_df.write_parquet(test_path, compression="snappy")
    print(f"  ✓ {test_path.name} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")


def save_atlanta_metadata(atlanta_biz_ids: set, biz_df: pl.DataFrame, output_dir: Path):
    """Save Atlanta business IDs and metadata."""
    print("\n[6/7] Saving Atlanta metadata...")
    
    # Save Atlanta business IDs as JSON
    atlanta_ids_path = output_dir / "atlanta_business_ids.json"
    with open(atlanta_ids_path, "w") as f:
        json.dump(list(atlanta_biz_ids), f, indent=2)
    print(f"  ✓ {atlanta_ids_path.name} ({len(atlanta_biz_ids):,} businesses)")
    
    # Copy business metadata
    biz_path = output_dir / "biz_ga.parquet"
    biz_df.write_parquet(biz_path, compression="snappy")
    print(f"  ✓ {biz_path.name} ({biz_path.stat().st_size / 1024 / 1024:.1f} MB)")


def print_summary(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    """Print final summary."""
    print("\n[7/7] Summary")
    print("=" * 80)
    
    total_visits = len(train_df) + len(val_df) + len(test_df)
    total_users = train_df["user_id"].n_unique() + val_df["user_id"].n_unique() + test_df["user_id"].n_unique()
    
    print(f"\n📊 FINAL SPLIT:")
    print(f"  Train:  {len(train_df):>9,} visits ({len(train_df)/total_visits*100:>5.1f}%) from {train_df['user_id'].n_unique():>7,} users")
    print(f"  Val:    {len(val_df):>9,} visits ({len(val_df)/total_visits*100:>5.1f}%) from {val_df['user_id'].n_unique():>7,} users")
    print(f"  Test:   {len(test_df):>9,} visits ({len(test_df)/total_visits*100:>5.1f}%) from {test_df['user_id'].n_unique():>7,} users (ATLANTA)")
    print(f"  Total:  {total_visits:>9,} visits (100.0%) from {total_users:>7,} users")
    
    print(f"\n✓ Zero data leakage: Atlanta businesses never seen in train/val")
    print(f"✓ Test set = Inference set for venue manager dashboard")
    print(f"✓ Model will learn Georgia patterns and generalize to Atlanta")


def main():
    """Main Atlanta-specific splitting pipeline."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "data" / "processed" / "ga"
    lstm_dir = processed_dir / "lstm_data"
    
    sequences_input = processed_dir / "user_sequences_ga.parquet"
    biz_input = processed_dir / "biz_ga.parquet"
    
    print("=" * 80)
    print("PHASE A4: ATLANTA-SPECIFIC DATA SPLITTING (LSTM)")
    print("=" * 80)
    print(f"\nInputs:")
    print(f"  - {sequences_input}")
    print(f"  - {biz_input}")
    print(f"\nOutput:")
    print(f"  - {lstm_dir}/")
    
    # Load data
    print(f"\n[Loading data...]")
    sequences_df = pl.read_parquet(sequences_input)
    biz_df = pl.read_parquet(biz_input)
    print(f"  Loaded {len(sequences_df):,} visits from {sequences_df['user_id'].n_unique():,} users")
    print(f"  Loaded {len(biz_df):,} businesses")
    
    # Identify Atlanta businesses
    atlanta_biz_ids = identify_atlanta_businesses(biz_df)
    
    # Split users by Atlanta
    non_atlanta_users, atlanta_users = split_users_by_atlanta(sequences_df, atlanta_biz_ids)
    
    # Split non-Atlanta users into train/val
    train_df, val_df = temporal_split_non_atlanta(sequences_df, non_atlanta_users, train_ratio=0.85)
    
    # Create test set from Atlanta users
    test_df = create_test_set(sequences_df, atlanta_users)
    
    # Save splits
    save_splits(train_df, val_df, test_df, lstm_dir)
    
    # Save Atlanta metadata
    save_atlanta_metadata(atlanta_biz_ids, biz_df, lstm_dir)
    
    # Print summary
    print_summary(train_df, val_df, test_df)
    
    print("\n" + "=" * 80)
    print("✓✓✓ PHASE A4 COMPLETE ✓✓✓")
    print("=" * 80)
    print("\nNext: Run Phase A5 (LSTM-specific preprocessing)")
    print("  python -m src.data.lstm_preprocessing")
    print("=" * 80)


if __name__ == "__main__":
    main()

