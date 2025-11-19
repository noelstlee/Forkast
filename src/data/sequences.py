"""
Phase A2: User Sequence Derivation
Creates user visit sequences and consecutive visit pairs from reviews and metadata.
"""

import polars as pl
from pathlib import Path
from typing import Optional


# Time window for consecutive visits (in hours)
MIN_DELTA_HOURS = 0
MAX_DELTA_HOURS = 168  # 7 days


def derive_user_sequences(reviews_df, biz_df, output_path):
    """
    Create user visit sequences by joining reviews with business metadata.
    
    Args:
        reviews_df: DataFrame of reviews
        biz_df: DataFrame of businesses
        output_path: Path to save user_sequences_ga.parquet
        
    Returns:
        DataFrame of user sequences
    """
    print("=" * 80)
    print("DERIVING USER SEQUENCES")
    print("=" * 80)
    
    print("\n[1/4] Joining reviews with business metadata...")
    # Join reviews with business data to get location and category
    sequences = reviews_df.join(
        biz_df.select(["gmap_id", "lat", "lon", "category_main"]),
        on="gmap_id",
        how="inner"
    )
    print(f"  Joined {len(sequences):,} reviews with business data")
    
    print("\n[2/4] Sorting by user and timestamp...")
    # Sort by user_id and timestamp
    sequences = sequences.sort(["user_id", "ts"])
    
    print("\n[3/4] Creating sequence indices...")
    # Add sequence index for each user
    sequences = sequences.with_columns([
        pl.col("user_id").cum_count().over("user_id").alias("seq_idx")
    ])
    
    # Select final columns
    sequences = sequences.select([
        "user_id",
        "seq_idx",
        "gmap_id",
        "ts",
        "category_main",
        "lat",
        "lon",
        "rating"
    ])
    
    print(f"\n[4/4] Final sequence count: {len(sequences):,}")
    
    # Write to Parquet
    print(f"\nWriting to {output_path}...")
    sequences.write_parquet(output_path, compression="snappy")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return sequences


def derive_consecutive_pairs(sequences_df, output_path):
    """
    Create consecutive visit pairs from user sequences.
    
    Args:
        sequences_df: DataFrame of user sequences
        output_path: Path to save pairs_ga.parquet
        
    Returns:
        DataFrame of consecutive pairs
    """
    print("\n" + "=" * 80)
    print("DERIVING CONSECUTIVE PAIRS")
    print("=" * 80)
    
    print("\n[1/5] Creating shifted columns for next visit...")
    # Create columns for the next visit in sequence
    pairs = sequences_df.with_columns([
        pl.col("gmap_id").shift(-1).over("user_id").alias("dst_gmap_id"),
        pl.col("ts").shift(-1).over("user_id").alias("dst_ts"),
        pl.col("category_main").shift(-1).over("user_id").alias("dst_category_main"),
        pl.col("lat").shift(-1).over("user_id").alias("dst_lat"),
        pl.col("lon").shift(-1).over("user_id").alias("dst_lon"),
        pl.col("rating").shift(-1).over("user_id").alias("dst_rating"),
    ])
    
    # Rename source columns for clarity
    pairs = pairs.rename({
        "gmap_id": "src_gmap_id",
        "ts": "src_ts",
        "category_main": "src_category_main",
        "lat": "src_lat",
        "lon": "src_lon",
        "rating": "src_rating"
    })
    
    print("\n[2/5] Filtering out null destinations (last visit in sequence)...")
    # Remove rows where dst is null (last visit in each user's sequence)
    pairs = pairs.filter(pl.col("dst_gmap_id").is_not_null())
    print(f"  Retained {len(pairs):,} pairs")
    
    print("\n[3/5] Calculating time delta...")
    # Calculate time difference in hours
    pairs = pairs.with_columns([
        ((pl.col("dst_ts") - pl.col("src_ts")).dt.total_seconds() / 3600).alias("delta_hours")
    ])
    
    print("\n[4/5] Filtering by time window...")
    # Filter to meaningful time windows (0 < delta <= 168 hours / 7 days)
    pairs = pairs.filter(
        (pl.col("delta_hours") > MIN_DELTA_HOURS) &
        (pl.col("delta_hours") <= MAX_DELTA_HOURS)
    )
    print(f"  Retained {len(pairs):,} pairs within {MIN_DELTA_HOURS}-{MAX_DELTA_HOURS} hour window")
    
    # Select final columns
    pairs = pairs.select([
        "user_id",
        "src_gmap_id",
        "dst_gmap_id",
        "src_ts",
        "dst_ts",
        "delta_hours",
        "src_category_main",
        "dst_category_main",
        "src_lat",
        "src_lon",
        "dst_lat",
        "dst_lon",
        "src_rating",
        "dst_rating"
    ])
    
    print(f"\n[5/5] Final pair count: {len(pairs):,}")
    
    # Write to Parquet
    print(f"\nWriting to {output_path}...")
    pairs.write_parquet(output_path, compression="snappy")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return pairs


def print_sequence_statistics(sequences_df, pairs_df):
    """Print summary statistics of the derived sequences and pairs."""
    print("\n" + "=" * 80)
    print("SEQUENCE STATISTICS")
    print("=" * 80)
    
    print("\nUSER SEQUENCES:")
    print(f"  Total visits: {len(sequences_df):,}")
    print(f"  Unique users: {sequences_df['user_id'].n_unique():,}")
    print(f"  Unique businesses: {sequences_df['gmap_id'].n_unique():,}")
    
    # Sequence length distribution
    seq_lengths = sequences_df.group_by("user_id").agg(pl.len().alias("seq_length"))
    print(f"\nSEQUENCE LENGTH DISTRIBUTION:")
    print(f"  Mean: {seq_lengths['seq_length'].mean():.1f}")
    print(f"  Median: {seq_lengths['seq_length'].median():.0f}")
    print(f"  Max: {seq_lengths['seq_length'].max()}")
    
    # Users with multiple visits
    multi_visit_users = seq_lengths.filter(pl.col("seq_length") > 1)
    print(f"  Users with 2+ visits: {len(multi_visit_users):,} ({len(multi_visit_users)/len(seq_lengths)*100:.1f}%)")
    
    print("\nCONSECUTIVE PAIRS:")
    print(f"  Total pairs: {len(pairs_df):,}")
    print(f"  Unique users: {pairs_df['user_id'].n_unique():,}")
    print(f"  Unique src businesses: {pairs_df['src_gmap_id'].n_unique():,}")
    print(f"  Unique dst businesses: {pairs_df['dst_gmap_id'].n_unique():,}")
    
    print(f"\nTIME DELTA DISTRIBUTION:")
    print(f"  Mean: {pairs_df['delta_hours'].mean():.1f} hours")
    print(f"  Median: {pairs_df['delta_hours'].median():.1f} hours")
    print(f"  Min: {pairs_df['delta_hours'].min():.2f} hours")
    print(f"  Max: {pairs_df['delta_hours'].max():.1f} hours")
    
    # Category transitions
    print(f"\nTOP CATEGORY TRANSITIONS:")
    cat_transitions = pairs_df.group_by(["src_category_main", "dst_category_main"]).agg(
        pl.len().alias("count")
    ).sort("count", descending=True)
    
    for row in cat_transitions.head(10).iter_rows():
        print(f"  {row[0]:15s} → {row[1]:15s}: {row[2]:,}")
    
    print("\n" + "=" * 80)


def main(processed_dir: Optional[Path] = None):
    """
    Main sequence derivation pipeline.
    
    Args:
        processed_dir: Optional path to processed data directory.
                      If None, will check both SSD and local locations.
    """
    # Paths - check both SSD and local locations
    base_dir = Path(__file__).parent.parent.parent
    
    if processed_dir is None:
        # Try SSD first, then fall back to local
        ssd_path = Path("/Volumes/SunnySSD") / "Forkast_processed" / "ga"
        local_path = base_dir / "data" / "processed" / "ga"
        
        if ssd_path.exists() and (ssd_path / "reviews_ga.parquet").exists():
            processed_dir = ssd_path
            print(f"  Using SSD location: {processed_dir}")
        elif local_path.exists() and (local_path / "reviews_ga.parquet").exists():
            processed_dir = local_path
            print(f"  Using local location: {processed_dir}")
        else:
            # Default to local if neither exists (will create files there)
            processed_dir = local_path
            print(f"  Defaulting to local location: {processed_dir}")
            print(f"  ⚠️  reviews_ga.parquet not found - ensure Phase A1 is completed first")
    else:
        processed_dir = Path(processed_dir)
    
    input_dir = processed_dir
    output_dir = processed_dir
    
    reviews_input = input_dir / "reviews_ga.parquet"
    biz_input = input_dir / "biz_ga.parquet"
    
    sequences_output = output_dir / "user_sequences_ga.parquet"
    pairs_output = output_dir / "pairs_ga.parquet"
    
    # Check if input files exist
    if not reviews_input.exists():
        raise FileNotFoundError(
            f"Reviews file not found: {reviews_input}\n"
            f"Please run Phase A1 (data ingestion) first to generate this file."
        )
    
    if not biz_input.exists():
        raise FileNotFoundError(
            f"Business file not found: {biz_input}\n"
            f"Please run Phase A1 (data ingestion) first to generate this file."
        )
    
    # Load data
    print("Loading data...")
    try:
        reviews_df = pl.read_parquet(reviews_input)
        biz_df = pl.read_parquet(biz_input)
        print(f"  Loaded {len(reviews_df):,} reviews")
        print(f"  Loaded {len(biz_df):,} businesses")
    except Exception as e:
        raise RuntimeError(
            f"Error reading parquet files:\n"
            f"  reviews: {reviews_input}\n"
            f"  businesses: {biz_input}\n"
            f"  Error: {e}\n"
            f"\nThe files may be corrupted or incomplete. Try re-running Phase A1."
        ) from e
    
    # Derive sequences
    sequences_df = derive_user_sequences(reviews_df, biz_df, str(sequences_output))
    
    # Derive pairs
    pairs_df = derive_consecutive_pairs(sequences_df, str(pairs_output))
    
    # Print statistics
    print_sequence_statistics(sequences_df, pairs_df)
    
    print("\n✓✓✓ PHASE A2 COMPLETE ✓✓✓\n")


if __name__ == "__main__":
    main()

