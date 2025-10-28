"""
Phase A2.5: Data Quality Filtering
Improves data quality by:
1. Re-categorizing 'restaurant' businesses using name-based heuristics
2. Filtering sequences to users with 5+ visits
3. Filtering pairs to remove very short time deltas (<=0.2 hours)
4. Regenerating pairs from filtered sequences
"""

import polars as pl
from pathlib import Path
import re


def infer_category_from_name(name: str) -> str:
    """
    Infer food category from business name using keyword matching.
    
    Args:
        name: Business name
        
    Returns:
        Inferred category or 'other' if no match
    """
    name_lower = name.lower()
    
    # Category keywords (order matters - more specific first)
    category_patterns = {
        'pizza': ['pizza', 'pizzeria', 'pie', 'pizza hut', 'domino', 'papa john', 'little caesar'],
        'burger': ['burger', 'burgers', 'five guys', 'shake shack', 'smashburger', 'whataburger', 'in-n-out'],
        'chinese': ['chinese', 'china', 'wok', 'panda express', 'dim sum', 'szechuan', 'hunan'],
        'mexican': ['mexican', 'taco', 'burrito', 'chipotle', 'qdoba', 'taqueria', 'cantina', 'tortilla'],
        'italian': ['italian', 'pasta', 'trattoria', 'ristorante', 'olive garden', 'spaghetti'],
        'sushi': ['sushi', 'ramen', 'hibachi', 'teriyaki', 'japanese'],
        'bbq': ['bbq', 'barbecue', 'barbeque', 'smokehouse', 'grill house', 'pit'],
        'steakhouse': ['steakhouse', 'steak house', 'chophouse', 'chop house', 'longhorn', 'outback'],
        'seafood': ['seafood', 'fish', 'lobster', 'crab', 'oyster', 'shrimp', 'red lobster'],
        'breakfast': ['breakfast', 'pancake', 'waffle', 'ihop', 'denny', 'cracker barrel'],
        'cafe': ['cafe', 'coffee', 'espresso', 'starbucks', 'dunkin'],
        'bakery': ['bakery', 'bakehouse', 'bread', 'pastry', 'donut', 'doughnut', 'krispy kreme'],
        'ice_cream': ['ice cream', 'gelato', 'frozen yogurt', 'froyo', 'baskin', 'cold stone'],
        'dessert': ['dessert', 'sweet', 'cupcake', 'cookie', 'cake'],
        'bar': ['bar', 'pub', 'tavern', 'lounge', 'sports bar', 'grill'],
        'fast_food': ['fast food', 'drive thru', 'drive-thru', 'quick service'],
        'american': ['diner', 'grill', 'american', 'wings', 'chicken'],
        'asian': ['asian', 'thai', 'vietnamese', 'pho', 'korean', 'noodle'],
        'indian': ['indian', 'curry', 'tandoor'],
        'mediterranean': ['mediterranean', 'greek', 'gyro', 'kebab', 'falafel'],
    }
    
    # Check each category
    for category, keywords in category_patterns.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    
    return 'other'


def recategorize_restaurants(biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Re-categorize businesses with category_main='restaurant' using name-based inference.
    
    Args:
        biz_df: Business DataFrame
        
    Returns:
        Updated DataFrame with re-categorized businesses
    """
    print("\n" + "=" * 80)
    print("STEP 1: RE-CATEGORIZING 'RESTAURANT' BUSINESSES")
    print("=" * 80)
    
    # Count original restaurants
    restaurant_count = biz_df.filter(pl.col('category_main') == 'restaurant').shape[0]
    print(f"\nOriginal 'restaurant' businesses: {restaurant_count:,}")
    
    # Apply name-based inference to restaurants
    print("\n[1/3] Inferring categories from business names...")
    
    # Create a new column with inferred categories for restaurants
    biz_df = biz_df.with_columns([
        pl.when(pl.col('category_main') == 'restaurant')
          .then(pl.col('name').map_elements(infer_category_from_name, return_dtype=pl.Utf8))
          .otherwise(pl.col('category_main'))
          .alias('category_main')
    ])
    
    # Count results
    print("\n[2/3] Re-categorization results:")
    recategorized = biz_df.filter(
        (pl.col('category_main') != 'restaurant') & 
        (pl.col('category_main') != 'other')
    )
    
    still_other = biz_df.filter(pl.col('category_main') == 'other').shape[0]
    
    print(f"  Successfully re-categorized: {restaurant_count - still_other:,}")
    print(f"  Moved to 'other': {still_other:,}")
    
    # Show new category distribution
    print("\n[3/3] New category distribution (top 15):")
    cat_dist = biz_df.group_by('category_main').agg(pl.len().alias('count')).sort('count', descending=True)
    for i, row in enumerate(cat_dist.head(15).iter_rows(), 1):
        pct = row[1] / len(biz_df) * 100
        print(f"  {i:2d}. {row[0]:20s}: {row[1]:6,} ({pct:5.1f}%)")
    
    return biz_df


def filter_sequences_by_length(sequences_df: pl.DataFrame, min_visits: int = 5) -> pl.DataFrame:
    """
    Filter sequences to keep only users with minimum number of visits.
    
    Args:
        sequences_df: User sequences DataFrame
        min_visits: Minimum number of visits required
        
    Returns:
        Filtered sequences DataFrame
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: FILTERING SEQUENCES (MIN {min_visits} VISITS)")
    print("=" * 80)
    
    print(f"\nOriginal sequences: {len(sequences_df):,}")
    print(f"Original users: {sequences_df['user_id'].n_unique():,}")
    
    # Count visits per user
    print("\n[1/2] Counting visits per user...")
    user_counts = sequences_df.group_by('user_id').agg(pl.len().alias('visit_count'))
    
    # Filter to users with min_visits or more
    valid_users = user_counts.filter(pl.col('visit_count') >= min_visits)
    print(f"  Users with {min_visits}+ visits: {len(valid_users):,}")
    
    # Filter sequences
    print(f"\n[2/2] Filtering sequences...")
    filtered_sequences = sequences_df.filter(
        pl.col('user_id').is_in(valid_users['user_id'])
    )
    
    print(f"\nFiltered sequences: {len(filtered_sequences):,}")
    print(f"Filtered users: {filtered_sequences['user_id'].n_unique():,}")
    print(f"Retention: {len(filtered_sequences)/len(sequences_df)*100:.1f}% of sequences")
    
    return filtered_sequences


def filter_pairs_by_time_delta(pairs_df: pl.DataFrame, min_hours: float = 0.2) -> pl.DataFrame:
    """
    Filter pairs to remove very short time deltas (likely batch reviews).
    
    Args:
        pairs_df: Pairs DataFrame
        min_hours: Minimum time delta in hours
        
    Returns:
        Filtered pairs DataFrame
    """
    print("\n" + "=" * 80)
    print(f"STEP 3: FILTERING PAIRS (MIN {min_hours} HOURS)")
    print("=" * 80)
    
    print(f"\nOriginal pairs: {len(pairs_df):,}")
    
    # Show time delta distribution before filtering
    print("\n[1/2] Time delta distribution (before):")
    print(f"  <= 0.2 hours: {(pairs_df['delta_hours'] <= 0.2).sum():,}")
    print(f"  0.2-1 hours: {((pairs_df['delta_hours'] > 0.2) & (pairs_df['delta_hours'] <= 1)).sum():,}")
    print(f"  1-6 hours: {((pairs_df['delta_hours'] > 1) & (pairs_df['delta_hours'] <= 6)).sum():,}")
    print(f"  6-24 hours: {((pairs_df['delta_hours'] > 6) & (pairs_df['delta_hours'] <= 24)).sum():,}")
    print(f"  1-7 days: {(pairs_df['delta_hours'] > 24).sum():,}")
    
    # Filter
    print(f"\n[2/2] Filtering pairs with delta_hours > {min_hours}...")
    filtered_pairs = pairs_df.filter(pl.col('delta_hours') > min_hours)
    
    print(f"\nFiltered pairs: {len(filtered_pairs):,}")
    print(f"Removed: {len(pairs_df) - len(filtered_pairs):,} ({(len(pairs_df) - len(filtered_pairs))/len(pairs_df)*100:.1f}%)")
    print(f"Retention: {len(filtered_pairs)/len(pairs_df)*100:.1f}%")
    
    return filtered_pairs


def regenerate_pairs_from_sequences(sequences_df: pl.DataFrame, output_path: str, 
                                    min_delta_hours: float = 0.2, max_delta_hours: float = 168):
    """
    Regenerate consecutive pairs from filtered sequences.
    
    Args:
        sequences_df: Filtered user sequences DataFrame
        output_path: Path to save pairs
        min_delta_hours: Minimum time delta
        max_delta_hours: Maximum time delta (7 days)
        
    Returns:
        Pairs DataFrame
    """
    print("\n" + "=" * 80)
    print("STEP 4: REGENERATING PAIRS FROM FILTERED SEQUENCES")
    print("=" * 80)
    
    print("\n[1/4] Creating shifted columns for next visit...")
    pairs = sequences_df.with_columns([
        pl.col("gmap_id").shift(-1).over("user_id").alias("dst_gmap_id"),
        pl.col("ts").shift(-1).over("user_id").alias("dst_ts"),
        pl.col("category_main").shift(-1).over("user_id").alias("dst_category_main"),
        pl.col("lat").shift(-1).over("user_id").alias("dst_lat"),
        pl.col("lon").shift(-1).over("user_id").alias("dst_lon"),
        pl.col("rating").shift(-1).over("user_id").alias("dst_rating"),
    ])
    
    # Rename source columns
    pairs = pairs.rename({
        "gmap_id": "src_gmap_id",
        "ts": "src_ts",
        "category_main": "src_category_main",
        "lat": "src_lat",
        "lon": "src_lon",
        "rating": "src_rating"
    })
    
    print("\n[2/4] Filtering out null destinations...")
    pairs = pairs.filter(pl.col("dst_gmap_id").is_not_null())
    print(f"  Retained {len(pairs):,} pairs")
    
    print("\n[3/4] Calculating time delta and filtering...")
    pairs = pairs.with_columns([
        ((pl.col("dst_ts") - pl.col("src_ts")).dt.total_seconds() / 3600).alias("delta_hours")
    ])
    
    # Filter by time window
    pairs = pairs.filter(
        (pl.col("delta_hours") > min_delta_hours) &
        (pl.col("delta_hours") <= max_delta_hours)
    )
    print(f"  Retained {len(pairs):,} pairs within {min_delta_hours}-{max_delta_hours} hour window")
    
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
    
    print(f"\n[4/4] Final pair count: {len(pairs):,}")
    
    # Write to Parquet
    print(f"\nWriting to {output_path}...")
    pairs.write_parquet(output_path, compression="snappy")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return pairs


def print_final_statistics(biz_df: pl.DataFrame, sequences_df: pl.DataFrame, pairs_df: pl.DataFrame):
    """Print final statistics after filtering."""
    print("\n" + "=" * 80)
    print("FINAL STATISTICS AFTER PHASE A2.5")
    print("=" * 80)
    
    print("\n📊 BUSINESSES:")
    print(f"  Total: {len(biz_df):,}")
    print(f"  Categories: {biz_df['category_main'].n_unique()}")
    print(f"  'other' category: {biz_df.filter(pl.col('category_main') == 'other').shape[0]:,}")
    
    print("\n👥 USER SEQUENCES:")
    print(f"  Total visits: {len(sequences_df):,}")
    print(f"  Unique users: {sequences_df['user_id'].n_unique():,}")
    print(f"  Unique businesses: {sequences_df['gmap_id'].n_unique():,}")
    
    seq_lengths = sequences_df.group_by('user_id').agg(pl.len().alias('seq_length'))
    print(f"\n  Sequence length distribution:")
    print(f"    Mean: {seq_lengths['seq_length'].mean():.1f}")
    print(f"    Median: {seq_lengths['seq_length'].median():.0f}")
    print(f"    Min: {seq_lengths['seq_length'].min()}")
    print(f"    Max: {seq_lengths['seq_length'].max()}")
    
    print("\n🔗 CONSECUTIVE PAIRS:")
    print(f"  Total pairs: {len(pairs_df):,}")
    print(f"  Unique users: {pairs_df['user_id'].n_unique():,}")
    print(f"  Unique src businesses: {pairs_df['src_gmap_id'].n_unique():,}")
    print(f"  Unique dst businesses: {pairs_df['dst_gmap_id'].n_unique():,}")
    
    print(f"\n  Time delta distribution:")
    print(f"    Mean: {pairs_df['delta_hours'].mean():.1f} hours")
    print(f"    Median: {pairs_df['delta_hours'].median():.1f} hours")
    print(f"    Min: {pairs_df['delta_hours'].min():.2f} hours")
    print(f"    Max: {pairs_df['delta_hours'].max():.1f} hours")
    
    print("\n  Top 10 category transitions:")
    cat_trans = pairs_df.group_by(['src_category_main', 'dst_category_main']).agg(
        pl.len().alias('count')
    ).sort('count', descending=True)
    
    for i, row in enumerate(cat_trans.head(10).iter_rows(), 1):
        print(f"    {i:2d}. {row[0]:15s} → {row[1]:15s}: {row[2]:,}")
    
    print("\n" + "=" * 80)


def main():
    """Main Phase A2.5 pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "data" / "processed" / "ga"
    
    biz_input = processed_dir / "biz_ga.parquet"
    sequences_input = processed_dir / "user_sequences_ga.parquet"
    pairs_input = processed_dir / "pairs_ga.parquet"
    
    biz_output = processed_dir / "biz_ga.parquet"
    sequences_output = processed_dir / "user_sequences_filtered_ga.parquet"
    pairs_output = processed_dir / "pairs_filtered_ga.parquet"
    
    print("=" * 80)
    print("PHASE A2.5: DATA QUALITY FILTERING")
    print("=" * 80)
    print(f"\nInputs:")
    print(f"  - {biz_input}")
    print(f"  - {sequences_input}")
    print(f"  - {pairs_input}")
    print(f"\nOutputs:")
    print(f"  - {biz_output} (updated)")
    print(f"  - {sequences_output}")
    print(f"  - {pairs_output}")
    
    # Load data
    print("\n[Loading data...]")
    biz_df = pl.read_parquet(biz_input)
    sequences_df = pl.read_parquet(sequences_input)
    pairs_df = pl.read_parquet(pairs_input)
    
    # Step 1: Re-categorize restaurants
    biz_df = recategorize_restaurants(biz_df)
    
    # Update business data (overwrite original)
    print(f"\nSaving updated business data to {biz_output}...")
    biz_df.write_parquet(biz_output, compression="snappy")
    print(f"  Output size: {Path(biz_output).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step 2: Filter sequences by length (5+ visits)
    sequences_filtered = filter_sequences_by_length(sequences_df, min_visits=5)
    
    # Save filtered sequences
    print(f"\nSaving filtered sequences to {sequences_output}...")
    sequences_filtered.write_parquet(sequences_output, compression="snappy")
    print(f"  Output size: {Path(sequences_output).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step 3: Filter pairs by time delta (just for comparison)
    pairs_time_filtered = filter_pairs_by_time_delta(pairs_df, min_hours=0.2)
    
    # Step 4: Regenerate pairs from filtered sequences (this also applies time filter)
    pairs_regenerated = regenerate_pairs_from_sequences(
        sequences_filtered, 
        str(pairs_output),
        min_delta_hours=0.2,
        max_delta_hours=168
    )
    
    # Print final statistics
    print_final_statistics(biz_df, sequences_filtered, pairs_regenerated)
    
    print("\n✓✓✓ PHASE A2.5 COMPLETE ✓✓✓\n")


def filter_sequences_only():
    """Filter only sequences (for LSTM pipeline)."""
    # Input/output paths
    biz_input = Path("data/processed/ga/biz_ga.parquet")
    sequences_input = Path("data/processed/ga/user_sequences_ga.parquet")
    sequences_output = Path("data/processed/ga/user_sequences_filtered_ga.parquet")
    
    # Load data
    print("\n[Loading data...]")
    biz_df = pl.read_parquet(biz_input)
    sequences_df = pl.read_parquet(sequences_input)
    
    print(f"  Businesses: {len(biz_df):,}")
    print(f"  Sequences: {len(sequences_df):,} visits from {sequences_df['user_id'].n_unique():,} users")
    
    # Step 1: Re-categorize 'restaurant' businesses
    biz_df = recategorize_restaurants(biz_df)
    
    # Update business data (overwrite original)
    print(f"\nSaving updated business data to {biz_input}...")
    biz_df.write_parquet(biz_input, compression="snappy")
    print(f"  Output size: {Path(biz_input).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Step 2: Filter sequences by minimum visit count
    sequences_filtered = filter_sequences_by_length(sequences_df, min_visits=5)
    
    # Save filtered sequences
    print(f"\nSaving filtered sequences to {sequences_output}...")
    sequences_filtered.write_parquet(sequences_output, compression="snappy")
    print(f"  Output size: {Path(sequences_output).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print final statistics (sequences only)
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 SEQUENCES:")
    print(f"  Total visits: {len(sequences_filtered):,}")
    print(f"  Unique users: {sequences_filtered['user_id'].n_unique():,}")
    print(f"  Unique businesses: {sequences_filtered['gmap_id'].n_unique():,}")
    print(f"  Mean visits/user: {len(sequences_filtered) / sequences_filtered['user_id'].n_unique():.1f}")


if __name__ == "__main__":
    main()

