"""
Phase A1: Data Ingestion & Normalization
Converts raw JSON files to cleaned Parquet format with schema normalization.
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.category_mapping import normalize_category_list, get_primary_food_category


# Georgia geographic bounds
GA_LAT_MIN, GA_LAT_MAX = 30.0, 35.0
GA_LON_MIN, GA_LON_MAX = -85.6, -80.8

# Valid timestamp range (year 2000 onwards, not in future)
MIN_TIMESTAMP = datetime(2000, 1, 1).timestamp() * 1000  # Convert to milliseconds
MAX_TIMESTAMP = datetime.now().timestamp() * 1000


def parse_price_bucket(price_str: str) -> int:
    """Convert price string ('$', '$$', etc.) to numeric bucket (1-4, 0 for null)."""
    if not price_str:
        return 0
    return len(price_str.strip())


def ingest_reviews(input_path: str, output_path: str, biz_gmap_ids: set = None):
    """
    Ingest and normalize review data from JSON to Parquet.
    
    Args:
        input_path: Path to review-Georgia.json
        output_path: Path to save reviews_ga.parquet
        biz_gmap_ids: Optional set of valid business IDs to filter reviews
    """
    print("=" * 80)
    print("PHASE A1: INGESTING REVIEWS")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Read JSON with streaming (lazy evaluation for memory efficiency)
    print("\n[1/5] Reading JSON file...")
    df = pl.read_ndjson(input_path)
    print(f"  Loaded {len(df):,} raw reviews")
    
    # Convert timestamp from unix milliseconds to datetime
    print("\n[2/5] Converting timestamps...")
    df = df.with_columns([
        pl.col("time").cast(pl.Int64).alias("time_ms"),
    ])
    
    # Filter invalid timestamps
    print("\n[3/5] Filtering invalid timestamps...")
    df = df.filter(
        (pl.col("time_ms").is_not_null()) &
        (pl.col("time_ms") >= MIN_TIMESTAMP) &
        (pl.col("time_ms") <= MAX_TIMESTAMP)
    )
    print(f"  Retained {len(df):,} reviews with valid timestamps")
    
    # Convert to datetime
    df = df.with_columns([
        (pl.col("time_ms") / 1000).cast(pl.Int64).alias("ts_seconds")
    ]).with_columns([
        pl.from_epoch("ts_seconds", time_unit="s").alias("ts")
    ])
    
    # Create derived boolean columns
    print("\n[4/5] Creating derived columns...")
    df = df.with_columns([
        pl.col("user_id").cast(pl.Utf8),
        pl.col("gmap_id").cast(pl.Utf8),
        pl.col("rating").cast(pl.Int8),
        pl.col("text").cast(pl.Utf8).str.strip_chars(),
        pl.col("pics").is_not_null().alias("has_pics"),
        pl.col("resp").is_not_null().alias("has_resp"),
    ])
    
    # Filter out reviews with null user_id or rating
    print("\n  Filtering out reviews with missing user_id or rating...")
    before_count = len(df)
    df = df.filter(
        pl.col("user_id").is_not_null() &
        pl.col("rating").is_not_null()
    )
    removed = before_count - len(df)
    print(f"  Removed {removed:,} reviews with null user_id or rating")
    print(f"  Retained {len(df):,} reviews")
    
    # Select final columns
    df = df.select([
        "user_id",
        "gmap_id", 
        "ts",
        "rating",
        "text",
        "has_pics",
        "has_resp"
    ])
    
    # Filter to valid businesses if provided
    if biz_gmap_ids:
        print(f"\n  Filtering to {len(biz_gmap_ids):,} valid businesses...")
        df = df.filter(pl.col("gmap_id").is_in(list(biz_gmap_ids)))
        print(f"  Retained {len(df):,} reviews")
    
    # Deduplicate
    print("\n[5/5] Deduplicating...")
    df = df.unique(subset=["user_id", "gmap_id", "ts", "rating"])
    print(f"  Final count: {len(df):,} unique reviews")
    
    # Write to Parquet
    print(f"\nWriting to {output_path}...")
    df.write_parquet(output_path, compression="snappy")
    
    print("\n✓ Reviews ingestion complete!")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    return df


def ingest_metadata(input_path: str, output_path: str):
    """
    Ingest and normalize business metadata from JSON to Parquet.
    
    Args:
        input_path: Path to meta-Georgia.json
        output_path: Path to save biz_ga.parquet
        
    Returns:
        DataFrame of businesses and set of valid gmap_ids
    """
    print("\n" + "=" * 80)
    print("PHASE A1: INGESTING METADATA")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Read JSON
    print("\n[1/6] Reading JSON file...")
    df = pl.read_ndjson(input_path)
    print(f"  Loaded {len(df):,} raw businesses")
    
    # Filter to Georgia bounds
    print("\n[2/6] Filtering to Georgia geographic bounds...")
    df = df.filter(
        (pl.col("latitude").is_not_null()) &
        (pl.col("longitude").is_not_null()) &
        (pl.col("latitude") >= GA_LAT_MIN) &
        (pl.col("latitude") <= GA_LAT_MAX) &
        (pl.col("longitude") >= GA_LON_MIN) &
        (pl.col("longitude") <= GA_LON_MAX)
    )
    print(f"  Retained {len(df):,} businesses in Georgia")
    
    # Parse price to numeric bucket
    print("\n[3/6] Parsing price buckets...")
    df = df.with_columns([
        pl.col("price").map_elements(parse_price_bucket, return_dtype=pl.Int8).alias("price_bucket")
    ])
    
    # Check if business is closed
    print("\n[4/6] Detecting closed businesses...")
    df = df.with_columns([
        pl.when(pl.col("state").is_not_null())
        .then(pl.col("state").str.to_lowercase().str.contains("closed"))
        .otherwise(False)
        .alias("is_closed")
    ])
    
    # Normalize categories using Python UDF
    print("\n[5/7] Normalizing categories...")
    
    def extract_category_main(categories):
        """Extract main food category from list, prioritizing food categories."""
        if categories is None or (isinstance(categories, list) and len(categories) == 0):
            return None  # Changed from "other"
        
        # Normalize all categories
        _, all_cats = normalize_category_list(categories)
        
        # Get primary food category (None if no food categories)
        return get_primary_food_category(all_cats)
    
    def extract_category_all(categories):
        """Extract all normalized categories from list."""
        if categories is None or (isinstance(categories, list) and len(categories) == 0):
            return []
        _, all_cats = normalize_category_list(categories)
        return all_cats
    
    df = df.with_columns([
        pl.col("category").map_elements(extract_category_main, return_dtype=pl.Utf8, skip_nulls=False).alias("category_main"),
        pl.col("category").map_elements(extract_category_all, return_dtype=pl.List(pl.Utf8), skip_nulls=False).alias("category_all"),
    ])
    
    # Filter to food-only businesses
    print("\n[6/7] Filtering to food-only businesses...")
    df = df.filter(pl.col("category_main").is_not_null())
    print(f"  Retained {len(df):,} food-related businesses")
    
    # Cast and rename columns
    print("\n[7/7] Finalizing schema...")
    df = df.with_columns([
        pl.col("gmap_id").cast(pl.Utf8),
        pl.col("name").cast(pl.Utf8).str.strip_chars(),
        pl.col("latitude").cast(pl.Float32).alias("lat"),
        pl.col("longitude").cast(pl.Float32).alias("lon"),
        pl.col("avg_rating").cast(pl.Float32),
        pl.col("num_of_reviews").cast(pl.Int32).alias("num_reviews"),
    ])
    
    # Select final columns
    df = df.select([
        "gmap_id",
        "name",
        "lat",
        "lon",
        "category_main",
        "category_all",
        "avg_rating",
        "num_reviews",
        "price_bucket",
        "is_closed",
        "relative_results"
    ])
    
    # Remove duplicates
    df = df.unique(subset=["gmap_id"])
    print(f"  Final count: {len(df):,} unique businesses")
    
    # Write to Parquet
    print(f"\nWriting to {output_path}...")
    df.write_parquet(output_path, compression="snappy")
    
    print("\n✓ Metadata ingestion complete!")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Return valid gmap_ids for filtering reviews
    valid_ids = set(df["gmap_id"].to_list())
    
    return df, valid_ids


def print_data_statistics(reviews_df, biz_df):
    """Print summary statistics of the ingested data."""
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)
    
    print("\nBUSINESSES:")
    print(f"  Total businesses: {len(biz_df):,}")
    print(f"  Closed businesses: {biz_df['is_closed'].sum():,}")
    print(f"  Average rating: {biz_df['avg_rating'].mean():.2f}")
    print(f"  Total reviews (metadata): {biz_df['num_reviews'].sum():,}")
    
    print("\nCATEGORY DISTRIBUTION:")
    cat_dist = biz_df.group_by("category_main").agg(pl.len().alias("count")).sort("count", descending=True)
    for row in cat_dist.head(10).iter_rows():
        print(f"  {row[0]:20s}: {row[1]:,}")
    
    print("\nREVIEWS:")
    print(f"  Total reviews: {len(reviews_df):,}")
    print(f"  Unique users: {reviews_df['user_id'].n_unique():,}")
    print(f"  Reviews with pics: {reviews_df['has_pics'].sum():,}")
    print(f"  Reviews with responses: {reviews_df['has_resp'].sum():,}")
    print(f"  Average rating: {reviews_df['rating'].mean():.2f}")
    
    # Temporal distribution
    print("\nTEMPORAL DISTRIBUTION:")
    print(f"  Earliest review: {reviews_df['ts'].min()}")
    print(f"  Latest review: {reviews_df['ts'].max()}")
    
    # Reviews per user
    reviews_per_user = reviews_df.group_by("user_id").agg(pl.len().alias("count"))
    print(f"\nREVIEWS PER USER:")
    print(f"  Mean: {reviews_per_user['count'].mean():.1f}")
    print(f"  Median: {reviews_per_user['count'].median():.0f}")
    print(f"  Max: {reviews_per_user['count'].max()}")
    
    print("\n" + "=" * 80)


def main():
    """Main ingestion pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed" / "ga"
    
    reviews_input = raw_dir / "review-Georgia.json"
    metadata_input = raw_dir / "meta-Georgia.json"
    
    reviews_output = output_dir / "reviews_ga.parquet"
    metadata_output = output_dir / "biz_ga.parquet"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ingest metadata first to get valid business IDs
    biz_df, valid_gmap_ids = ingest_metadata(str(metadata_input), str(metadata_output))
    
    # Ingest reviews, filtering to valid businesses
    reviews_df = ingest_reviews(str(reviews_input), str(reviews_output), valid_gmap_ids)
    
    # Print statistics
    print_data_statistics(reviews_df, biz_df)
    
    print("\n✓✓✓ PHASE A1 COMPLETE ✓✓✓\n")


if __name__ == "__main__":
    main()

