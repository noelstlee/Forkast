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

# Valid timestamp range (year 2000 onwards)
MIN_TIMESTAMP = datetime(2000, 1, 1).timestamp() * 1000  # Convert to milliseconds
MAX_TIMESTAMP = datetime.now().timestamp() * 1000


def parse_price_bucket(price_str: str) -> int:
    """Convert price string ('$', '$$', etc.) to numeric bucket (1-4, 0 for null)."""
    if not price_str:
        return 0
    return len(price_str.strip())


def parse_operating_hours(hours_list):
    """
    Parse hours array into structured format.
    
    Args:
        hours_list: List of [day, time_range] pairs
        
    Returns:
        Dict with parsed hours info or None if no hours
    """
    if not hours_list or not isinstance(hours_list, list):
        return None
    
    import re
    import json
    
    parsed_hours = {}
    days_open = 0
    total_hours = 0
    has_late_night = False
    is_24hr = False
    
    for entry in hours_list:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
            
        day, time_range = entry
        if not day or not time_range:
            continue
            
        # Handle closed days
        if "closed" in time_range.lower():
            parsed_hours[day.lower()] = None
            continue
            
        # Handle 24 hour operations
        if "24" in time_range and "hour" in time_range.lower():
            parsed_hours[day.lower()] = (0, 24)
            days_open += 1
            total_hours += 24
            is_24hr = True
            continue
        
        # Parse time ranges like "8AM–6PM", "8:30AM–9:30PM"
        time_pattern = r'(\d{1,2}):?(\d{0,2})\s*(AM|PM)\s*[–\-]\s*(\d{1,2}):?(\d{0,2})\s*(AM|PM)'
        match = re.search(time_pattern, time_range, re.IGNORECASE)
        
        if match:
            open_hour, open_min, open_ampm, close_hour, close_min, close_ampm = match.groups()
            
            # Convert to 24-hour format
            open_hour = int(open_hour)
            close_hour = int(close_hour)
            open_min = int(open_min) if open_min else 0
            close_min = int(close_min) if close_min else 0
            
            if open_ampm.upper() == 'PM' and open_hour != 12:
                open_hour += 12
            elif open_ampm.upper() == 'AM' and open_hour == 12:
                open_hour = 0
                
            if close_ampm.upper() == 'PM' and close_hour != 12:
                close_hour += 12
            elif close_ampm.upper() == 'AM' and close_hour == 12:
                close_hour = 0
            
            # Convert to decimal hours
            open_decimal = open_hour + open_min / 60.0
            close_decimal = close_hour + close_min / 60.0
            
            # Handle overnight hours (close < open)
            if close_decimal < open_decimal:
                close_decimal += 24
            
            parsed_hours[day.lower()] = (open_decimal, close_decimal)
            days_open += 1
            
            # Calculate daily hours
            daily_hours = close_decimal - open_decimal
            if daily_hours > 24:
                daily_hours = 24
            total_hours += daily_hours
            
            # Check for late night (open after 11 PM)
            if close_decimal >= 23:
                has_late_night = True
    
    if not parsed_hours:
        return None
    
    avg_hours_per_day = total_hours / max(days_open, 1)
    
    # Ensure correct types for Polars Struct
    return {
        'parsed_hours': json.dumps(parsed_hours),
        'days_open_count': int(days_open),  # Will be cast to Int8 by Polars
        'avg_hours_per_day': float(round(avg_hours_per_day, 2)),  # Will be cast to Float32 by Polars
        'has_late_night': bool(has_late_night),
        'is_24hr': bool(is_24hr),
        'is_weekend_only': bool(days_open <= 2 and any(day in parsed_hours for day in ['saturday', 'sunday']))
    }


def extract_service_options(misc_dict):
    """
    Extract service options from MISC dict.
    
    Args:
        misc_dict: Dictionary containing MISC information
        
    Returns:
        Dict with service option flags
    """
    if not misc_dict or not isinstance(misc_dict, dict):
        return {
            'has_delivery': False,
            'has_takeout': False,
            'has_dinein': False,
            'accepts_reservations': False,
            'has_quick_visit': False,
            'requires_mask': False
        }
    
    service_opts = misc_dict.get("Service options", [])
    planning_opts = misc_dict.get("Planning", [])
    health_opts = misc_dict.get("Health & safety", [])
    
    # Ensure they are lists (handle None values)
    if service_opts is None:
        service_opts = []
    elif isinstance(service_opts, str):
        service_opts = [service_opts]
    elif not isinstance(service_opts, list):
        service_opts = []
    
    if planning_opts is None:
        planning_opts = []
    elif isinstance(planning_opts, str):
        planning_opts = [planning_opts]
    elif not isinstance(planning_opts, list):
        planning_opts = []
    
    if health_opts is None:
        health_opts = []
    elif isinstance(health_opts, str):
        health_opts = [health_opts]
    elif not isinstance(health_opts, list):
        health_opts = []
    
    # Ensure all items in lists are strings and not None
    service_opts = [str(opt).lower() for opt in service_opts if opt is not None]
    planning_opts = [str(opt).lower() for opt in planning_opts if opt is not None]
    health_opts = [str(opt).lower() for opt in health_opts if opt is not None]
    
    return {
        'has_delivery': any('delivery' in opt for opt in service_opts),
        'has_takeout': any('takeout' in opt or 'take' in opt for opt in service_opts),
        'has_dinein': any('dine' in opt for opt in service_opts),
        'accepts_reservations': any('reservation' in opt for opt in planning_opts),
        'has_quick_visit': any('quick' in opt for opt in planning_opts),
        'requires_mask': any('mask' in opt for opt in health_opts)
    }


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
    print("\n[5/9] Normalizing categories...")
    
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
    print("\n[6/9] Filtering to food-only businesses...")
    df = df.filter(pl.col("category_main").is_not_null())
    print(f"  Retained {len(df):,} food-related businesses")
    
    # Parse operating hours
    print("\n[7/9] Parsing operating hours...")
    
    def extract_parsed_hours(hours_list):
        """Extract just the parsed hours JSON string."""
        result = parse_operating_hours(hours_list)
        return result['parsed_hours'] if result else None
    
    def extract_days_open(hours_list):
        """Extract days_open_count."""
        result = parse_operating_hours(hours_list)
        return result['days_open_count'] if result else None
    
    def extract_avg_hours(hours_list):
        """Extract avg_hours_per_day."""
        result = parse_operating_hours(hours_list)
        return result['avg_hours_per_day'] if result else None
    
    def extract_has_late_night(hours_list):
        """Extract has_late_night."""
        result = parse_operating_hours(hours_list)
        return result['has_late_night'] if result else False
    
    def extract_is_24hr(hours_list):
        """Extract is_24hr."""
        result = parse_operating_hours(hours_list)
        return result['is_24hr'] if result else False
    
    def extract_is_weekend_only(hours_list):
        """Extract is_weekend_only."""
        result = parse_operating_hours(hours_list)
        return result['is_weekend_only'] if result else False
    
    # Extract each field separately to avoid Struct type issues
    df = df.with_columns([
        pl.col("hours").map_elements(extract_parsed_hours, return_dtype=pl.Utf8, skip_nulls=False).alias("operating_hours_parsed"),
        pl.col("hours").map_elements(extract_days_open, return_dtype=pl.Int64, skip_nulls=False).cast(pl.Int8, strict=False).alias("days_open_count"),
        pl.col("hours").map_elements(extract_avg_hours, return_dtype=pl.Float64, skip_nulls=False).cast(pl.Float32, strict=False).alias("avg_hours_per_day"),
        pl.col("hours").map_elements(extract_has_late_night, return_dtype=pl.Boolean, skip_nulls=False).alias("has_late_night"),
        pl.col("hours").map_elements(extract_is_24hr, return_dtype=pl.Boolean, skip_nulls=False).alias("is_24hr"),
        pl.col("hours").map_elements(extract_is_weekend_only, return_dtype=pl.Boolean, skip_nulls=False).alias("is_weekend_only"),
    ])
    
    # Parse service options from MISC
    print("\n[8/9] Extracting service options...")
    df = df.with_columns([
        pl.col("MISC").map_elements(extract_service_options, return_dtype=pl.Struct({
            'has_delivery': pl.Boolean,
            'has_takeout': pl.Boolean,
            'has_dinein': pl.Boolean,
            'accepts_reservations': pl.Boolean,
            'has_quick_visit': pl.Boolean,
            'requires_mask': pl.Boolean
        }), skip_nulls=False).alias("service_options_parsed")
    ])
    
    # Extract service option fields
    df = df.with_columns([
        pl.col("service_options_parsed").struct.field("has_delivery").alias("has_delivery"),
        pl.col("service_options_parsed").struct.field("has_takeout").alias("has_takeout"),
        pl.col("service_options_parsed").struct.field("has_dinein").alias("has_dinein"),
        pl.col("service_options_parsed").struct.field("accepts_reservations").alias("accepts_reservations"),
        pl.col("service_options_parsed").struct.field("has_quick_visit").alias("has_quick_visit"),
        pl.col("service_options_parsed").struct.field("requires_mask").alias("requires_mask"),
    ]).drop("service_options_parsed")
    
    # Cast and rename columns
    print("\n[9/9] Finalizing schema...")
    df = df.with_columns([
        pl.col("gmap_id").cast(pl.Utf8),
        pl.col("name").cast(pl.Utf8).str.strip_chars(),
        pl.col("latitude").cast(pl.Float32).alias("lat"),
        pl.col("longitude").cast(pl.Float32).alias("lon"),
        pl.col("avg_rating").cast(pl.Float32),
        pl.col("num_of_reviews").cast(pl.Int32).alias("num_reviews"),
    ])
    
    # Select final columns (including new metadata fields)
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
        "relative_results",
        # Operating hours fields
        "operating_hours_parsed",
        "days_open_count",
        "avg_hours_per_day",
        "has_late_night",
        "is_24hr",
        "is_weekend_only",
        # Service options fields
        "has_delivery",
        "has_takeout",
        "has_dinein",
        "accepts_reservations",
        "has_quick_visit",
        "requires_mask"
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

