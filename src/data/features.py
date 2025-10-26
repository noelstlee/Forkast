"""
Phase A3: Feature Engineering
Creates features for XGBoost training from consecutive visit pairs.
Includes spatial, temporal, quality, price, category, and relationship features.
Also implements hybrid negative sampling.
"""

import polars as pl
import numpy as np
from pathlib import Path
from haversine import haversine, Unit
from typing import List, Tuple
import random


def calculate_haversine_distance(src_lat: float, src_lon: float, 
                                 dst_lat: float, dst_lon: float) -> float:
    """
    Calculate haversine distance between two points in kilometers.
    
    Args:
        src_lat, src_lon: Source coordinates
        dst_lat, dst_lon: Destination coordinates
        
    Returns:
        Distance in kilometers
    """
    return haversine((src_lat, src_lon), (dst_lat, dst_lon), unit=Unit.KILOMETERS)


def calculate_bearing(src_lat: float, src_lon: float,
                     dst_lat: float, dst_lon: float) -> str:
    """
    Calculate bearing/direction from source to destination.
    
    Returns:
        Direction string: N, NE, E, SE, S, SW, W, NW
    """
    # Convert to radians
    lat1, lon1 = np.radians(src_lat), np.radians(src_lon)
    lat2, lon2 = np.radians(dst_lat), np.radians(dst_lon)
    
    # Calculate bearing
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    
    # Normalize to 0-360
    bearing = (bearing + 360) % 360
    
    # Convert to 8-direction compass
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((bearing + 22.5) / 45) % 8
    return directions[idx]


def add_spatial_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add spatial features to pairs dataframe.
    
    Features:
    - distance_km: Haversine distance
    - distance_bucket: Categorical distance (0-1km, 1-5km, etc.)
    - same_neighborhood: Within 2km
    - direction: Compass direction (N, NE, E, etc.)
    
    Args:
        pairs_df: Pairs DataFrame with src/dst coordinates
        
    Returns:
        DataFrame with spatial features added
    """
    print("\n[1/7] Adding spatial features...")
    
    # Calculate haversine distance
    print("  - Calculating haversine distances...")
    distances = []
    for row in pairs_df.select(['src_lat', 'src_lon', 'dst_lat', 'dst_lon']).iter_rows():
        dist = calculate_haversine_distance(row[0], row[1], row[2], row[3])
        distances.append(dist)
    
    pairs_df = pairs_df.with_columns([
        pl.Series("distance_km", distances, dtype=pl.Float32)
    ])
    
    # Distance buckets
    print("  - Creating distance buckets...")
    pairs_df = pairs_df.with_columns([
        pl.when(pl.col("distance_km") < 1).then(pl.lit("0-1km"))
          .when(pl.col("distance_km") < 5).then(pl.lit("1-5km"))
          .when(pl.col("distance_km") < 10).then(pl.lit("5-10km"))
          .when(pl.col("distance_km") < 20).then(pl.lit("10-20km"))
          .otherwise(pl.lit("20+km"))
          .alias("distance_bucket")
    ])
    
    # Same neighborhood (within 2km)
    pairs_df = pairs_df.with_columns([
        (pl.col("distance_km") <= 2.0).alias("same_neighborhood")
    ])
    
    # Calculate bearing/direction
    print("  - Calculating directions...")
    directions = []
    for row in pairs_df.select(['src_lat', 'src_lon', 'dst_lat', 'dst_lon']).iter_rows():
        direction = calculate_bearing(row[0], row[1], row[2], row[3])
        directions.append(direction)
    
    pairs_df = pairs_df.with_columns([
        pl.Series("direction", directions, dtype=pl.Utf8)
    ])
    
    print(f"  ✓ Added spatial features")
    return pairs_df


def add_temporal_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add temporal features to pairs dataframe.
    
    Features:
    - delta_hours_bucket: Categorical time delta
    - src_hour, dst_hour: Hour of day (0-23)
    - src_day_of_week, dst_day_of_week: Day of week (0-6)
    - is_weekend: Boolean
    - is_meal_time: Boolean (breakfast/lunch/dinner hours)
    - meal_type: breakfast/lunch/dinner/other
    
    Args:
        pairs_df: Pairs DataFrame with timestamps
        
    Returns:
        DataFrame with temporal features added
    """
    print("\n[2/7] Adding temporal features...")
    
    # Time delta buckets
    print("  - Creating time delta buckets...")
    pairs_df = pairs_df.with_columns([
        pl.when(pl.col("delta_hours") < 3).then(pl.lit("0-3h"))
          .when(pl.col("delta_hours") < 6).then(pl.lit("3-6h"))
          .when(pl.col("delta_hours") < 12).then(pl.lit("6-12h"))
          .when(pl.col("delta_hours") < 24).then(pl.lit("12-24h"))
          .when(pl.col("delta_hours") < 72).then(pl.lit("1-3d"))
          .otherwise(pl.lit("3-7d"))
          .alias("delta_hours_bucket")
    ])
    
    # Extract hour and day of week
    print("  - Extracting hour and day of week...")
    pairs_df = pairs_df.with_columns([
        pl.col("src_ts").dt.hour().alias("src_hour"),
        pl.col("dst_ts").dt.hour().alias("dst_hour"),
        pl.col("src_ts").dt.weekday().alias("src_day_of_week"),
        pl.col("dst_ts").dt.weekday().alias("dst_day_of_week"),
    ])
    
    # Weekend indicator
    pairs_df = pairs_df.with_columns([
        (pl.col("src_day_of_week") >= 5).alias("src_is_weekend"),
        (pl.col("dst_day_of_week") >= 5).alias("dst_is_weekend"),
    ])
    
    # Meal time indicators
    print("  - Identifying meal times...")
    pairs_df = pairs_df.with_columns([
        # Breakfast: 6-10am
        ((pl.col("src_hour") >= 6) & (pl.col("src_hour") < 10)).alias("src_is_breakfast"),
        ((pl.col("dst_hour") >= 6) & (pl.col("dst_hour") < 10)).alias("dst_is_breakfast"),
        # Lunch: 11am-2pm
        ((pl.col("src_hour") >= 11) & (pl.col("src_hour") < 14)).alias("src_is_lunch"),
        ((pl.col("dst_hour") >= 11) & (pl.col("dst_hour") < 14)).alias("dst_is_lunch"),
        # Dinner: 5-9pm
        ((pl.col("src_hour") >= 17) & (pl.col("src_hour") < 21)).alias("src_is_dinner"),
        ((pl.col("dst_hour") >= 17) & (pl.col("dst_hour") < 21)).alias("dst_is_dinner"),
    ])
    
    # Meal type (categorical)
    pairs_df = pairs_df.with_columns([
        pl.when(pl.col("dst_is_breakfast")).then(pl.lit("breakfast"))
          .when(pl.col("dst_is_lunch")).then(pl.lit("lunch"))
          .when(pl.col("dst_is_dinner")).then(pl.lit("dinner"))
          .otherwise(pl.lit("other"))
          .alias("dst_meal_type")
    ])
    
    print(f"  ✓ Added temporal features")
    return pairs_df


def add_quality_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add quality features based on ratings and reviews.
    
    Features:
    - rating_diff: dst_rating - src_rating
    - is_rating_upgrade: dst_rating > src_rating
    - is_highly_rated_src/dst: rating >= 4.5
    - Both ratings are already in the dataframe
    
    Args:
        pairs_df: Pairs DataFrame with ratings
        
    Returns:
        DataFrame with quality features added
    """
    print("\n[3/7] Adding quality features...")
    
    # Rating difference
    pairs_df = pairs_df.with_columns([
        (pl.col("dst_rating") - pl.col("src_rating")).alias("rating_diff")
    ])
    
    # Rating upgrade/downgrade
    pairs_df = pairs_df.with_columns([
        (pl.col("dst_rating") > pl.col("src_rating")).alias("is_rating_upgrade"),
        (pl.col("dst_rating") < pl.col("src_rating")).alias("is_rating_downgrade"),
    ])
    
    # Highly rated indicators
    pairs_df = pairs_df.with_columns([
        (pl.col("src_rating") >= 4.5).alias("src_is_highly_rated"),
        (pl.col("dst_rating") >= 4.5).alias("dst_is_highly_rated"),
    ])
    
    print(f"  ✓ Added quality features")
    return pairs_df


def add_price_features(pairs_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add price features by joining with business data.
    
    Features:
    - src_price, dst_price: Price buckets (1-4)
    - price_diff: dst_price - src_price
    - is_price_upgrade: dst_price > src_price
    - same_price_level: dst_price == src_price
    
    Args:
        pairs_df: Pairs DataFrame
        biz_df: Business DataFrame with price_bucket
        
    Returns:
        DataFrame with price features added
    """
    print("\n[4/7] Adding price features...")
    
    # Join to get price buckets
    print("  - Joining with business data for prices...")
    pairs_df = pairs_df.join(
        biz_df.select(["gmap_id", "price_bucket"]).rename({"price_bucket": "src_price"}),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    ).join(
        biz_df.select(["gmap_id", "price_bucket"]).rename({"price_bucket": "dst_price"}),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Fill nulls with 0 (unknown price)
    pairs_df = pairs_df.with_columns([
        pl.col("src_price").fill_null(0),
        pl.col("dst_price").fill_null(0),
    ])
    
    # Price difference
    pairs_df = pairs_df.with_columns([
        (pl.col("dst_price") - pl.col("src_price")).alias("price_diff")
    ])
    
    # Price upgrade/downgrade
    pairs_df = pairs_df.with_columns([
        (pl.col("dst_price") > pl.col("src_price")).alias("is_price_upgrade"),
        (pl.col("dst_price") < pl.col("src_price")).alias("is_price_downgrade"),
        (pl.col("dst_price") == pl.col("src_price")).alias("same_price_level"),
    ])
    
    print(f"  ✓ Added price features")
    return pairs_df


def add_category_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add category features.
    
    Features:
    - same_category: Boolean
    - Category columns are already in the dataframe (will be one-hot encoded later)
    
    Args:
        pairs_df: Pairs DataFrame with categories
        
    Returns:
        DataFrame with category features added
    """
    print("\n[5/7] Adding category features...")
    
    # Same category indicator
    pairs_df = pairs_df.with_columns([
        (pl.col("src_category_main") == pl.col("dst_category_main")).alias("same_category")
    ])
    
    print(f"  ✓ Added category features")
    return pairs_df


def add_relationship_features(pairs_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add business relationship features using relative_results.
    
    Features:
    - is_in_relative_results: Boolean
    - relative_results_rank: Position in list (1-5) or null
    
    Args:
        pairs_df: Pairs DataFrame
        biz_df: Business DataFrame with relative_results
        
    Returns:
        DataFrame with relationship features added
    """
    print("\n[6/7] Adding relationship features...")
    
    # Join to get relative_results
    print("  - Joining with business data for relative_results...")
    pairs_df = pairs_df.join(
        biz_df.select(["gmap_id", "relative_results"]),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Check if dst is in src's relative_results
    print("  - Checking relative_results membership...")
    
    # Create helper columns
    pairs_df = pairs_df.with_columns([
        pl.col("relative_results").fill_null([]),
        pl.col("dst_gmap_id").alias("dst_id_check")
    ])
    
    # Check membership (this is slow but necessary)
    is_in_relative = []
    relative_rank = []
    
    for row in pairs_df.select(['relative_results', 'dst_gmap_id']).iter_rows():
        rel_results = row[0] if row[0] else []
        dst_id = row[1]
        
        if dst_id in rel_results:
            is_in_relative.append(True)
            relative_rank.append(rel_results.index(dst_id) + 1)  # 1-indexed
        else:
            is_in_relative.append(False)
            relative_rank.append(None)
    
    pairs_df = pairs_df.with_columns([
        pl.Series("is_in_relative_results", is_in_relative, dtype=pl.Boolean),
        pl.Series("relative_results_rank", relative_rank, dtype=pl.Int8),
    ])
    
    # Drop the temporary relative_results column
    pairs_df = pairs_df.drop("relative_results", "dst_id_check")
    
    print(f"  ✓ Added relationship features")
    return pairs_df


def add_all_features(pairs_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add all feature groups to pairs dataframe.
    
    Args:
        pairs_df: Pairs DataFrame
        biz_df: Business DataFrame
        
    Returns:
        DataFrame with all features added
    """
    print("\n" + "=" * 80)
    print("ADDING FEATURES TO POSITIVE PAIRS")
    print("=" * 80)
    
    pairs_df = add_spatial_features(pairs_df)
    pairs_df = add_temporal_features(pairs_df)
    pairs_df = add_quality_features(pairs_df)
    pairs_df = add_price_features(pairs_df, biz_df)
    pairs_df = add_category_features(pairs_df)
    pairs_df = add_relationship_features(pairs_df, biz_df)
    
    # Add label (1 for positive pairs)
    pairs_df = pairs_df.with_columns([
        pl.lit(1).alias("label")
    ])
    
    print("\n[7/7] Feature engineering complete!")
    print(f"  Total features: {len(pairs_df.columns)}")
    print(f"  Total positive pairs: {len(pairs_df):,}")
    
    return pairs_df


def generate_negative_samples(pairs_df: pl.DataFrame, biz_df: pl.DataFrame,
                              n_negatives: int = 4, random_seed: int = 42) -> pl.DataFrame:
    """
    Generate negative samples using hybrid sampling strategy.
    
    Strategy:
    - 50% random geographic (within 10km of source)
    - 30% from relative_results
    - 20% same category, different location
    
    Args:
        pairs_df: Positive pairs DataFrame
        biz_df: Business DataFrame
        n_negatives: Number of negative samples per positive
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame of negative pairs
    """
    print("\n" + "=" * 80)
    print(f"GENERATING NEGATIVE SAMPLES ({n_negatives}:1 ratio)")
    print("=" * 80)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Calculate how many of each type
    n_geo = int(n_negatives * 0.5)  # 50% geographic
    n_rel = int(n_negatives * 0.3)  # 30% relative_results
    n_cat = n_negatives - n_geo - n_rel  # 20% same category
    
    print(f"\nNegative sample distribution per positive:")
    print(f"  - Geographic (10km): {n_geo}")
    print(f"  - Relative results: {n_rel}")
    print(f"  - Same category: {n_cat}")
    print(f"  - Total: {n_negatives}")
    
    # Create business lookup dictionaries
    print("\n[1/4] Creating business lookup structures...")
    biz_dict = {}
    for row in biz_df.iter_rows(named=True):
        biz_dict[row['gmap_id']] = {
            'lat': row['lat'],
            'lon': row['lon'],
            'category': row['category_main'],
            'relative_results': row['relative_results'] if row['relative_results'] else []
        }
    
    all_biz_ids = list(biz_dict.keys())
    
    # Group businesses by category for same-category sampling
    category_businesses = {}
    for gmap_id, info in biz_dict.items():
        cat = info['category']
        if cat not in category_businesses:
            category_businesses[cat] = []
        category_businesses[cat].append(gmap_id)
    
    print(f"  ✓ Indexed {len(biz_dict):,} businesses")
    
    # Generate negative samples
    print(f"\n[2/4] Generating {len(pairs_df) * n_negatives:,} negative samples...")
    
    negative_samples = []
    
    for i, row in enumerate(pairs_df.iter_rows(named=True)):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i:,}/{len(pairs_df):,} ({i/len(pairs_df)*100:.1f}%)")
        
        src_id = row['src_gmap_id']
        dst_id = row['dst_gmap_id']  # The actual destination (to exclude)
        src_info = biz_dict.get(src_id)
        
        if not src_info:
            continue
        
        src_lat, src_lon = src_info['lat'], src_info['lon']
        src_cat = src_info['category']
        relative_results = src_info['relative_results']
        
        # Generate n_negatives samples
        sampled_negatives = set()
        
        # 1. Geographic negatives (within 10km)
        for _ in range(n_geo):
            attempts = 0
            while attempts < 50:  # Max attempts to find valid negative
                candidate = random.choice(all_biz_ids)
                if candidate != dst_id and candidate != src_id and candidate not in sampled_negatives:
                    cand_info = biz_dict[candidate]
                    dist = calculate_haversine_distance(src_lat, src_lon, cand_info['lat'], cand_info['lon'])
                    if dist <= 10.0:  # Within 10km
                        sampled_negatives.add(candidate)
                        break
                attempts += 1
        
        # 2. Relative results negatives
        if relative_results:
            available_rel = [r for r in relative_results if r != dst_id and r in biz_dict]
            if available_rel:
                n_sample = min(n_rel, len(available_rel))
                sampled_rel = random.sample(available_rel, n_sample)
                sampled_negatives.update(sampled_rel)
        
        # 3. Same category negatives
        if src_cat in category_businesses:
            available_cat = [b for b in category_businesses[src_cat] 
                           if b != dst_id and b != src_id and b not in sampled_negatives]
            if available_cat:
                n_sample = min(n_cat, len(available_cat))
                sampled_cat = random.sample(available_cat, n_sample)
                sampled_negatives.update(sampled_cat)
        
        # Create negative pair entries
        for neg_id in sampled_negatives:
            neg_info = biz_dict[neg_id]
            negative_samples.append({
                'user_id': row['user_id'],
                'src_gmap_id': src_id,
                'dst_gmap_id': neg_id,
                'src_ts': row['src_ts'],
                'dst_ts': row['dst_ts'],  # Keep same timestamp
                'delta_hours': row['delta_hours'],
                'src_category_main': row['src_category_main'],
                'dst_category_main': neg_info['category'],
                'src_lat': row['src_lat'],
                'src_lon': row['src_lon'],
                'dst_lat': neg_info['lat'],
                'dst_lon': neg_info['lon'],
                'src_rating': row['src_rating'],
                'dst_rating': 0,  # We don't have rating for negatives yet
            })
    
    print(f"  ✓ Generated {len(negative_samples):,} negative samples")
    
    # Convert to DataFrame
    print("\n[3/4] Converting to DataFrame...")
    neg_df = pl.DataFrame(negative_samples)
    
    # Add features to negative samples
    print("\n[4/4] Adding features to negative samples...")
    neg_df = add_all_features(neg_df, biz_df)
    
    # Override label to 0
    neg_df = neg_df.with_columns([
        pl.lit(0).alias("label")
    ])
    
    print(f"\n✓ Negative sampling complete!")
    print(f"  Total negative pairs: {len(neg_df):,}")
    print(f"  Negative:Positive ratio: {len(neg_df)/len(pairs_df):.1f}:1")
    
    return neg_df


def main():
    """Main feature engineering pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    processed_dir = base_dir / "data" / "processed" / "ga"
    
    pairs_input = processed_dir / "pairs_filtered_ga.parquet"
    biz_input = processed_dir / "biz_ga.parquet"
    
    features_output = processed_dir / "features_ga.parquet"
    
    print("=" * 80)
    print("PHASE A3: FEATURE ENGINEERING")
    print("=" * 80)
    print(f"\nInputs:")
    print(f"  - {pairs_input}")
    print(f"  - {biz_input}")
    print(f"\nOutput:")
    print(f"  - {features_output}")
    
    # Load data
    print("\n[Loading data...]")
    pairs_df = pl.read_parquet(pairs_input)
    biz_df = pl.read_parquet(biz_input)
    print(f"  Loaded {len(pairs_df):,} positive pairs")
    print(f"  Loaded {len(biz_df):,} businesses")
    
    # Add features to positive pairs
    pairs_with_features = add_all_features(pairs_df, biz_df)
    
    # Generate negative samples
    negative_pairs = generate_negative_samples(pairs_with_features, biz_df, n_negatives=4)
    
    # Combine positive and negative samples
    print("\n" + "=" * 80)
    print("COMBINING POSITIVE AND NEGATIVE SAMPLES")
    print("=" * 80)
    
    # Ensure schema compatibility by casting numeric columns to consistent types
    print("\nAligning schemas...")
    
    # Get common schema from positive pairs
    target_schema = pairs_with_features.schema
    
    # Cast negative pairs to match positive pairs schema
    cast_exprs = []
    for col_name, dtype in target_schema.items():
        if col_name in negative_pairs.columns:
            cast_exprs.append(pl.col(col_name).cast(dtype))
        else:
            # Column doesn't exist in negatives, this shouldn't happen
            print(f"  Warning: Column {col_name} not in negative pairs")
    
    if cast_exprs:
        negative_pairs = negative_pairs.select(cast_exprs)
    
    all_pairs = pl.concat([pairs_with_features, negative_pairs])
    
    print(f"\nFinal dataset:")
    print(f"  Positive pairs: {len(pairs_with_features):,}")
    print(f"  Negative pairs: {len(negative_pairs):,}")
    print(f"  Total pairs: {len(all_pairs):,}")
    print(f"  Label distribution: {(all_pairs['label'] == 1).sum():,} positive, {(all_pairs['label'] == 0).sum():,} negative")
    
    # Save to parquet
    print(f"\nWriting to {features_output}...")
    all_pairs.write_parquet(features_output, compression="snappy")
    print(f"  Output size: {Path(features_output).stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print feature summary
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    print(f"\nTotal columns: {len(all_pairs.columns)}")
    print(f"\nColumn names:")
    for i, col in enumerate(all_pairs.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n✓✓✓ PHASE A3 COMPLETE ✓✓✓\n")


if __name__ == "__main__":
    main()

