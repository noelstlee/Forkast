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
from datetime import datetime, timedelta
import random
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.review_analysis import analyze_reviews
from data.user_profiling import create_user_profiles


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


def add_review_sentiment_features(pairs_df: pl.DataFrame, reviews_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add review sentiment features to pairs dataframe.
    
    Features:
    - src_sentiment_score: Average sentiment of reviews for source restaurant
    - dst_sentiment_score: Average sentiment of reviews for destination
    - sentiment_transition: dst_sentiment - src_sentiment
    - is_sentiment_upgrade: dst_sentiment > src_sentiment
    - sentiment_similarity: 1 - abs(dst_sentiment - src_sentiment)
    - both_positive_sentiment: both src and dst have positive sentiment
    
    Args:
        pairs_df: Pairs DataFrame
        reviews_df: Reviews DataFrame
        
    Returns:
        DataFrame with sentiment features added
    """
    print("\n[8/14] Adding review sentiment features...")
    
    # Get restaurant sentiment analysis
    review_analysis_results = analyze_reviews(reviews_df)
    sentiment_df = review_analysis_results['sentiment']
    
    # Join source sentiment
    pairs_df = pairs_df.join(
        sentiment_df.select([
            "gmap_id", 
            "avg_sentiment", 
            "sentiment_std", 
            "positive_review_pct",
            "avg_review_length"
        ]).rename({
            "avg_sentiment": "src_sentiment_score",
            "sentiment_std": "src_sentiment_std", 
            "positive_review_pct": "src_positive_review_pct",
            "avg_review_length": "src_avg_review_length"
        }),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Join destination sentiment
    pairs_df = pairs_df.join(
        sentiment_df.select([
            "gmap_id", 
            "avg_sentiment", 
            "sentiment_std", 
            "positive_review_pct",
            "avg_review_length"
        ]).rename({
            "avg_sentiment": "dst_sentiment_score",
            "sentiment_std": "dst_sentiment_std", 
            "positive_review_pct": "dst_positive_review_pct",
            "avg_review_length": "dst_avg_review_length"
        }),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Create derived sentiment features
    pairs_df = pairs_df.with_columns([
        # Sentiment transition
        (pl.col("dst_sentiment_score") - pl.col("src_sentiment_score")).alias("sentiment_transition"),
        
        # Sentiment upgrade
        (pl.col("dst_sentiment_score") > pl.col("src_sentiment_score")).alias("is_sentiment_upgrade"),
        
        # Sentiment similarity (1 - normalized difference)
        (1.0 - pl.col("dst_sentiment_score").sub(pl.col("src_sentiment_score")).abs() / 2.0).alias("sentiment_similarity"),
        
        # Both positive sentiment
        ((pl.col("src_sentiment_score") > 0.2) & (pl.col("dst_sentiment_score") > 0.2)).alias("both_positive_sentiment"),
        
        # Review length difference
        (pl.col("dst_avg_review_length") - pl.col("src_avg_review_length")).alias("review_length_diff")
    ])
    
    # Fill nulls with defaults
    pairs_df = pairs_df.with_columns([
        pl.col("src_sentiment_score").fill_null(0.0),
        pl.col("dst_sentiment_score").fill_null(0.0),
        pl.col("src_sentiment_std").fill_null(0.0),
        pl.col("dst_sentiment_std").fill_null(0.0),
        pl.col("src_positive_review_pct").fill_null(0.5),
        pl.col("dst_positive_review_pct").fill_null(0.5),
        pl.col("src_avg_review_length").fill_null(50.0),
        pl.col("dst_avg_review_length").fill_null(50.0),
        pl.col("sentiment_transition").fill_null(0.0),
        pl.col("is_sentiment_upgrade").fill_null(False),
        pl.col("sentiment_similarity").fill_null(1.0),
        pl.col("both_positive_sentiment").fill_null(False),
        pl.col("review_length_diff").fill_null(0.0)
    ])
    
    print(f"  ✓ Added sentiment features")
    return pairs_df


def add_user_behavioral_features(pairs_df: pl.DataFrame, reviews_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add user behavioral features to pairs dataframe.
    
    Features:
    - user_avg_rating: User's average rating across all reviews
    - user_visit_frequency: User's visit frequency (reviews per month)
    - user_cuisine_diversity: Number of unique cuisines visited by user
    - user_is_explorer: Boolean (visits many unique places)
    - rating_vs_user_preference: dst_rating - user_avg_rating
    - price_vs_user_preference: dst_price - user_price_preference
    
    Args:
        pairs_df: Pairs DataFrame
        reviews_df: Reviews DataFrame
        biz_df: Business DataFrame
        
    Returns:
        DataFrame with user behavioral features added
    """
    print("\n[9/14] Adding user behavioral features...")
    
    # Get user profiles
    user_profiles = create_user_profiles(reviews_df, biz_df)
    
    # Join user profiles to pairs
    pairs_df = pairs_df.join(user_profiles, on="user_id", how="left")
    
    # Create derived features
    pairs_df = pairs_df.with_columns([
        # Rating vs user preference
        (pl.col("dst_rating") - pl.col("user_avg_rating")).alias("rating_vs_user_preference"),
        
        # Price vs user preference (need to get dst_price first)
        (pl.col("dst_price") - pl.col("user_price_preference")).alias("price_vs_user_preference"),
    ])
    
    # Fill nulls with global defaults
    pairs_df = pairs_df.with_columns([
        pl.col("user_avg_rating").fill_null(3.5),
        pl.col("user_rating_std").fill_null(1.0),
        pl.col("user_total_reviews").fill_null(1),
        pl.col("user_unique_restaurants").fill_null(1),
        pl.col("user_avg_review_length").fill_null(50.0),
        pl.col("user_visit_frequency").fill_null(1.0),
        pl.col("user_cuisine_diversity").fill_null(1),
        pl.col("user_price_preference").fill_null(2.0),
        pl.col("user_price_std").fill_null(1.0),
        pl.col("user_loyalty_score").fill_null(0.0),
        pl.col("user_is_explorer").fill_null(False),
        pl.col("user_is_frequent").fill_null(False),
        pl.col("user_is_diverse_eater").fill_null(False),
        pl.col("user_has_high_standards").fill_null(False),
        pl.col("user_is_price_sensitive").fill_null(False),
        pl.col("user_visit_frequency_bucket").fill_null("low"),
        pl.col("rating_vs_user_preference").fill_null(0.0),
        pl.col("price_vs_user_preference").fill_null(0.0)
    ])
    
    print(f"  ✓ Added user behavioral features")
    return pairs_df


def add_cuisine_complementarity(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add cuisine complementarity features to pairs dataframe.
    
    Features:
    - cuisine_pair_type: 'complementary', 'similar', 'diverse'
    - is_dessert_followup: Boolean (main meal → dessert/cafe)
    - is_meal_progression: Boolean (breakfast → lunch → dinner pattern)
    
    Args:
        pairs_df: Pairs DataFrame
        
    Returns:
        DataFrame with cuisine complementarity features added
    """
    print("\n[10/14] Adding cuisine complementarity features...")
    
    # Define complementary pairs
    COMPLEMENTARY_PAIRS = {
        ('pizza', 'ice_cream'), ('pizza', 'dessert'),
        ('burger', 'ice_cream'), ('burger', 'dessert'),
        ('sushi', 'dessert'), ('sushi', 'cafe'),
        ('mexican', 'dessert'), ('mexican', 'cafe'),
        ('korean', 'cafe'), ('korean', 'dessert'),
        ('bbq', 'dessert'), ('steakhouse', 'dessert'),
        ('breakfast', 'cafe'), ('brunch', 'cafe'),
        ('chinese', 'dessert'), ('italian', 'dessert'),
        ('american', 'ice_cream'), ('fast_food', 'dessert')
    }
    
    # Dessert categories
    DESSERT_CATEGORIES = {'dessert', 'ice_cream', 'cafe', 'bakery'}
    MAIN_MEAL_CATEGORIES = {'pizza', 'burger', 'sushi', 'mexican', 'korean', 'bbq', 'steakhouse', 'chinese', 'italian', 'american'}
    MEAL_PROGRESSION = {
        ('breakfast', 'american'), ('breakfast', 'burger'), ('breakfast', 'mexican'),
        ('american', 'dessert'), ('burger', 'cafe'), ('mexican', 'ice_cream')
    }
    
    def classify_cuisine_pair(src_cat, dst_cat):
        """Classify the cuisine pair type."""
        if not src_cat or not dst_cat:
            return 'diverse'
        
        pair = (src_cat, dst_cat)
        if pair in COMPLEMENTARY_PAIRS:
            return 'complementary'
        elif src_cat == dst_cat:
            return 'similar'
        else:
            return 'diverse'
    
    def is_dessert_followup(src_cat, dst_cat):
        """Check if this is a main meal to dessert transition."""
        if not src_cat or not dst_cat:
            return False
        return src_cat in MAIN_MEAL_CATEGORIES and dst_cat in DESSERT_CATEGORIES
    
    def is_meal_progression(src_cat, dst_cat):
        """Check if this follows meal progression pattern."""
        if not src_cat or not dst_cat:
            return False
        return (src_cat, dst_cat) in MEAL_PROGRESSION
    
    # Add cuisine complementarity features
    pairs_df = pairs_df.with_columns([
        pl.struct(['src_category_main', 'dst_category_main'])
        .map_elements(
            lambda x: classify_cuisine_pair(x['src_category_main'], x['dst_category_main']),
            return_dtype=pl.Utf8
        )
        .alias('cuisine_pair_type'),
        
        pl.struct(['src_category_main', 'dst_category_main'])
        .map_elements(
            lambda x: is_dessert_followup(x['src_category_main'], x['dst_category_main']),
            return_dtype=pl.Boolean
        )
        .alias('is_dessert_followup'),
        
        pl.struct(['src_category_main', 'dst_category_main'])
        .map_elements(
            lambda x: is_meal_progression(x['src_category_main'], x['dst_category_main']),
            return_dtype=pl.Boolean
        )
        .alias('is_meal_progression')
    ])
    
    print(f"  ✓ Added cuisine complementarity features")
    return pairs_df


def add_operating_hours_features(pairs_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add operating hours features to pairs dataframe.
    
    Features:
    - src_is_open_at_time: Boolean (was source open when visited)
    - dst_is_open_at_time: Boolean (will destination be open at predicted time)
    - hours_overlap: Boolean (do both restaurants have overlapping operating hours)
    - is_late_night_transition: Boolean (both open after 11pm)
    - is_early_morning_transition: Boolean (both open before 9am)
    
    Args:
        pairs_df: Pairs DataFrame
        biz_df: Business DataFrame with operating hours
        
    Returns:
        DataFrame with operating hours features added
    """
    print("\n[11/14] Adding operating hours features...")
    
    import json
    
    def check_restaurant_open(operating_hours_json, timestamp, day_of_week, hour):
        """Check if restaurant is open at given time."""
        if not operating_hours_json:
            return True  # Assume open if no hours data
        
        try:
            hours_dict = json.loads(operating_hours_json)
            if not hours_dict:
                return True
            
            # Get day name from day_of_week (0=Monday, 6=Sunday)
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day_name = day_names[day_of_week % 7]
            
            if day_name not in hours_dict:
                return True  # Assume open if day not specified
            
            day_hours = hours_dict[day_name]
            if day_hours is None:  # Closed day
                return False
            
            if isinstance(day_hours, list) and len(day_hours) == 2:
                open_time, close_time = day_hours
                
                # Handle overnight hours (close_time > 24)
                if close_time > 24:
                    return hour >= open_time or hour <= (close_time - 24)
                else:
                    return open_time <= hour <= close_time
            
            return True  # Default to open if can't parse
            
        except Exception:
            return True  # Default to open if parsing fails
    
    def calculate_hours_overlap(src_hours_json, dst_hours_json):
        """Calculate if two restaurants have overlapping operating hours."""
        if not src_hours_json or not dst_hours_json:
            return True  # Assume overlap if no data
        
        try:
            src_hours = json.loads(src_hours_json)
            dst_hours = json.loads(dst_hours_json)
            
            if not src_hours or not dst_hours:
                return True
            
            # Check overlap for each day
            overlap_days = 0
            total_days = 0
            
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            
            for day in day_names:
                if day in src_hours and day in dst_hours:
                    total_days += 1
                    src_day = src_hours[day]
                    dst_day = dst_hours[day]
                    
                    if src_day is None or dst_day is None:
                        continue  # One is closed
                    
                    if (isinstance(src_day, list) and len(src_day) == 2 and
                        isinstance(dst_day, list) and len(dst_day) == 2):
                        
                        src_open, src_close = src_day
                        dst_open, dst_close = dst_day
                        
                        # Check for overlap
                        if not (src_close <= dst_open or dst_close <= src_open):
                            overlap_days += 1
            
            return overlap_days > 0 if total_days > 0 else True
            
        except Exception:
            return True
    
    def has_late_night_hours(hours_json):
        """Check if restaurant is open after 11pm."""
        if not hours_json:
            return False
        
        try:
            hours_dict = json.loads(hours_json)
            if not hours_dict:
                return False
            
            for day_hours in hours_dict.values():
                if day_hours and isinstance(day_hours, list) and len(day_hours) == 2:
                    _, close_time = day_hours
                    if close_time >= 23 or close_time <= 3:  # Open after 11pm or until 3am
                        return True
            
            return False
        except Exception:
            return False
    
    def has_early_morning_hours(hours_json):
        """Check if restaurant is open before 9am."""
        if not hours_json:
            return False
        
        try:
            hours_dict = json.loads(hours_json)
            if not hours_dict:
                return False
            
            for day_hours in hours_dict.values():
                if day_hours and isinstance(day_hours, list) and len(day_hours) == 2:
                    open_time, _ = day_hours
                    if open_time <= 9:  # Open before 9am
                        return True
            
            return False
        except Exception:
            return False
    
    # Join operating hours data
    pairs_df = pairs_df.join(
        biz_df.select(["gmap_id", "operating_hours_parsed", "has_late_night", "is_24hr"]).rename({
            "operating_hours_parsed": "src_operating_hours",
            "has_late_night": "src_has_late_night",
            "is_24hr": "src_is_24hr"
        }),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    ).join(
        biz_df.select(["gmap_id", "operating_hours_parsed", "has_late_night", "is_24hr"]).rename({
            "operating_hours_parsed": "dst_operating_hours",
            "has_late_night": "dst_has_late_night",
            "is_24hr": "dst_is_24hr"
        }),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Add time-based features
    pairs_df = pairs_df.with_columns([
        # Extract day of week and hour from timestamps
        pl.col("src_ts").dt.weekday().alias("src_day_of_week_hours"),
        pl.col("dst_ts").dt.weekday().alias("dst_day_of_week_hours"),
        pl.col("src_ts").dt.hour().alias("src_hour_hours"),
        pl.col("dst_ts").dt.hour().alias("dst_hour_hours"),
    ])
    
    # Check if restaurants are open at visit times
    pairs_df = pairs_df.with_columns([
        pl.struct(["src_operating_hours", "src_ts", "src_day_of_week_hours", "src_hour_hours"])
        .map_elements(
            lambda x: check_restaurant_open(x["src_operating_hours"], x["src_ts"], x["src_day_of_week_hours"], x["src_hour_hours"]),
            return_dtype=pl.Boolean
        )
        .alias("src_is_open_at_time"),
        
        pl.struct(["dst_operating_hours", "dst_ts", "dst_day_of_week_hours", "dst_hour_hours"])
        .map_elements(
            lambda x: check_restaurant_open(x["dst_operating_hours"], x["dst_ts"], x["dst_day_of_week_hours"], x["dst_hour_hours"]),
            return_dtype=pl.Boolean
        )
        .alias("dst_is_open_at_time"),
        
        # Hours overlap
        pl.struct(["src_operating_hours", "dst_operating_hours"])
        .map_elements(
            lambda x: calculate_hours_overlap(x["src_operating_hours"], x["dst_operating_hours"]),
            return_dtype=pl.Boolean
        )
        .alias("hours_overlap"),
        
        # Late night transition (both open late)
        (pl.col("src_has_late_night") & pl.col("dst_has_late_night")).alias("is_late_night_transition"),
        
        # Early morning transition (both open early)
        pl.struct(["src_operating_hours"])
        .map_elements(
            lambda x: has_early_morning_hours(x["src_operating_hours"]),
            return_dtype=pl.Boolean
        )
        .alias("src_has_early_hours"),
        
        pl.struct(["dst_operating_hours"])
        .map_elements(
            lambda x: has_early_morning_hours(x["dst_operating_hours"]),
            return_dtype=pl.Boolean
        )
        .alias("dst_has_early_hours"),
    ]).with_columns([
        (pl.col("src_has_early_hours") & pl.col("dst_has_early_hours")).alias("is_early_morning_transition"),
        
        # 24 hour transition
        (pl.col("src_is_24hr") | pl.col("dst_is_24hr")).alias("involves_24hr_restaurant")
    ])
    
    # Fill nulls with defaults
    pairs_df = pairs_df.with_columns([
        pl.col("src_is_open_at_time").fill_null(True),
        pl.col("dst_is_open_at_time").fill_null(True),
        pl.col("hours_overlap").fill_null(True),
        pl.col("is_late_night_transition").fill_null(False),
        pl.col("is_early_morning_transition").fill_null(False),
        pl.col("involves_24hr_restaurant").fill_null(False),
        pl.col("src_has_late_night").fill_null(False),
        pl.col("dst_has_late_night").fill_null(False),
        pl.col("src_is_24hr").fill_null(False),
        pl.col("dst_is_24hr").fill_null(False)
    ])
    
    # Drop temporary columns
    pairs_df = pairs_df.drop([
        "src_operating_hours", "dst_operating_hours", 
        "src_day_of_week_hours", "dst_day_of_week_hours",
        "src_hour_hours", "dst_hour_hours",
        "src_has_early_hours", "dst_has_early_hours"
    ])
    
    print(f"  ✓ Added operating hours features")
    return pairs_df


def add_review_topic_features(pairs_df: pl.DataFrame, reviews_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add review topic features to pairs dataframe.
    
    Features:
    - src_mentions_dessert: Percentage of reviews mentioning dessert topics
    - dst_mentions_dessert: Percentage of reviews mentioning dessert topics
    - topic_transition_dessert: Transition from non-dessert to dessert mentions
    - topic_similarity: Count of matching topic flags
    
    Args:
        pairs_df: Pairs DataFrame
        reviews_df: Reviews DataFrame
        
    Returns:
        DataFrame with topic features added
    """
    print("\n[12/14] Adding review topic features...")
    
    # Get restaurant topic analysis
    review_analysis_results = analyze_reviews(reviews_df)
    topics_df = review_analysis_results['topics']
    
    # Join source topics
    pairs_df = pairs_df.join(
        topics_df.select([
            "gmap_id", 
            "mentions_dessert", 
            "mentions_drinks", 
            "mentions_service", 
            "mentions_atmosphere",
            "mentions_price"
        ]).rename({
            "mentions_dessert": "src_mentions_dessert",
            "mentions_drinks": "src_mentions_drinks",
            "mentions_service": "src_mentions_service",
            "mentions_atmosphere": "src_mentions_atmosphere",
            "mentions_price": "src_mentions_price"
        }),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Join destination topics
    pairs_df = pairs_df.join(
        topics_df.select([
            "gmap_id", 
            "mentions_dessert", 
            "mentions_drinks", 
            "mentions_service", 
            "mentions_atmosphere",
            "mentions_price"
        ]).rename({
            "mentions_dessert": "dst_mentions_dessert",
            "mentions_drinks": "dst_mentions_drinks",
            "mentions_service": "dst_mentions_service",
            "mentions_atmosphere": "dst_mentions_atmosphere",
            "mentions_price": "dst_mentions_price"
        }),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Fill nulls with defaults FIRST before creating derived features
    topic_columns = [
        "src_mentions_dessert", "dst_mentions_dessert",
        "src_mentions_drinks", "dst_mentions_drinks", 
        "src_mentions_service", "dst_mentions_service",
        "src_mentions_atmosphere", "dst_mentions_atmosphere",
        "src_mentions_price", "dst_mentions_price"
    ]
    
    # Fill nulls for all topic columns
    fill_exprs = [pl.col(col).fill_null(0.0) for col in topic_columns if col in pairs_df.columns]
    if fill_exprs:
        pairs_df = pairs_df.with_columns(fill_exprs)
    
    # Create topic transition features (now that nulls are filled)
    # First create boolean columns, then convert to integers for counting
    pairs_df = pairs_df.with_columns([
        # Dessert transition (non-dessert to dessert)
        ((pl.col("src_mentions_dessert") < 0.3) & (pl.col("dst_mentions_dessert") > 0.3)).alias("topic_transition_dessert"),
        
        # Drinks transition
        ((pl.col("src_mentions_drinks") < 0.3) & (pl.col("dst_mentions_drinks") > 0.3)).alias("topic_transition_drinks"),
        
        # Individual similarity flags
        ((pl.col("src_mentions_dessert") - pl.col("dst_mentions_dessert")).abs() < 0.2).alias("_sim_dessert"),
        ((pl.col("src_mentions_drinks") - pl.col("dst_mentions_drinks")).abs() < 0.2).alias("_sim_drinks"),
        ((pl.col("src_mentions_service") - pl.col("dst_mentions_service")).abs() < 0.2).alias("_sim_service"),
        ((pl.col("src_mentions_atmosphere") - pl.col("dst_mentions_atmosphere")).abs() < 0.2).alias("_sim_atmosphere"),
        ((pl.col("src_mentions_price") - pl.col("dst_mentions_price")).abs() < 0.2).alias("_sim_price"),
    ])
    
    # Calculate similarity count by summing boolean flags converted to integers
    pairs_df = pairs_df.with_columns([
        (
            pl.col("_sim_dessert").cast(pl.Int8) + 
            pl.col("_sim_drinks").cast(pl.Int8) + 
            pl.col("_sim_service").cast(pl.Int8) + 
            pl.col("_sim_atmosphere").cast(pl.Int8) + 
            pl.col("_sim_price").cast(pl.Int8)
        ).alias("topic_similarity_count")
    ]).drop(["_sim_dessert", "_sim_drinks", "_sim_service", "_sim_atmosphere", "_sim_price"])
    
    # Fill nulls for derived columns (safety check)
    pairs_df = pairs_df.with_columns([
        pl.col("topic_transition_dessert").fill_null(False),
        pl.col("topic_transition_drinks").fill_null(False),
        pl.col("topic_similarity_count").fill_null(0)
    ])
    
    print(f"  ✓ Added review topic features")
    return pairs_df


def add_service_options_features(pairs_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add service options features to pairs dataframe.
    
    Features:
    - src_has_delivery: Boolean (source has delivery)
    - dst_has_delivery: Boolean (destination has delivery)
    - service_match: Boolean (same service options)
    - dst_adds_delivery: Boolean (destination adds delivery option)
    
    Args:
        pairs_df: Pairs DataFrame
        biz_df: Business DataFrame with service options
        
    Returns:
        DataFrame with service options features added
    """
    print("\n[13/14] Adding service options features...")
    
    # Join source service options
    pairs_df = pairs_df.join(
        biz_df.select([
            "gmap_id", 
            "has_delivery", 
            "has_takeout", 
            "has_dinein",
            "accepts_reservations"
        ]).rename({
            "has_delivery": "src_has_delivery",
            "has_takeout": "src_has_takeout",
            "has_dinein": "src_has_dinein",
            "accepts_reservations": "src_accepts_reservations"
        }),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Join destination service options
    pairs_df = pairs_df.join(
        biz_df.select([
            "gmap_id", 
            "has_delivery", 
            "has_takeout", 
            "has_dinein",
            "accepts_reservations"
        ]).rename({
            "has_delivery": "dst_has_delivery",
            "has_takeout": "dst_has_takeout",
            "has_dinein": "dst_has_dinein",
            "accepts_reservations": "dst_accepts_reservations"
        }),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    # Create service option features
    pairs_df = pairs_df.with_columns([
        # Service match (same delivery and takeout options)
        ((pl.col("src_has_delivery") == pl.col("dst_has_delivery")) & 
         (pl.col("src_has_takeout") == pl.col("dst_has_takeout"))).alias("service_match"),
        
        # Destination adds delivery
        ((pl.col("src_has_delivery") == False) & (pl.col("dst_has_delivery") == True)).alias("dst_adds_delivery"),
        
        # Destination adds takeout
        ((pl.col("src_has_takeout") == False) & (pl.col("dst_has_takeout") == True)).alias("dst_adds_takeout"),
        
        # Both have reservations
        (pl.col("src_accepts_reservations") & pl.col("dst_accepts_reservations")).alias("both_accept_reservations")
    ])
    
    # Fill nulls with defaults
    service_columns = [
        "src_has_delivery", "dst_has_delivery",
        "src_has_takeout", "dst_has_takeout",
        "src_has_dinein", "dst_has_dinein",
        "src_accepts_reservations", "dst_accepts_reservations"
    ]
    
    for col in service_columns:
        pairs_df = pairs_df.with_columns([pl.col(col).fill_null(False)])
    
    pairs_df = pairs_df.with_columns([
        pl.col("service_match").fill_null(True),
        pl.col("dst_adds_delivery").fill_null(False),
        pl.col("dst_adds_takeout").fill_null(False),
        pl.col("both_accept_reservations").fill_null(False)
    ])
    
    print(f"  ✓ Added service options features")
    return pairs_df


def add_all_features(pairs_df: pl.DataFrame, biz_df: pl.DataFrame, reviews_df: pl.DataFrame = None) -> pl.DataFrame:
    """
    Add all feature groups to pairs dataframe.
    
    Args:
        pairs_df: Pairs DataFrame
        biz_df: Business DataFrame
        reviews_df: Reviews DataFrame (optional, for sentiment and user features)
        
    Returns:
        DataFrame with all features added
    """
    print("\n" + "=" * 80)
    print("ADDING FEATURES TO POSITIVE PAIRS")
    print("=" * 80)
    
    # Original features (1-7)
    pairs_df = add_spatial_features(pairs_df)
    pairs_df = add_temporal_features(pairs_df)
    pairs_df = add_quality_features(pairs_df)
    pairs_df = add_price_features(pairs_df, biz_df)
    pairs_df = add_category_features(pairs_df)
    pairs_df = add_relationship_features(pairs_df, biz_df)
    
    # New enhanced features (8-14)
    if reviews_df is not None:
        pairs_df = add_review_sentiment_features(pairs_df, reviews_df)
        pairs_df = add_user_behavioral_features(pairs_df, reviews_df, biz_df)
        pairs_df = add_review_topic_features(pairs_df, reviews_df)
    
    pairs_df = add_cuisine_complementarity(pairs_df)
    pairs_df = add_operating_hours_features(pairs_df, biz_df)
    pairs_df = add_service_options_features(pairs_df, biz_df)
    
    # Add label (1 for positive pairs)
    pairs_df = pairs_df.with_columns([
        pl.lit(1).alias("label")
    ])
    
    print(f"\n[14/14] Feature engineering complete!")
    print(f"  Total features: {len(pairs_df.columns)}")
    print(f"  Total positive pairs: {len(pairs_df):,}")
    
    return pairs_df


def generate_negative_samples(pairs_df: pl.DataFrame, biz_df: pl.DataFrame,
                              n_negatives: int = 4, random_seed: int = 42) -> pl.DataFrame:
    """
    Generate negative samples using hybrid sampling strategy with realistic timestamps.
    
    Strategy:
    - 50% random geographic (within 10km of source)
    - 30% from relative_results
    - 20% same category, different location
    
    Key Fix: Generate realistic timestamps for negatives to prevent data leakage
    
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
        
        # Create negative pair entries with realistic timestamps
        for neg_id in sampled_negatives:
            neg_info = biz_dict[neg_id]
            
            # Generate realistic time gap (0.2 to 168 hours like positive samples)
            realistic_delta_hours = random.uniform(0.2, 168.0)
            
            # Calculate realistic destination timestamp
            src_ts = row['src_ts']
            
            # Handle different timestamp formats
            # Polars datetime objects can be added to timedelta directly
            if isinstance(src_ts, str):
                # Parse string timestamp
                try:
                    src_datetime = datetime.fromisoformat(src_ts.replace('Z', '+00:00'))
                except:
                    # Fallback: use a default datetime
                    src_datetime = datetime(2020, 1, 1)
            else:
                # Already a datetime-like object (Polars datetime or Python datetime)
                src_datetime = src_ts
            
            # Add realistic time delta
            # Works for both Python datetime and Polars datetime objects
            dst_datetime = src_datetime + timedelta(hours=realistic_delta_hours)
            
            negative_samples.append({
                'user_id': row['user_id'],
                'src_gmap_id': src_id,
                'dst_gmap_id': neg_id,
                'src_ts': row['src_ts'],
                'dst_ts': dst_datetime,  # Realistic timestamp
                'delta_hours': realistic_delta_hours,  # Realistic time gap
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
    
    # Load reviews for sentiment and user features
    reviews_input = processed_dir / "reviews_ga.parquet"
    reviews_df = pl.read_parquet(reviews_input)
    
    print(f"  Loaded {len(pairs_df):,} positive pairs")
    print(f"  Loaded {len(biz_df):,} businesses")
    print(f"  Loaded {len(reviews_df):,} reviews")
    
    # Add features to positive pairs
    pairs_with_features = add_all_features(pairs_df, biz_df, reviews_df)
    
    # Generate negative samples
    negative_pairs = generate_negative_samples(pairs_with_features, biz_df, n_negatives=4)
    
    # Add features to negative samples (they need the same features)
    if len(negative_pairs) > 0:
        print("\nAdding features to negative samples...")
        if reviews_df is not None:
            negative_pairs = add_review_sentiment_features(negative_pairs, reviews_df)
            negative_pairs = add_user_behavioral_features(negative_pairs, reviews_df, biz_df)
            negative_pairs = add_review_topic_features(negative_pairs, reviews_df)
        
        negative_pairs = add_cuisine_complementarity(negative_pairs)
        negative_pairs = add_operating_hours_features(negative_pairs, biz_df)
        negative_pairs = add_service_options_features(negative_pairs, biz_df)
    
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

