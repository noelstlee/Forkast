"""
User Behavioral Profiling Module
Creates user-level features for personalization.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta


def create_user_profiles(
    reviews_df: pl.DataFrame,
    biz_df: pl.DataFrame,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False
) -> pl.DataFrame:
    """
    Generate comprehensive user behavioral profiles.
    
    Args:
        reviews_df: DataFrame with user reviews
        biz_df: DataFrame with business metadata
        
    Returns:
        DataFrame with user-level features
    """
    cache_file: Optional[Path] = Path(cache_path) if cache_path else None

    if (
        cache_file is not None
        and not force_refresh
        and cache_file.exists()
    ):
        print("\n" + "=" * 60)
        print("USER PROFILING (cached)")
        print("=" * 60)
        user_profiles = pl.read_parquet(cache_file)
        print(f"  ✓ Loaded cached user profiles ({len(user_profiles):,} users)")
        return user_profiles

    print("\n" + "=" * 60)
    print("USER PROFILING")
    print("=" * 60)
    
    print("  [1/6] Calculating basic user statistics...")
    
    # Basic user statistics
    user_basic_stats = reviews_df.group_by("user_id").agg([
        pl.col("rating").mean().alias("user_avg_rating"),
        pl.col("rating").std().alias("user_rating_std"),
        pl.col("rating").count().alias("user_total_reviews"),
        pl.col("gmap_id").n_unique().alias("user_unique_restaurants"),
        pl.col("ts").min().alias("user_first_review"),
        pl.col("ts").max().alias("user_last_review"),
        pl.col("text").str.len_chars().mean().alias("user_avg_review_length")
    ])
    
    print("  [2/6] Calculating visit frequency and time patterns...")
    
    # Calculate visit frequency (reviews per month)
    user_basic_stats = user_basic_stats.with_columns([
        # Time span in days
        (pl.col("user_last_review") - pl.col("user_first_review")).dt.total_days().alias("user_timespan_days"),
    ]).with_columns([
        # Visit frequency (reviews per month)
        pl.when(pl.col("user_timespan_days") > 0)
        .then(pl.col("user_total_reviews") * 30.0 / pl.col("user_timespan_days"))
        .otherwise(pl.col("user_total_reviews"))
        .alias("user_visit_frequency")
    ])
    
    print("  [3/6] Analyzing cuisine diversity and preferences...")
    
    # Join with business data to get categories and prices
    reviews_with_biz = reviews_df.join(
        biz_df.select(["gmap_id", "category_main", "price_bucket"]),
        on="gmap_id",
        how="left"
    )
    
    # Cuisine diversity and preferences
    user_cuisine_stats = reviews_with_biz.group_by("user_id").agg([
        pl.col("category_main").n_unique().alias("user_cuisine_diversity"),
        pl.col("price_bucket").mean().alias("user_price_preference"),
        pl.col("price_bucket").std().alias("user_price_std"),
        # Most common cuisine (mode)
        pl.col("category_main").mode().first().alias("user_favorite_cuisine")
    ])
    
    print("  [4/6] Calculating loyalty and exploration patterns...")
    
    # Loyalty score (percentage of repeat visits)
    user_loyalty = reviews_df.group_by("user_id").agg([
        # Count restaurants visited more than once
        (pl.col("gmap_id").value_counts().struct.field("count") > 1).sum().alias("repeat_restaurants"),
        pl.col("gmap_id").n_unique().alias("total_unique_restaurants")
    ]).with_columns([
        # Loyalty score as percentage
        pl.when(pl.col("total_unique_restaurants") > 0)
        .then(pl.col("repeat_restaurants") / pl.col("total_unique_restaurants"))
        .otherwise(0.0)
        .alias("user_loyalty_score")
    ]).select(["user_id", "user_loyalty_score"])
    
    print("  [5/6] Analyzing rating patterns and sentiment...")
    
    # Rating patterns
    user_rating_patterns = reviews_df.group_by("user_id").agg([
        # Rating distribution
        (pl.col("rating") == 5).mean().alias("user_pct_5_star"),
        (pl.col("rating") == 1).mean().alias("user_pct_1_star"),
        (pl.col("rating") >= 4).mean().alias("user_pct_positive"),
        (pl.col("rating") <= 2).mean().alias("user_pct_negative"),
        # Consistency
        (pl.col("rating").std() < 1.0).alias("user_is_consistent_rater")
    ])
    
    print("  [6/6] Combining and deriving final features...")
    
    # Combine all user statistics
    user_profiles = user_basic_stats.join(user_cuisine_stats, on="user_id", how="left") \
                                   .join(user_loyalty, on="user_id", how="left") \
                                   .join(user_rating_patterns, on="user_id", how="left")
    
    # Calculate global statistics for comparison
    global_stats = user_profiles.select([
        pl.col("user_unique_restaurants").median().alias("median_unique_restaurants"),
        pl.col("user_visit_frequency").median().alias("median_visit_frequency"),
        pl.col("user_cuisine_diversity").median().alias("median_cuisine_diversity")
    ]).row(0)
    
    median_unique_restaurants = global_stats[0]
    median_visit_frequency = global_stats[1]
    median_cuisine_diversity = global_stats[2]
    
    # Derive behavioral flags
    user_profiles = user_profiles.with_columns([
        # Explorer: visits many unique places
        (pl.col("user_unique_restaurants") > median_unique_restaurants).alias("user_is_explorer"),
        
        # Frequent visitor
        (pl.col("user_visit_frequency") > median_visit_frequency).alias("user_is_frequent"),
        
        # Diverse eater
        (pl.col("user_cuisine_diversity") > median_cuisine_diversity).alias("user_is_diverse_eater"),
        
        # High standards (avg rating > 4.0)
        (pl.col("user_avg_rating") > 4.0).alias("user_has_high_standards"),
        
        # Price sensitive (low price preference and low std)
        ((pl.col("user_price_preference") <= 2.0) & (pl.col("user_price_std") <= 0.5)).alias("user_is_price_sensitive"),
        
        # Visit frequency bucket
        pl.when(pl.col("user_visit_frequency") < median_visit_frequency * 0.5)
        .then(pl.lit("low"))
        .when(pl.col("user_visit_frequency") > median_visit_frequency * 2.0)
        .then(pl.lit("high"))
        .otherwise(pl.lit("medium"))
        .alias("user_visit_frequency_bucket")
    ])
    
    # Fill nulls with appropriate defaults
    user_profiles = user_profiles.with_columns([
        pl.col("user_avg_rating").fill_null(3.5),
        pl.col("user_rating_std").fill_null(1.0),
        pl.col("user_avg_review_length").fill_null(50.0),
        pl.col("user_visit_frequency").fill_null(1.0),
        pl.col("user_cuisine_diversity").fill_null(1),
        pl.col("user_price_preference").fill_null(2.0),
        pl.col("user_price_std").fill_null(1.0),
        pl.col("user_loyalty_score").fill_null(0.0),
        pl.col("user_favorite_cuisine").fill_null("american"),
        pl.col("user_pct_5_star").fill_null(0.2),
        pl.col("user_pct_1_star").fill_null(0.1),
        pl.col("user_pct_positive").fill_null(0.6),
        pl.col("user_pct_negative").fill_null(0.2),
        pl.col("user_is_consistent_rater").fill_null(False),
        pl.col("user_is_explorer").fill_null(False),
        pl.col("user_is_frequent").fill_null(False),
        pl.col("user_is_diverse_eater").fill_null(False),
        pl.col("user_has_high_standards").fill_null(False),
        pl.col("user_is_price_sensitive").fill_null(False),
        pl.col("user_visit_frequency_bucket").fill_null("low")
    ])
    
    # Select final columns
    user_profiles = user_profiles.select([
        "user_id",
        # Basic stats
        "user_avg_rating",
        "user_rating_std", 
        "user_total_reviews",
        "user_unique_restaurants",
        "user_avg_review_length",
        "user_visit_frequency",
        # Cuisine and price preferences
        "user_cuisine_diversity",
        "user_price_preference",
        "user_price_std",
        "user_favorite_cuisine",
        # Behavioral patterns
        "user_loyalty_score",
        "user_pct_5_star",
        "user_pct_1_star", 
        "user_pct_positive",
        "user_pct_negative",
        # Derived flags
        "user_is_explorer",
        "user_is_frequent",
        "user_is_diverse_eater",
        "user_has_high_standards",
        "user_is_price_sensitive",
        "user_is_consistent_rater",
        "user_visit_frequency_bucket"
    ])
    
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        user_profiles.write_parquet(cache_file)
        print(f"  ✓ Cached user profiles ({cache_file})")

    print(f"\n✓ User profiling complete!")
    print(f"  Processed {len(user_profiles):,} users")
    print(f"  Features per user: {len(user_profiles.columns) - 1}")
    
    # Print some summary statistics
    print(f"\n  User behavior summary:")
    print(f"    Explorers: {user_profiles['user_is_explorer'].sum():,} ({user_profiles['user_is_explorer'].mean()*100:.1f}%)")
    print(f"    Frequent visitors: {user_profiles['user_is_frequent'].sum():,} ({user_profiles['user_is_frequent'].mean()*100:.1f}%)")
    print(f"    Diverse eaters: {user_profiles['user_is_diverse_eater'].sum():,} ({user_profiles['user_is_diverse_eater'].mean()*100:.1f}%)")
    print(f"    High standards: {user_profiles['user_has_high_standards'].sum():,} ({user_profiles['user_has_high_standards'].mean()*100:.1f}%)")
    print(f"    Price sensitive: {user_profiles['user_is_price_sensitive'].sum():,} ({user_profiles['user_is_price_sensitive'].mean()*100:.1f}%)")
    
    return user_profiles


def get_user_top_cuisines(reviews_df: pl.DataFrame, biz_df: pl.DataFrame, top_n: int = 3) -> pl.DataFrame:
    """
    Get top N cuisines for each user.
    
    Args:
        reviews_df: DataFrame with user reviews
        biz_df: DataFrame with business metadata
        top_n: Number of top cuisines to return per user
        
    Returns:
        DataFrame with user's top cuisines
    """
    # Join with business data
    reviews_with_biz = reviews_df.join(
        biz_df.select(["gmap_id", "category_main"]),
        on="gmap_id",
        how="left"
    )
    
    # Count visits per cuisine per user
    user_cuisine_counts = reviews_with_biz.group_by(["user_id", "category_main"]).agg([
        pl.len().alias("visit_count")
    ])
    
    # Get top N cuisines per user
    user_top_cuisines = user_cuisine_counts.sort(["user_id", "visit_count"], descending=[False, True]) \
                                          .group_by("user_id") \
                                          .head(top_n) \
                                          .group_by("user_id") \
                                          .agg([
                                              pl.col("category_main").alias("top_cuisines"),
                                              pl.col("visit_count").alias("top_cuisine_counts")
                                          ])
    
    return user_top_cuisines


def calculate_user_similarity(user_profiles_df: pl.DataFrame, user_id_1: str, user_id_2: str) -> float:
    """
    Calculate similarity between two users based on their profiles.
    
    Args:
        user_profiles_df: DataFrame with user profiles
        user_id_1: First user ID
        user_id_2: Second user ID
        
    Returns:
        Similarity score between 0 and 1
    """
    # Get user profiles
    user1 = user_profiles_df.filter(pl.col("user_id") == user_id_1)
    user2 = user_profiles_df.filter(pl.col("user_id") == user_id_2)
    
    if len(user1) == 0 or len(user2) == 0:
        return 0.0
    
    # Compare numerical features
    numerical_features = [
        "user_avg_rating", "user_cuisine_diversity", "user_price_preference", 
        "user_loyalty_score", "user_visit_frequency"
    ]
    
    similarity_scores = []
    
    for feature in numerical_features:
        val1 = user1[feature].item()
        val2 = user2[feature].item()
        
        if val1 is None or val2 is None:
            continue
            
        # Normalize difference to 0-1 scale
        max_diff = abs(val1) + abs(val2) + 1e-6  # Avoid division by zero
        diff = abs(val1 - val2)
        similarity = 1 - (diff / max_diff)
        similarity_scores.append(similarity)
    
    # Compare categorical features
    categorical_features = [
        "user_is_explorer", "user_is_frequent", "user_is_diverse_eater",
        "user_has_high_standards", "user_is_price_sensitive"
    ]
    
    for feature in categorical_features:
        val1 = user1[feature].item()
        val2 = user2[feature].item()
        
        if val1 == val2:
            similarity_scores.append(1.0)
        else:
            similarity_scores.append(0.0)
    
    # Return average similarity
    return np.mean(similarity_scores) if similarity_scores else 0.0
