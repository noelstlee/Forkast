"""
Review Text Analysis Module
Extracts sentiment, topics, and behavioral patterns from review text.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import re


def extract_restaurant_sentiment(reviews_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract sentiment and text features per restaurant.
    
    Args:
        reviews_df: DataFrame with columns ['gmap_id', 'text', 'rating']
        
    Returns:
        DataFrame with sentiment features per restaurant
    """
    print("  [1/3] Calculating sentiment scores...")
    
    def calculate_sentiment(text):
        """Calculate sentiment using TextBlob."""
        if not text or pd.isna(text):
            return 0.0
        
        try:
            from textblob import TextBlob
            blob = TextBlob(str(text))
            return float(blob.sentiment.polarity)
        except Exception:
            # Fallback: simple keyword-based sentiment
            text_lower = str(text).lower()
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'awesome', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'poor']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.0
            return (pos_count - neg_count) / (pos_count + neg_count)
    
    def calculate_text_length(text):
        """Calculate text length safely."""
        if not text or pd.isna(text):
            return 0
        return len(str(text))
    
    # Add sentiment and length columns
    reviews_with_sentiment = reviews_df.with_columns([
        pl.col("text").map_elements(calculate_sentiment, return_dtype=pl.Float32).alias("sentiment"),
        pl.col("text").map_elements(calculate_text_length, return_dtype=pl.Int32).alias("text_length")
    ])
    
    print("  [2/3] Aggregating by restaurant...")
    
    # Aggregate by restaurant
    restaurant_sentiment = reviews_with_sentiment.group_by("gmap_id").agg([
        pl.col("sentiment").mean().alias("avg_sentiment"),
        pl.col("sentiment").std().alias("sentiment_std"),
        (pl.col("sentiment") > 0.2).mean().alias("positive_review_pct"),
        (pl.col("sentiment") < -0.2).mean().alias("negative_review_pct"),
        pl.col("text_length").mean().alias("avg_review_length"),
        pl.col("sentiment").count().alias("review_count_for_sentiment")
    ])
    
    # Fill nulls with defaults
    restaurant_sentiment = restaurant_sentiment.with_columns([
        pl.col("avg_sentiment").fill_null(0.0),
        pl.col("sentiment_std").fill_null(0.0),
        pl.col("positive_review_pct").fill_null(0.0),
        pl.col("negative_review_pct").fill_null(0.0),
        pl.col("avg_review_length").fill_null(0.0),
        pl.col("review_count_for_sentiment").fill_null(0)
    ])
    
    print(f"  ✓ Processed sentiment for {len(restaurant_sentiment):,} restaurants")
    return restaurant_sentiment


def extract_restaurant_topics(reviews_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract topic mentions per restaurant using keyword matching.
    
    Args:
        reviews_df: DataFrame with columns ['gmap_id', 'text']
        
    Returns:
        DataFrame with topic flags per restaurant
    """
    print("  [3/3] Extracting topic mentions...")
    
    # Define topic keywords
    TOPIC_KEYWORDS = {
        'dessert': ['dessert', 'sweet', 'cake', 'ice cream', 'pie', 'cookie', 'chocolate', 'candy', 'sugar'],
        'drinks': ['drink', 'cocktail', 'wine', 'beer', 'coffee', 'tea', 'juice', 'soda', 'alcohol', 'beverage'],
        'service': ['service', 'waiter', 'waitress', 'staff', 'server', 'friendly', 'rude', 'slow', 'fast', 'attentive'],
        'atmosphere': ['atmosphere', 'ambiance', 'ambience', 'decor', 'vibe', 'cozy', 'romantic', 'loud', 'quiet', 'music'],
        'price': ['price', 'expensive', 'cheap', 'affordable', 'cost', 'value', 'money', 'budget', 'pricey', 'deal']
    }
    
    def extract_topics(text):
        """Extract topic mentions from text."""
        if not text or pd.isna(text):
            return {topic: False for topic in TOPIC_KEYWORDS.keys()}
        
        text_lower = str(text).lower()
        topic_flags = {}
        
        for topic, keywords in TOPIC_KEYWORDS.items():
            topic_flags[topic] = any(keyword in text_lower for keyword in keywords)
        
        return topic_flags
    
    # Extract topics for each review
    reviews_with_topics = reviews_df.with_columns([
        pl.col("text").map_elements(extract_topics, return_dtype=pl.Struct({
            'dessert': pl.Boolean,
            'drinks': pl.Boolean,
            'service': pl.Boolean,
            'atmosphere': pl.Boolean,
            'price': pl.Boolean
        })).alias("topics")
    ])
    
    # Extract topic fields
    reviews_with_topics = reviews_with_topics.with_columns([
        pl.col("topics").struct.field("dessert").alias("mentions_dessert"),
        pl.col("topics").struct.field("drinks").alias("mentions_drinks"),
        pl.col("topics").struct.field("service").alias("mentions_service"),
        pl.col("topics").struct.field("atmosphere").alias("mentions_atmosphere"),
        pl.col("topics").struct.field("price").alias("mentions_price"),
    ]).drop("topics")
    
    # Aggregate by restaurant (percentage of reviews mentioning each topic)
    restaurant_topics = reviews_with_topics.group_by("gmap_id").agg([
        pl.col("mentions_dessert").mean().alias("mentions_dessert"),
        pl.col("mentions_drinks").mean().alias("mentions_drinks"),
        pl.col("mentions_service").mean().alias("mentions_service"),
        pl.col("mentions_atmosphere").mean().alias("mentions_atmosphere"),
        pl.col("mentions_price").mean().alias("mentions_price"),
    ])
    
    # Fill nulls
    restaurant_topics = restaurant_topics.with_columns([
        pl.col("mentions_dessert").fill_null(0.0),
        pl.col("mentions_drinks").fill_null(0.0),
        pl.col("mentions_service").fill_null(0.0),
        pl.col("mentions_atmosphere").fill_null(0.0),
        pl.col("mentions_price").fill_null(0.0),
    ])
    
    print(f"  ✓ Processed topics for {len(restaurant_topics):,} restaurants")
    return restaurant_topics


def calculate_review_similarity(src_reviews: pl.Series, dst_reviews: pl.Series) -> float:
    """
    Calculate semantic similarity between review sets using TF-IDF.
    
    Args:
        src_reviews: Series of review texts for source restaurant
        dst_reviews: Series of review texts for destination restaurant
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Combine and clean texts
        src_text = ' '.join([str(text) for text in src_reviews if text and not pd.isna(text)])
        dst_text = ' '.join([str(text) for text in dst_reviews if text and not pd.isna(text)])
        
        if not src_text or not dst_text:
            return 0.0
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([src_text, dst_text])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
        
    except Exception:
        # Fallback: simple word overlap
        src_words = set(src_text.lower().split()) if src_text else set()
        dst_words = set(dst_text.lower().split()) if dst_text else set()
        
        if not src_words or not dst_words:
            return 0.0
        
        intersection = len(src_words & dst_words)
        union = len(src_words | dst_words)
        
        return intersection / union if union > 0 else 0.0


def analyze_reviews(
    reviews_df: pl.DataFrame,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False
) -> Dict[str, pl.DataFrame]:
    """
    Complete review analysis pipeline.
    
    Args:
        reviews_df: DataFrame with review data
        
    Returns:
        Dictionary with sentiment and topic DataFrames
    """
    print("\n" + "=" * 60)
    print("REVIEW ANALYSIS")
    print("=" * 60)
    
    cache_dir: Optional[Path] = Path(cache_path) if cache_path else None
    sentiment_cache: Optional[Path] = None
    topics_cache: Optional[Path] = None

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        sentiment_cache = cache_dir / "review_sentiment.parquet"
        topics_cache = cache_dir / "review_topics.parquet"

    sentiment_df: Optional[pl.DataFrame] = None
    topics_df: Optional[pl.DataFrame] = None

    if (
        not force_refresh
        and sentiment_cache is not None
        and topics_cache is not None
        and sentiment_cache.exists()
        and topics_cache.exists()
    ):
        print("  ✓ Loaded cached review analysis")
        sentiment_df = pl.read_parquet(sentiment_cache)
        topics_df = pl.read_parquet(topics_cache)

    if sentiment_df is None or topics_df is None:
        # Check if TextBlob is available
        try:
            from textblob import TextBlob
            print("  ✓ TextBlob available for sentiment analysis")
        except ImportError:
            print("  ⚠ TextBlob not available, using fallback sentiment analysis")
            print("    Install with: pip install textblob")

        # Extract sentiment features
        sentiment_df = extract_restaurant_sentiment(reviews_df)

        # Extract topic features
        topics_df = extract_restaurant_topics(reviews_df)

        if sentiment_cache is not None and topics_cache is not None:
            sentiment_df.write_parquet(sentiment_cache)
            topics_df.write_parquet(topics_cache)
            print("  ✓ Cached review analysis results")
    
    print(f"\n✓ Review analysis complete!")
    print(f"  Sentiment features: {len(sentiment_df):,} restaurants")
    print(f"  Topic features: {len(topics_df):,} restaurants")
    
    return {
        'sentiment': sentiment_df,
        'topics': topics_df
    }


# Import pandas for null checking (used in functions above)
try:
    import pandas as pd
except ImportError:
    # Create a minimal pandas-like interface for null checking
    class pd:
        @staticmethod
        def isna(value):
            return value is None or (isinstance(value, str) and value.lower() in ['null', 'none', ''])
