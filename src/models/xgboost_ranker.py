"""
Phase B1: XGBoost Ranking Model
Trains a Learning-to-Rank model to predict which restaurant a user will visit next.

Input:
- data/processed/ga/xgboost_data/train.parquet
- data/processed/ga/xgboost_data/val.parquet
- data/processed/ga/xgboost_data/test.parquet

Output:
- models/xgboost_ranker.json (trained model)
- models/predictions/xgboost_predictions.parquet (test predictions)
- models/metrics/xgboost_metrics.json (evaluation metrics)
"""

import polars as pl
import xgboost as xgb
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from collections import defaultdict


# Feature groups based on actual data schema
# Categorical features need to be one-hot encoded for XGBoost

CATEGORICAL_FEATURES = [
    'distance_bucket',
    'direction',
    'delta_hours_bucket',
    'dst_meal_type',
    # New categorical features
    'cuisine_pair_type',
    'user_visit_frequency_bucket'
]

NUMERICAL_FEATURES = [
    # Spatial
    'distance_km',

    # Temporal
    'delta_hours',
    'src_hour', 'dst_hour',
    'src_day_of_week', 'dst_day_of_week',

    # Quality/Rating
    'src_rating', 'dst_rating',
    'rating_diff',

    # Price
    'src_price', 'dst_price',
    'price_diff',

    # Relationship
    'relative_results_rank',
    
    # Review Sentiment Features
    'src_sentiment_score', 'dst_sentiment_score',
    'sentiment_transition', 'sentiment_similarity',
    'src_avg_review_length', 'dst_avg_review_length',
    'review_length_diff',
    'src_positive_review_pct', 'dst_positive_review_pct',
    
    # User Behavioral Features
    'user_avg_rating', 'user_rating_std',
    'user_total_reviews', 'user_unique_restaurants',
    'user_visit_frequency', 'user_cuisine_diversity',
    'user_price_preference', 'user_price_std',
    'user_loyalty_score', 'user_pct_5_star', 'user_pct_1_star',
    'user_pct_positive', 'user_pct_negative',
    'rating_vs_user_preference', 'price_vs_user_preference',
    
    # Topic Features
    'src_mentions_dessert', 'dst_mentions_dessert',
    'src_mentions_drinks', 'dst_mentions_drinks',
    'src_mentions_service', 'dst_mentions_service',
    'src_mentions_atmosphere', 'dst_mentions_atmosphere',
    'src_mentions_price', 'dst_mentions_price',
    'topic_similarity_count'
]

BOOLEAN_FEATURES = [
    # Spatial
    'same_neighborhood',

    # Temporal
    'src_is_weekend', 'dst_is_weekend',
    'src_is_breakfast', 'dst_is_breakfast',
    'src_is_lunch', 'dst_is_lunch',
    'src_is_dinner', 'dst_is_dinner',

    # Quality
    'is_rating_upgrade', 'is_rating_downgrade',
    'src_is_highly_rated', 'dst_is_highly_rated',

    # Price
    'is_price_upgrade', 'is_price_downgrade',
    'same_price_level',

    # Category
    'same_category',

    # Relationship
    'is_in_relative_results',
    
    # Review Sentiment Features
    'is_sentiment_upgrade', 'both_positive_sentiment',
    
    # User Behavioral Features
    'user_is_explorer', 'user_is_frequent', 'user_is_diverse_eater',
    'user_has_high_standards', 'user_is_price_sensitive', 'user_is_consistent_rater',
    
    # Cuisine Complementarity Features
    'is_dessert_followup', 'is_meal_progression',
    
    # Operating Hours Features
    'src_is_open_at_time', 'dst_is_open_at_time',
    'hours_overlap', 'is_late_night_transition', 'is_early_morning_transition',
    'involves_24hr_restaurant', 'src_has_late_night', 'dst_has_late_night',
    'src_is_24hr', 'dst_is_24hr',
    
    # Topic Features
    'topic_transition_dessert', 'topic_transition_drinks',
    
    # Service Options Features
    'src_has_delivery', 'dst_has_delivery', 'src_has_takeout', 'dst_has_takeout',
    'src_has_dinein', 'dst_has_dinein', 'src_accepts_reservations', 'dst_accepts_reservations',
    'service_match', 'dst_adds_delivery', 'dst_adds_takeout', 'both_accept_reservations'
]

ALL_BASE_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES


def create_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create interaction features to capture relationships between different aspects.
    
    These interaction features help the model learn complex patterns like:
    - Distance × Rating interaction (nearby high-rated places)
    - Price × Category interaction (expensive sushi vs cheap burger)
    - Time × Service options (late night delivery)
    - User preferences × Restaurant attributes
    
    Args:
        df: Input DataFrame with base features
        
    Returns:
        DataFrame with additional interaction features
    """
    print("  🔗 Creating interaction features...")
    
    interactions = []
    
    # Distance × Quality: Nearby high-rated restaurants
    if 'distance_km' in df.columns and 'dst_rating' in df.columns:
        interactions.append(
            (pl.col('distance_km') * pl.col('dst_rating')).alias('distance_x_dst_rating')
        )
        interactions.append(
            (pl.col('distance_km') * pl.col('rating_diff')).alias('distance_x_rating_diff')
        )
    
    # Price × Quality: Value proposition
    if 'dst_price' in df.columns and 'dst_rating' in df.columns:
        interactions.append(
            (pl.col('dst_price') * pl.col('dst_rating')).alias('price_x_rating')
        )
    
    # Time × Distance: Temporal proximity
    if 'delta_hours' in df.columns and 'distance_km' in df.columns:
        interactions.append(
            (pl.col('delta_hours') * pl.col('distance_km')).alias('time_x_distance')
        )
    
    # User preferences × Restaurant attributes
    if 'user_avg_rating' in df.columns and 'dst_rating' in df.columns:
        interactions.append(
            (pl.col('user_avg_rating') * pl.col('dst_rating')).alias('user_pref_x_dst_rating')
        )
    
    if 'user_price_preference' in df.columns and 'dst_price' in df.columns:
        interactions.append(
            (pl.col('user_price_preference') * pl.col('dst_price')).alias('user_price_pref_x_dst_price')
        )
    
    # Sentiment × Quality
    if 'src_sentiment_score' in df.columns and 'dst_sentiment_score' in df.columns:
        interactions.append(
            (pl.col('src_sentiment_score') * pl.col('dst_sentiment_score')).alias('sentiment_match_score')
        )
    
    # Category match × Distance (same category nearby)
    if 'same_category' in df.columns and 'distance_km' in df.columns:
        interactions.append(
            (pl.col('same_category').cast(pl.Float32) * pl.col('distance_km')).alias('same_cat_x_distance')
        )
    
    # Rating upgrade × Distance (worth traveling for better place)
    if 'is_rating_upgrade' in df.columns and 'distance_km' in df.columns:
        interactions.append(
            (pl.col('is_rating_upgrade').cast(pl.Float32) * pl.col('distance_km')).alias('rating_upgrade_x_distance')
        )
    
    # User diversity × Category difference
    if 'user_cuisine_diversity' in df.columns and 'same_category' in df.columns:
        interactions.append(
            (pl.col('user_cuisine_diversity') * (1 - pl.col('same_category').cast(pl.Float32))).alias('diversity_x_different_cat')
        )
    
    # Service match × Time
    if 'service_match' in df.columns and 'delta_hours' in df.columns:
        interactions.append(
            (pl.col('service_match').cast(pl.Float32) * pl.col('delta_hours')).alias('service_match_x_time')
        )
    
    # Relative results × Distance (Google recommendations nearby)
    if 'is_in_relative_results' in df.columns and 'distance_km' in df.columns:
        interactions.append(
            (pl.col('is_in_relative_results').cast(pl.Float32) * (1 / (pl.col('distance_km') + 0.1))).alias('relative_results_proximity')
        )
    
    # Add polynomial features for key metrics (squared terms capture non-linear relationships)
    if 'distance_km' in df.columns:
        interactions.append(
            (pl.col('distance_km') ** 2).alias('distance_km_squared')
        )
    
    if 'delta_hours' in df.columns:
        interactions.append(
            (pl.col('delta_hours') ** 2).alias('delta_hours_squared')
        )
    
    if 'rating_diff' in df.columns:
        interactions.append(
            (pl.col('rating_diff') ** 2).alias('rating_diff_squared')
        )
    
    # Ratio features (capture relative differences)
    if 'dst_rating' in df.columns and 'src_rating' in df.columns:
        interactions.append(
            (pl.col('dst_rating') / (pl.col('src_rating') + 0.1)).alias('rating_ratio')
        )
    
    if 'dst_price' in df.columns and 'src_price' in df.columns:
        interactions.append(
            (pl.col('dst_price') / (pl.col('src_price') + 0.1)).alias('price_ratio')
        )
    
    # Add interactions if we have the base features
    if interactions:
        df = df.with_columns(interactions)
        print(f"    Added {len(interactions)} interaction features")
    
    return df


def load_data(data_dir: Path) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load train, validation, and test data.

    Args:
        data_dir: Path to xgboost_data directory

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"

    print(f"\n[1/3] Loading training data...")
    train_df = pl.read_parquet(train_path)
    print(f"  ✓ {len(train_df):,} samples")
    print(f"  Positive: {(train_df['label'] == 1).sum():,}")
    print(f"  Negative: {(train_df['label'] == 0).sum():,}")

    print(f"\n[2/3] Loading validation data...")
    val_df = pl.read_parquet(val_path)
    print(f"  ✓ {len(val_df):,} samples")
    print(f"  Positive: {(val_df['label'] == 1).sum():,}")
    print(f"  Negative: {(val_df['label'] == 0).sum():,}")

    print(f"\n[3/3] Loading test data...")
    test_df = pl.read_parquet(test_path)
    print(f"  ✓ {len(test_df):,} samples")
    print(f"  Positive: {(test_df['label'] == 1).sum():,}")
    print(f"  Negative: {(test_df['label'] == 0).sum():,}")

    return train_df, val_df, test_df


def prepare_ranking_data(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for XGBoost ranking (group by source business) with interaction features.

    In ranking problems, we need to:
    1. Group candidates by query (src_gmap_id)
    2. Rank candidates within each query based on features
    3. Labels indicate which candidate is the actual next visit

    Args:
        df: DataFrame with features and labels

    Returns:
        Tuple of (X, y, group_sizes, feature_names)
        - X: Feature matrix (n_samples, n_features)
        - y: Binary labels (1 = actual visit, 0 = negative sample)
        - group_sizes: Number of candidates per query (for ranking)
        - feature_names: List of feature column names
    """
    print("\n[Preparing ranking data with interaction features...]")

    # Sort by source business to group queries together
    df = df.sort(['src_gmap_id', 'src_ts'])
    
    # DIAGNOSTIC: Check columns before creating interaction features
    print(f"\n  🔍 DIAGNOSTIC: Columns before create_interaction_features: {len(df.columns)}")
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # DIAGNOSTIC: Check columns after creating interaction features
    print(f"  🔍 DIAGNOSTIC: Columns after create_interaction_features: {len(df.columns)}")
    interaction_cols = [col for col in df.columns 
                       if any(s in col for s in ['_x_', '_squared', '_ratio', '_proximity', '_match_score'])]
    print(f"  🔍 DIAGNOSTIC: Found {len(interaction_cols)} interaction columns")
    if len(interaction_cols) > 0:
        print(f"  🔍 DIAGNOSTIC: Examples: {interaction_cols[:5]}")
        # Check variance to ensure features have meaningful values
        for col in interaction_cols[:5]:
            col_mean = df[col].mean()
            col_std = df[col].std()
            col_min = df[col].min()
            col_max = df[col].max()
            print(f"      {col}: mean={col_mean:.4f}, std={col_std:.4f}, range=[{col_min:.4f}, {col_max:.4f}]")
    else:
        print(f"  ⚠️  WARNING: No interaction features created!")
        print(f"      Available columns (first 20): {df.columns[:20]}")

    # One-hot encode categorical features
    print("  - One-hot encoding categorical features...")
    df_encoded = df.clone()

    for cat_feature in CATEGORICAL_FEATURES:
        # Get unique values and create dummy columns
        dummies = df_encoded.select(pl.col(cat_feature)).to_dummies()
        df_encoded = pl.concat([df_encoded, dummies], how="horizontal")

    # Get all feature column names (base + interaction + one-hot encoded)
    # Interaction features have specific suffixes
    interaction_suffixes = ['_x_', '_squared', '_ratio', '_proximity', '_match_score']
    interaction_features = [col for col in df_encoded.columns 
                           if any(suffix in col for suffix in interaction_suffixes)]
    
    feature_cols = ALL_BASE_FEATURES + interaction_features + [col for col in df_encoded.columns
                                        if any(col.startswith(cat + '_') for cat in CATEGORICAL_FEATURES)]

    print(f"  - Total features after encoding: {len(feature_cols)}")
    print(f"    Base features: {len(ALL_BASE_FEATURES)}")
    print(f"    Interaction features: {len(interaction_features)}")
    print(f"    One-hot encoded: {len(feature_cols) - len(ALL_BASE_FEATURES) - len(interaction_features)}")

    # Extract features
    X = df_encoded.select(feature_cols).to_numpy().astype(np.float32)
    y = df_encoded['label'].to_numpy()

    # Calculate group sizes (number of candidates per source business visit)
    groups = df.group_by(['src_gmap_id', 'src_ts']).agg(pl.len().alias('group_size'))
    group_sizes = groups['group_size'].to_numpy()

    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Number of queries (source visits): {len(group_sizes):,}")
    print(f"  Avg candidates per query: {np.mean(group_sizes):.1f}")
    print(f"  Total samples: {np.sum(group_sizes):,}")

    return X, y, group_sizes, feature_cols


def train_xgboost_ranker(X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, group_val: np.ndarray,
                         params: Dict = None) -> xgb.Booster:
    """
    Train XGBoost ranking model with enhanced parameters for better feature learning.

    Args:
        X_train, y_train, group_train: Training data
        X_val, y_val, group_val: Validation data
        params: XGBoost parameters (optional)

    Returns:
        Trained XGBoost model
    """
    print("\n" + "=" * 80)
    print("TRAINING ENHANCED XGBOOST RANKER")
    print("=" * 80)

    # Default parameters for ranking with improved feature interaction capture
    if params is None:
        # Auto-detect GPU availability
        try:
            # Try to detect CUDA support
            test_param = {'tree_method': 'hist', 'device': 'cuda'}
            xgb.train(test_param, xgb.DMatrix(np.zeros((10, 5)), label=np.zeros(10)), num_boost_round=1)
            device = 'cuda'
            print("  🚀 GPU detected! Using CUDA for training")
        except:
            device = 'cpu'
            print("  💻 Using CPU for training")

        params = {
            'objective': 'rank:pairwise',  # Pairwise ranking objective
            'eval_metric': ['ndcg@10', 'map@10'],  # Ranking metrics
            'eta': 0.05,                   # Lower learning rate for better feature learning
            'max_depth': 8,                # Deeper trees to capture interactions
            'min_child_weight': 3,         # Slightly higher to prevent overfitting
            'subsample': 0.7,              # More aggressive row sampling
            'colsample_bytree': 0.7,       # More aggressive column sampling per tree
            'colsample_bylevel': 0.7,      # Column sampling per level (NEW)
            'colsample_bynode': 0.7,       # Column sampling per split (NEW)
            'gamma': 0.1,                  # Minimum loss reduction (NEW - encourages pruning)
            'lambda': 2.0,                 # Increased L2 regularization
            'alpha': 0.5,                  # Increased L1 regularization
            'scale_pos_weight': 3.0,       # Weight positive samples more (NEW)
            'seed': 42,
            'tree_method': 'hist',         # Fast histogram-based algorithm
            'device': device,              # Auto-detected: 'cuda' or 'cpu'
            'interaction_constraints': None,  # Allow all feature interactions
            'max_leaves': 0                # No limit on leaves (use max_depth instead)
        }

    print("\nModel parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Create DMatrix for ranking
    print("\n[1/3] Creating DMatrix objects...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(group_train)

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(group_val)

    print(f"  ✓ Training DMatrix: {X_train.shape}")
    print(f"  ✓ Validation DMatrix: {X_val.shape}")

    # Train model
    print("\n[2/3] Training XGBoost model...")
    print("  (This may take several minutes...)")

    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,           # Max number of trees
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=20,      # Stop if no improvement for 20 rounds
        verbose_eval=50                # Print every 50 rounds
    )

    # Print best iteration
    print(f"\n[3/3] Training complete!")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best NDCG@10: {evals_result['val']['ndcg@10'][model.best_iteration]:.4f}")
    print(f"  Best MAP@10: {evals_result['val']['map@10'][model.best_iteration]:.4f}")

    return model


def predict_ranking(model: xgb.Booster, df: pl.DataFrame, top_k: int = 10) -> pl.DataFrame:
    """
    Generate ranking predictions for each source business visit.

    For each source visit, rank all candidate destinations and return top-K.

    Args:
        model: Trained XGBoost model
        df: Test DataFrame
        top_k: Number of top predictions to return per query

    Returns:
        DataFrame with top-K predictions per source visit
    """
    print("\n" + "=" * 80)
    print(f"GENERATING TOP-{top_k} PREDICTIONS")
    print("=" * 80)

    # Sort by source business
    df = df.sort(['src_gmap_id', 'src_ts'])
    
    # Create interaction features
    print("\nCreating interaction features...")
    df = create_interaction_features(df)

    # One-hot encode categorical features (same as training)
    print("Encoding features and predicting...")
    df_encoded = df.clone()

    for cat_feature in CATEGORICAL_FEATURES:
        dummies = df_encoded.select(pl.col(cat_feature)).to_dummies()
        df_encoded = pl.concat([df_encoded, dummies], how="horizontal")

    # Get all feature column names (same as prepare_ranking_data)
    interaction_suffixes = ['_x_', '_squared', '_ratio', '_proximity', '_match_score']
    interaction_features = [col for col in df_encoded.columns 
                           if any(suffix in col for suffix in interaction_suffixes)]
    
    feature_cols = ALL_BASE_FEATURES + interaction_features + [col for col in df_encoded.columns
                                        if any(col.startswith(cat + '_') for cat in CATEGORICAL_FEATURES)]

    # Get features and predict
    X = df_encoded.select(feature_cols).to_numpy().astype(np.float32)
    dtest = xgb.DMatrix(X)
    scores = model.predict(dtest)
    df = df.with_columns(pl.Series("score", scores))

    # Create query identifier and rank
    print(f"Ranking top-{top_k} per query...")
    df = df.with_columns([
        (pl.col('src_gmap_id') + '_' + pl.col('src_ts').cast(pl.Utf8)).alias('query_id')
    ])

    # Rank candidates per query (higher score = rank 1)
    df = df.with_columns([
        pl.col('score')
        .rank(method='ordinal', descending=True)
        .over('query_id')
        .alias('rank')
    ])

    # Keep only top-K per query
    predictions_df = (
        df
        .filter(pl.col('rank') <= top_k)
        .with_columns(pl.col('rank').cast(pl.Int32))
        .sort(['query_id', 'rank'])
    )

    num_queries = predictions_df['query_id'].n_unique()
    print(f"✓ Generated {len(predictions_df):,} predictions for {num_queries:,} queries")

    return predictions_df


def evaluate_ranking(predictions_df: pl.DataFrame, k_values: List[int] = [1, 5, 10], 
                     compute_ndcg: bool = False, ndcg_sample_size: int = 10000) -> Dict:
    """
    Evaluate ranking predictions using Recall@K, MRR, and optionally nDCG@K.

    Metrics:
    - Recall@K: % of queries where actual next visit is in top-K
    - MRR (Mean Reciprocal Rank): Average of 1/rank for actual visits
    - nDCG@K: Normalized Discounted Cumulative Gain (quality of ranking) [OPTIONAL - memory intensive]

    Args:
        predictions_df: Predictions DataFrame with rank and label columns
        k_values: List of K values to evaluate
        compute_ndcg: Whether to compute nDCG (can be memory intensive, disabled by default)
        ndcg_sample_size: Sample this many queries for nDCG computation to save memory

    Returns:
        Dictionary with metric results
    """
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    metrics = {}
    
    total_queries = predictions_df['query_id'].n_unique()
    print(f"\nEvaluating {total_queries:,} queries...")
    
    # Diagnostic information (condensed)
    total_predictions = len(predictions_df)
    positive_predictions = (predictions_df['label'] == 1).sum()
    
    if positive_predictions == 0:
        print("  ⚠️  WARNING: No positive labels found! Check data preparation.")
        return {}
    
    queries_with_positives = predictions_df.filter(pl.col('label') == 1)['query_id'].n_unique()
    
    if queries_with_positives == total_queries:
        print("  ⚠️  WARNING: Every query has exactly one positive - this may indicate data leakage!")

    # Recall@K - vectorized, memory efficient
    print(f"\nCalculating Recall@K and MRR...")
    for k in k_values:
        # For each query, check if actual visit (label=1) is in top-K
        top_k_df = predictions_df.filter(pl.col('rank') <= k)
        queries_with_hit = top_k_df.filter(pl.col('label') == 1)['query_id'].unique()

        recall = len(queries_with_hit) / total_queries
        metrics[f'recall@{k}'] = recall
        print(f"  Recall@{k}: {recall:.4f}")

    # MRR (Mean Reciprocal Rank)
    actual_visits = predictions_df.filter(pl.col('label') == 1)
    if len(actual_visits) > 0:
        reciprocal_ranks = 1.0 / actual_visits['rank'].to_numpy()
        mrr = np.mean(reciprocal_ranks)
        metrics['mrr'] = mrr
        print(f"  MRR: {mrr:.4f}")
    else:
        metrics['mrr'] = 0.0
        print(f"  MRR: 0.0000")

    # nDCG@K - OPTIONAL (memory intensive, sampled for efficiency)
    if compute_ndcg:
        print(f"\nCalculating nDCG@K (sampled: {min(ndcg_sample_size, total_queries):,} queries)...")
        
        # Sample queries to reduce memory usage
        all_query_ids = predictions_df['query_id'].unique().to_list()
        if len(all_query_ids) > ndcg_sample_size:
            import random
            random.seed(42)
            sampled_query_ids = random.sample(all_query_ids, ndcg_sample_size)
            sample_df = predictions_df.filter(pl.col('query_id').is_in(sampled_query_ids))
        else:
            sample_df = predictions_df
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        partitions = sample_df.sort(['query_id', 'rank']).partition_by('query_id', maintain_order=True)
        
        for k in k_values:
            ndcg_scores = []
            batch_scores = []
            
            for i, group_df in enumerate(partitions):
                top_k_df = group_df.filter(pl.col('rank') <= k)
                if top_k_df.is_empty():
                    continue

                relevance = top_k_df['label'].to_numpy()
                positions = np.arange(1, len(relevance) + 1)

                dcg = np.sum(relevance / np.log2(positions + 1))
                ideal_relevance = np.sort(relevance)[::-1]
                idcg = np.sum(ideal_relevance / np.log2(np.arange(1, len(ideal_relevance) + 1) + 1))

                ndcg = dcg / idcg if idcg > 0 else 0.0
                batch_scores.append(ndcg)
                
                # Process in batches to avoid memory accumulation
                if (i + 1) % batch_size == 0:
                    ndcg_scores.extend(batch_scores)
                    batch_scores = []
            
            # Add remaining scores
            if batch_scores:
                ndcg_scores.extend(batch_scores)
            
            ndcg_mean = np.mean(ndcg_scores) if ndcg_scores else 0.0
            metrics[f'ndcg@{k}'] = ndcg_mean
            print(f"  nDCG@{k}: {ndcg_mean:.4f}")
    else:
        print(f"\n⏭️  Skipping nDCG@K (set compute_ndcg=True to enable)")

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


def export_predictions_for_visualization(predictions_df: pl.DataFrame, biz_df: pl.DataFrame, 
                                         output_path: Path, top_k: int = 5) -> None:
    """
    Export predictions in format compatible with visualization API (similar to LSTM format).
    
    Creates a CSV with columns similar to LSTM visualization format:
    - source_business_idx, source_gmap_id, source_name, source_lat, source_lon, etc.
    - prediction_id, rank, score_raw, score_share, score_cum_share, score_z
    - dest_business_idx, dest_gmap_id, dest_name, dest_lat, dest_lon, etc.
    - link_distance_km, same_main_category
    
    Args:
        predictions_df: Predictions DataFrame with scores and ranks
        biz_df: Business metadata DataFrame
        output_path: Path to save CSV file
        top_k: Number of top predictions per source (default: 5)
    """
    print("\n" + "=" * 80)
    print(f"EXPORTING TOP-{top_k} PREDICTIONS FOR VISUALIZATION")
    print("=" * 80)
    
    # Filter to top-K predictions per source
    print(f"\n[1/4] Filtering to top-{top_k} predictions per source...")
    viz_df = predictions_df.filter(pl.col('rank') <= top_k)
    
    # Create business index mapping
    print(f"[2/4] Creating business index mapping...")
    biz_indexed = biz_df.with_columns([
        pl.col('gmap_id').rank('dense').alias('business_idx')
    ])
    
    # Join source business data
    print(f"[3/4] Joining source business metadata...")
    # Convert category_all list to string for CSV compatibility
    biz_for_join = biz_indexed.with_columns([
        pl.when(pl.col('category_all').is_not_null())
        .then(pl.col('category_all').list.join(", "))
        .otherwise(pl.lit(None))
        .alias('category_all_str')
    ])
    
    viz_df = viz_df.join(
        biz_for_join.select([
            pl.col('gmap_id').alias('src_gmap_id'),
            pl.col('business_idx').alias('source_business_idx'),
            pl.col('name').alias('source_name'),
            pl.col('lat').alias('source_lat'),
            pl.col('lon').alias('source_lon'),
            pl.col('category_main').alias('source_category_main'),
            pl.col('category_all_str').alias('source_category_all'),
            pl.col('avg_rating').alias('source_avg_rating'),
            pl.col('num_reviews').alias('source_num_reviews'),
            pl.col('price_bucket').alias('source_price_bucket'),
            pl.col('is_closed').alias('source_is_closed')
        ]),
        on='src_gmap_id',
        how='left'
    )
    
    # Join destination business data
    print(f"[4/4] Joining destination business metadata...")
    viz_df = viz_df.join(
        biz_for_join.select([
            pl.col('gmap_id').alias('dst_gmap_id'),
            pl.col('business_idx').alias('dest_business_idx'),
            pl.col('name').alias('dest_name'),
            pl.col('lat').alias('dest_lat'),
            pl.col('lon').alias('dest_lon'),
            pl.col('category_main').alias('dest_category_main'),
            pl.col('category_all_str').alias('dest_category_all'),
            pl.col('avg_rating').alias('dest_avg_rating'),
            pl.col('num_reviews').alias('dest_num_reviews'),
            pl.col('price_bucket').alias('dest_price_bucket'),
            pl.col('is_closed').alias('dest_is_closed')
        ]),
        on='dst_gmap_id',
        how='left'
    )
    
    # Calculate normalized scores per source (for score_share, score_cum_share, score_z)
    print(f"\nCalculating normalized prediction scores...")
    viz_df = viz_df.with_columns([
        pl.col('score').alias('score_raw'),
        # Score share: proportion of total score for this source
        (pl.col('score') / pl.col('score').sum().over('src_gmap_id')).alias('score_share'),
        # Score Z-score: standardized score within source
        ((pl.col('score') - pl.col('score').mean().over('src_gmap_id')) / 
         pl.col('score').std().over('src_gmap_id')).alias('score_z')
    ])
    
    # Cumulative share
    viz_df = viz_df.sort(['src_gmap_id', 'rank']).with_columns([
        pl.col('score_share').cum_sum().over('src_gmap_id').alias('score_cum_share')
    ])
    
    # Create prediction_id
    viz_df = viz_df.with_columns([
        pl.arange(0, pl.count()).alias('prediction_id')
    ])
    
    # Add link_distance_km (use existing distance_km if available)
    if 'distance_km' in viz_df.columns:
        viz_df = viz_df.rename({'distance_km': 'link_distance_km'})
    else:
        viz_df = viz_df.with_columns([
            pl.lit(None).alias('link_distance_km')
        ])
    
    # Select final columns in visualization format
    final_columns = [
        # Prediction info
        'prediction_id', 'rank', 'score_raw', 'score_share', 'score_cum_share', 'score_z',
        
        # Source business
        'source_business_idx', 'src_gmap_id', 'source_name', 
        'source_lat', 'source_lon', 'source_category_main', 'source_category_all',
        'source_avg_rating', 'source_num_reviews', 'source_price_bucket', 'source_is_closed',
        
        # Destination business  
        'dest_business_idx', 'dst_gmap_id', 'dest_name',
        'dest_lat', 'dest_lon', 'dest_category_main', 'dest_category_all',
        'dest_avg_rating', 'dest_num_reviews', 'dest_price_bucket', 'dest_is_closed',
        
        # Link properties
        'link_distance_km', 'same_category'
    ]
    
    # Keep only columns that exist
    existing_columns = [col for col in final_columns if col in viz_df.columns]
    
    # Handle same_category (rename if needed)
    if 'same_main_category' not in viz_df.columns and 'same_category' in viz_df.columns:
        viz_df = viz_df.rename({'same_category': 'same_main_category'})
        existing_columns = [col if col != 'same_category' else 'same_main_category' for col in existing_columns]
    
    viz_df = viz_df.select(existing_columns)
    
    # Save to CSV
    print(f"\nSaving predictions...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure all columns are CSV-compatible (no nested types)
    # Convert any remaining list/struct columns to strings
    for col in viz_df.columns:
        dtype = viz_df[col].dtype
        if dtype == pl.List or str(dtype).startswith('List'):
            viz_df = viz_df.with_columns([
                pl.when(pl.col(col).is_not_null())
                .then(pl.col(col).list.join(", "))
                .otherwise(pl.lit(None))
                .alias(col)
            ])
        elif dtype == pl.Struct or str(dtype).startswith('Struct'):
            viz_df = viz_df.with_columns([
                pl.col(col).cast(pl.Utf8).alias(col)
            ])
    
    viz_df.write_csv(output_path)
    
    print(f"✓ Exported {len(viz_df):,} predictions")
    print(f"  Unique sources: {viz_df['source_business_idx'].n_unique():,}")
    print(f"  File: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Print sample
    print(f"\nSample predictions (first source):")
    sample = viz_df.head(min(5, len(viz_df)))
    print(sample.select([
        'source_name', 'rank', 'dest_name', 'score_raw', 'score_share', 'link_distance_km'
    ]))


def analyze_feature_importance(model: xgb.Booster, feature_names: List[str]) -> Dict:
    """
    Extract and analyze feature importance for interpretability.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        Dictionary with importance scores by feature group
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Extract importance scores (XGBoost uses f0, f1, f2... internally)
    xgb_importance_scores = model.get_score(importance_type='gain')
    
    # Map XGBoost's internal feature names (f0, f1, f2...) back to our actual names
    importance_scores = {}
    for xgb_feature_name, score in xgb_importance_scores.items():
        # Check if it's an internal name (f0, f1, f2...)
        if xgb_feature_name.startswith('f') and xgb_feature_name[1:].isdigit():
            feature_idx = int(xgb_feature_name[1:])
            if feature_idx < len(feature_names):
                actual_feature_name = feature_names[feature_idx]
                importance_scores[actual_feature_name] = score
            else:
                # Feature index out of range - keep original name
                importance_scores[xgb_feature_name] = score
        else:
            # Already using actual names
            importance_scores[xgb_feature_name] = score
    
    # Group features by category
    importance_by_group = {
        'spatial': [],
        'temporal': [],
        'quality': [],
        'price': [],
        'category': [],
        'relationship': [],
        'review_sentiment': [],
        'user_behavior': [],
        'cuisine_complementarity': [],
        'operating_hours': [],
        'topics': [],
        'service_options': []
    }
    
    # Categorize features (including interaction features)
    for feature, score in importance_scores.items():
        # Check if it's an interaction feature first
        if '_x_' in feature or '_squared' in feature or '_ratio' in feature or '_proximity' in feature or '_match_score' in feature:
            # Categorize interaction features based on their components
            if 'distance' in feature and 'rating' in feature:
                importance_by_group['quality'].append((feature, score))  # Distance × Rating
            elif 'distance' in feature and 'time' in feature:
                importance_by_group['spatial'].append((feature, score))  # Time × Distance
            elif 'price' in feature and 'rating' in feature:
                importance_by_group['price'].append((feature, score))  # Price × Rating
            elif 'user' in feature:
                importance_by_group['user_behavior'].append((feature, score))  # User interactions
            elif 'sentiment' in feature:
                importance_by_group['review_sentiment'].append((feature, score))  # Sentiment interactions
            elif 'service' in feature:
                importance_by_group['service_options'].append((feature, score))  # Service interactions
            elif 'relative_results' in feature:
                importance_by_group['relationship'].append((feature, score))  # Relationship interactions
            else:
                importance_by_group['spatial'].append((feature, score))  # Default for other interactions
        # Regular features
        elif any(keyword in feature for keyword in ['distance', 'neighborhood', 'direction']):
            importance_by_group['spatial'].append((feature, score))
        elif any(keyword in feature for keyword in ['hour', 'day_of_week', 'weekend', 'delta', 'meal']):
            importance_by_group['temporal'].append((feature, score))
        elif any(keyword in feature for keyword in ['rating', 'highly_rated']):
            importance_by_group['quality'].append((feature, score))
        elif any(keyword in feature for keyword in ['price', 'upgrade', 'downgrade']):
            importance_by_group['price'].append((feature, score))
        elif any(keyword in feature for keyword in ['category', 'same_category', 'cuisine', 'dessert', 'meal_progression']):
            importance_by_group['category'].append((feature, score))
        elif any(keyword in feature for keyword in ['relative_results', 'is_in_relative']):
            importance_by_group['relationship'].append((feature, score))
        elif any(keyword in feature for keyword in ['sentiment', 'review_length', 'positive_review']):
            importance_by_group['review_sentiment'].append((feature, score))
        elif any(keyword in feature for keyword in ['user_', 'explorer', 'frequent', 'diverse', 'standards', 'consistent']):
            importance_by_group['user_behavior'].append((feature, score))
        elif any(keyword in feature for keyword in ['cuisine_pair', 'complementarity', 'followup', 'progression']):
            importance_by_group['cuisine_complementarity'].append((feature, score))
        elif any(keyword in feature for keyword in ['open_at_time', 'hours_overlap', 'late_night', 'early_morning', '24hr']):
            importance_by_group['operating_hours'].append((feature, score))
        elif any(keyword in feature for keyword in ['mentions_', 'topic_']):
            importance_by_group['topics'].append((feature, score))
        elif any(keyword in feature for keyword in ['delivery', 'takeout', 'dinein', 'reservations', 'service']):
            importance_by_group['service_options'].append((feature, score))
        else:
            # Default to spatial if no clear category
            importance_by_group['spatial'].append((feature, score))
    
    # Sort features within each group by importance
    for group in importance_by_group:
        importance_by_group[group].sort(key=lambda x: x[1], reverse=True)
    
    # Calculate group totals
    group_totals = {}
    for group, features in importance_by_group.items():
        group_totals[group] = sum(score for _, score in features)
    
    # Get top features overall
    all_features = [(feature, score) for group_features in importance_by_group.values() 
                   for feature, score in group_features]
    all_features.sort(key=lambda x: x[1], reverse=True)
    top_features = all_features[:20]
    
    print(f"\nTop 20 Most Important Features:")
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feature:40s} {score:8.4f}")
    
    print(f"\nImportance by Feature Group:")
    sorted_groups = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)
    for group, total_score in sorted_groups:
        feature_count = len(importance_by_group[group])
        avg_score = total_score / max(feature_count, 1)
        print(f"  {group:20s}: {total_score:8.4f} total, {avg_score:8.4f} avg ({feature_count:2d} features)")
    
    # Prepare results for export
    results = {
        'top_features': [{'feature': f, 'importance': s} for f, s in top_features],
        'group_totals': group_totals,
        'group_details': {group: [{'feature': f, 'importance': s} for f, s in features] 
                         for group, features in importance_by_group.items()},
        'total_features': len(all_features),
        'feature_coverage': len([f for f in feature_names if f in importance_scores]) / len(feature_names)
    }
    
    return results


# Wrapper functions for notebook convenience
def prepare_features(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Wrapper for prepare_ranking_data with a more intuitive name.
    """
    return prepare_ranking_data(df)


def train_model(X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, group_val: np.ndarray) -> xgb.Booster:
    """
    Wrapper for train_xgboost_ranker with a more intuitive name.
    """
    return train_xgboost_ranker(X_train, y_train, group_train, X_val, y_val, group_val)


def save_model_and_results(model: xgb.Booster, predictions_df: pl.DataFrame,
                           metrics: Dict, feature_importance: Dict, output_dir: Path):
    """
    Save trained model, predictions, and metrics.

    Args:
        model: Trained XGBoost model
        predictions_df: Predictions DataFrame
        metrics: Evaluation metrics dictionary
        feature_importance: Feature importance analysis results
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL AND RESULTS")
    print("=" * 80)

    # Create output directories
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "models" / "predictions"
    metrics_dir = output_dir / "models" / "metrics"

    models_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    # Save model
    print("\n[1/3] Saving model...")
    model_path = models_dir / "xgboost_ranker.json"
    model.save_model(str(model_path))
    print(f"  ✓ {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

    # Save predictions
    print("\n[2/3] Saving predictions...")
    pred_path = predictions_dir / "xgboost_predictions.parquet"
    predictions_df.write_parquet(pred_path, compression="snappy")
    print(f"  ✓ {pred_path}")
    print(f"  Size: {pred_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Rows: {len(predictions_df):,}")

    # Save metrics
    print("\n[3/4] Saving metrics...")
    metrics_path = metrics_dir / "xgboost_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {metrics_path}")
    
    # Save feature importance
    print("\n[4/4] Saving feature importance...")
    importance_path = metrics_dir / "feature_importance.json"
    with open(importance_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print(f"  ✓ {importance_path}")

    print("\n✓ All results saved!")


def main():
    """Main training pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "processed" / "ga" / "xgboost_data"
    output_dir = base_dir

    print("=" * 80)
    print("PHASE B1: XGBOOST RANKING MODEL")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir / 'models'}")

    # Load data
    train_df, val_df, test_df = load_data(data_dir)

    # Prepare ranking data
    print("\n" + "=" * 80)
    print("PREPARING RANKING DATA")
    print("=" * 80)

    print("\nTraining set:")
    X_train, y_train, group_train, feature_names = prepare_ranking_data(train_df)

    print("\nValidation set:")
    X_val, y_val, group_val, _ = prepare_ranking_data(val_df)

    print("\nTest set:")
    X_test, y_test, group_test, _ = prepare_ranking_data(test_df)

    # Train model
    model = train_xgboost_ranker(X_train, y_train, group_train,
                                  X_val, y_val, group_val)

    # Generate predictions on test set
    predictions_df = predict_ranking(model, test_df, top_k=10)

    # Evaluate
    metrics = evaluate_ranking(predictions_df, k_values=[1, 5, 10])
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, feature_names)

    # Save results
    save_model_and_results(model, predictions_df, metrics, feature_importance, output_dir)
    
    # Generate transition matrix
    print("\n" + "=" * 80)
    print("GENERATING TRANSITION MATRIX")
    print("=" * 80)
    
    try:
        from transition_matrix import build_transition_matrix, export_transition_matrix
        
        # Load reviews data for enhanced features
        reviews_input = data_dir / "reviews_ga.parquet"
        reviews_df = pl.read_parquet(reviews_input)
        
        # Build transition matrix
        transition_matrix_df = build_transition_matrix(
            model, biz_df, feature_names, reviews_df, top_k=10, max_candidates=100
        )
        
        # Export transition matrix
        export_transition_matrix(transition_matrix_df, output_dir)
        
        print("  ✓ Transition matrix generated successfully")
        
    except Exception as e:
        print(f"  ⚠ Failed to generate transition matrix: {e}")
        print("  This is optional and doesn't affect model training")

    print("\n✓✓✓ PHASE B1 COMPLETE ✓✓✓\n")
    print("Next steps:")
    print("  1. Review metrics in models/metrics/xgboost_metrics.json")
    print("  2. Examine predictions in models/predictions/xgboost_predictions.parquet")
    print("  3. Use model for inference: models/xgboost_ranker.json")
    print("  4. Explore transition matrix in transition_matrix/")
    print("  5. Review feature importance in models/metrics/feature_importance.json")


if __name__ == "__main__":
    main()
