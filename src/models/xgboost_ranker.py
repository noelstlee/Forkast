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
    Prepare data for XGBoost ranking (group by source business).

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
    print("\n[Preparing ranking data...]")

    # Sort by source business to group queries together
    df = df.sort(['src_gmap_id', 'src_ts'])

    # One-hot encode categorical features
    print("  - One-hot encoding categorical features...")
    df_encoded = df.clone()

    for cat_feature in CATEGORICAL_FEATURES:
        # Get unique values and create dummy columns
        dummies = df_encoded.select(pl.col(cat_feature)).to_dummies()
        df_encoded = pl.concat([df_encoded, dummies], how="horizontal")

    # Get all feature column names (base + one-hot encoded)
    feature_cols = ALL_BASE_FEATURES + [col for col in df_encoded.columns
                                        if any(col.startswith(cat + '_') for cat in CATEGORICAL_FEATURES)]

    print(f"  - Total features after encoding: {len(feature_cols)}")

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
    Train XGBoost ranking model.

    Args:
        X_train, y_train, group_train: Training data
        X_val, y_val, group_val: Validation data
        params: XGBoost parameters (optional)

    Returns:
        Trained XGBoost model
    """
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST RANKER")
    print("=" * 80)

    # Default parameters for ranking
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
            'eta': 0.1,                    # Learning rate
            'max_depth': 6,                # Tree depth
            'min_child_weight': 1,
            'subsample': 0.8,              # Row sampling
            'colsample_bytree': 0.8,       # Column sampling
            'lambda': 1.0,                 # L2 regularization
            'alpha': 0.1,                  # L1 regularization
            'seed': 42,
            'tree_method': 'hist',         # Fast histogram-based algorithm
            'device': device               # Auto-detected: 'cuda' or 'cpu'
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

    # One-hot encode categorical features (same as training)
    print("\nEncoding features and predicting...")
    df_encoded = df.clone()

    for cat_feature in CATEGORICAL_FEATURES:
        dummies = df_encoded.select(pl.col(cat_feature)).to_dummies()
        df_encoded = pl.concat([df_encoded, dummies], how="horizontal")

    # Get all feature column names
    feature_cols = ALL_BASE_FEATURES + [col for col in df_encoded.columns
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
    
    # Extract importance scores
    importance_scores = model.get_score(importance_type='gain')
    
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
    
    # Categorize features
    for feature, score in importance_scores.items():
        if any(keyword in feature for keyword in ['distance', 'neighborhood', 'direction']):
            importance_by_group['spatial'].append((feature, score))
        elif any(keyword in feature for keyword in ['hour', 'day_of_week', 'weekend', 'delta', 'meal']):
            importance_by_group['temporal'].append((feature, score))
        elif any(keyword in feature for keyword in ['rating', 'highly_rated', 'sentiment']):
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
