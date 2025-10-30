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
    'dst_meal_type'
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
    'relative_results_rank'
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
    'is_in_relative_results'
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


def prepare_ranking_data(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for XGBoost ranking (group by source business).

    In ranking problems, we need to:
    1. Group candidates by query (src_gmap_id)
    2. Rank candidates within each query based on features
    3. Labels indicate which candidate is the actual next visit

    Args:
        df: DataFrame with features and labels

    Returns:
        Tuple of (X, y, group_sizes)
        - X: Feature matrix (n_samples, n_features)
        - y: Binary labels (1 = actual visit, 0 = negative sample)
        - group_sizes: Number of candidates per query (for ranking)
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

    return X, y, group_sizes


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
    print("\n[1/4] One-hot encoding categorical features...")
    df_encoded = df.clone()

    for cat_feature in CATEGORICAL_FEATURES:
        dummies = df_encoded.select(pl.col(cat_feature)).to_dummies()
        df_encoded = pl.concat([df_encoded, dummies], how="horizontal")

    # Get all feature column names
    feature_cols = ALL_BASE_FEATURES + [col for col in df_encoded.columns
                                        if any(col.startswith(cat + '_') for cat in CATEGORICAL_FEATURES)]

    # Get features
    X = df_encoded.select(feature_cols).to_numpy().astype(np.float32)

    # Predict scores
    print("\n[2/4] Predicting scores...")
    dtest = xgb.DMatrix(X)
    scores = model.predict(dtest)

    # Add scores to dataframe
    df = df.with_columns(pl.Series("score", scores))

    # Group by source visit and get top-K per group
    print(f"\n[3/4] Selecting top-{top_k} per source visit...")

    # Create a query identifier
    df = df.with_columns([
        (pl.col('src_gmap_id') + '_' + pl.col('src_ts').cast(pl.Utf8)).alias('query_id')
    ])

    # Get top-K per query
    predictions = []

    for query_id in df['query_id'].unique():
        query_candidates = df.filter(pl.col('query_id') == query_id)
        top_k_candidates = query_candidates.sort('score', descending=True).head(top_k)
        predictions.append(top_k_candidates)

    predictions_df = pl.concat(predictions)

    print(f"  ✓ Generated {len(predictions_df):,} predictions")
    print(f"  ✓ For {len(df['query_id'].unique()):,} queries")

    # Add rank column
    print(f"\n[4/4] Adding rank column...")
    predictions_df = predictions_df.with_columns([
        pl.col('score').rank(method='ordinal', descending=True).over('query_id').alias('rank')
    ])

    return predictions_df


def evaluate_ranking(predictions_df: pl.DataFrame, k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Evaluate ranking predictions using Recall@K, MRR, and nDCG@K.

    Metrics:
    - Recall@K: % of queries where actual next visit is in top-K
    - MRR (Mean Reciprocal Rank): Average of 1/rank for actual visits
    - nDCG@K: Normalized Discounted Cumulative Gain (quality of ranking)

    Args:
        predictions_df: Predictions DataFrame with rank and label columns
        k_values: List of K values to evaluate

    Returns:
        Dictionary with metric results
    """
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    metrics = {}

    # Group by query
    queries = predictions_df.group_by('query_id')

    print(f"\nEvaluating {len(predictions_df['query_id'].unique()):,} queries...")

    # Recall@K
    print(f"\n[1/3] Calculating Recall@K...")
    for k in k_values:
        # For each query, check if actual visit (label=1) is in top-K
        top_k_df = predictions_df.filter(pl.col('rank') <= k)
        queries_with_hit = top_k_df.filter(pl.col('label') == 1)['query_id'].unique()
        total_queries = predictions_df['query_id'].n_unique()

        recall = len(queries_with_hit) / total_queries
        metrics[f'recall@{k}'] = recall
        print(f"  Recall@{k}: {recall:.4f} ({len(queries_with_hit):,}/{total_queries:,} queries)")

    # MRR (Mean Reciprocal Rank)
    print(f"\n[2/3] Calculating MRR...")
    actual_visits = predictions_df.filter(pl.col('label') == 1)
    reciprocal_ranks = 1.0 / actual_visits['rank'].to_numpy()
    mrr = np.mean(reciprocal_ranks)
    metrics['mrr'] = mrr
    print(f"  MRR: {mrr:.4f}")

    # nDCG@K
    print(f"\n[3/3] Calculating nDCG@K...")
    for k in k_values:
        dcg_scores = []

        for query_id in predictions_df['query_id'].unique():
            query_preds = predictions_df.filter(pl.col('query_id') == query_id).sort('rank')
            top_k = query_preds.head(k)

            # DCG = sum(relevance / log2(rank + 1))
            relevance = top_k['label'].to_numpy()
            ranks = top_k['rank'].to_numpy()
            dcg = np.sum(relevance / np.log2(ranks + 1))

            # IDCG (ideal DCG - if we had perfect ranking)
            ideal_relevance = np.sort(relevance)[::-1]
            idcg = np.sum(ideal_relevance / np.log2(np.arange(1, len(ideal_relevance) + 1) + 1))

            # nDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            dcg_scores.append(ndcg)

        ndcg_mean = np.mean(dcg_scores)
        metrics[f'ndcg@{k}'] = ndcg_mean
        print(f"  nDCG@{k}: {ndcg_mean:.4f}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


def save_model_and_results(model: xgb.Booster, predictions_df: pl.DataFrame,
                           metrics: Dict, output_dir: Path):
    """
    Save trained model, predictions, and metrics.

    Args:
        model: Trained XGBoost model
        predictions_df: Predictions DataFrame
        metrics: Evaluation metrics dictionary
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
    print("\n[3/3] Saving metrics...")
    metrics_path = metrics_dir / "xgboost_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {metrics_path}")

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
    X_train, y_train, group_train = prepare_ranking_data(train_df)

    print("\nValidation set:")
    X_val, y_val, group_val = prepare_ranking_data(val_df)

    print("\nTest set:")
    X_test, y_test, group_test = prepare_ranking_data(test_df)

    # Train model
    model = train_xgboost_ranker(X_train, y_train, group_train,
                                  X_val, y_val, group_val)

    # Generate predictions on test set
    predictions_df = predict_ranking(model, test_df, top_k=10)

    # Evaluate
    metrics = evaluate_ranking(predictions_df, k_values=[1, 5, 10])

    # Save results
    save_model_and_results(model, predictions_df, metrics, output_dir)

    print("\n✓✓✓ PHASE B1 COMPLETE ✓✓✓\n")
    print("Next steps:")
    print("  1. Review metrics in models/metrics/xgboost_metrics.json")
    print("  2. Examine predictions in models/predictions/xgboost_predictions.parquet")
    print("  3. Use model for inference: models/xgboost_ranker.json")


if __name__ == "__main__":
    main()
