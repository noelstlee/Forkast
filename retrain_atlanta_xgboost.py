"""
Retrain XGBoost model on Atlanta-only data.

This script uses the same architecture as the original XGBoost model
but trains exclusively on Atlanta restaurant pairs.
"""

import polars as pl
import xgboost as xgb
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import ndcg_score

# Feature definitions (same as original)
CATEGORICAL_FEATURES = ['distance_bucket', 'direction', 'delta_hours_bucket', 'dst_meal_type']

NUMERICAL_FEATURES = [
    'distance_km', 'delta_hours', 'src_rating', 'dst_rating',
    'src_hour', 'dst_hour', 'src_day_of_week', 'dst_day_of_week',
    'rating_diff', 'src_price', 'dst_price', 'price_diff', 'relative_results_rank'
]

BOOLEAN_FEATURES = [
    'same_neighborhood', 'same_category',
    'src_is_weekend', 'dst_is_weekend',
    'src_is_breakfast', 'dst_is_breakfast',
    'src_is_lunch', 'dst_is_lunch',
    'src_is_dinner', 'dst_is_dinner',
    'is_rating_upgrade', 'is_rating_downgrade',
    'src_is_highly_rated', 'dst_is_highly_rated',
    'is_price_upgrade', 'is_price_downgrade',
    'same_price_level', 'is_in_relative_results'
]

def prepare_features(df: pl.DataFrame, is_training: bool = False):
    """Prepare features for XGBoost training."""
    print("   Preparing features...")

    # One-hot encode categorical features
    df_encoded = df.clone()

    for cat_feature in CATEGORICAL_FEATURES:
        if cat_feature in df_encoded.columns:
            # Get unique values for this feature
            dummies = df_encoded.select(pl.col(cat_feature)).to_dummies()
            df_encoded = pl.concat([df_encoded, dummies], how="horizontal")

    # Collect all feature columns
    feature_cols = []

    # Add numerical features
    for feat in NUMERICAL_FEATURES:
        if feat in df_encoded.columns:
            feature_cols.append(feat)

    # Add boolean features
    for feat in BOOLEAN_FEATURES:
        if feat in df_encoded.columns:
            feature_cols.append(feat)

    # Add one-hot encoded features
    for cat_feat in CATEGORICAL_FEATURES:
        if cat_feat in df.columns:
            one_hot_cols = [col for col in df_encoded.columns if col.startswith(f"{cat_feat}_")]
            feature_cols.extend(one_hot_cols)

    print(f"   ✓ Total features: {len(feature_cols)}")

    # Extract feature matrix
    X = df_encoded.select(feature_cols).to_numpy()

    # Fill NaN values with 0
    X = np.nan_to_num(X, nan=0.0)

    return X, feature_cols


def calculate_metrics(y_true, y_pred, group_sizes, k_values=[5, 10, 20]):
    """Calculate ranking metrics."""
    metrics = {}

    # Calculate Recall@K
    start_idx = 0
    for k in k_values:
        recalls = []
        for group_size in group_sizes:
            end_idx = start_idx + group_size

            true_labels = y_true[start_idx:end_idx]
            pred_scores = y_pred[start_idx:end_idx]

            # Get top-k predictions
            if len(pred_scores) > k:
                top_k_indices = np.argsort(pred_scores)[-k:]
            else:
                top_k_indices = np.arange(len(pred_scores))

            # Check if any true positive in top-k
            recall = 1.0 if any(true_labels[i] == 1 for i in top_k_indices) else 0.0
            recalls.append(recall)

            start_idx = end_idx

        metrics[f'Recall@{k}'] = np.mean(recalls)
        start_idx = 0

    # Calculate MRR
    start_idx = 0
    reciprocal_ranks = []
    for group_size in group_sizes:
        end_idx = start_idx + group_size

        true_labels = y_true[start_idx:end_idx]
        pred_scores = y_pred[start_idx:end_idx]

        # Get ranking
        ranked_indices = np.argsort(pred_scores)[::-1]

        # Find rank of first relevant item
        for rank, idx in enumerate(ranked_indices, 1):
            if true_labels[idx] == 1:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

        start_idx = end_idx

    metrics['MRR'] = np.mean(reciprocal_ranks)

    # Calculate nDCG@10
    start_idx = 0
    ndcg_scores = []
    for group_size in group_sizes:
        end_idx = start_idx + group_size

        true_labels = y_true[start_idx:end_idx].reshape(1, -1)
        pred_scores = y_pred[start_idx:end_idx].reshape(1, -1)

        # Skip groups with only 1 document (nDCG not meaningful)
        if group_size > 1:
            if group_size >= 10:
                ndcg = ndcg_score(true_labels, pred_scores, k=10)
            else:
                ndcg = ndcg_score(true_labels, pred_scores, k=group_size)
            ndcg_scores.append(ndcg)

        start_idx = end_idx

    metrics['nDCG@10'] = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return metrics


def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("RETRAINING XGBOOST ON ATLANTA-ONLY DATA")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for GPU
    print("\n[1/6] Checking for GPU support...")
    try:
        test_param = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(test_param, xgb.DMatrix(np.zeros((10, 5)), label=np.zeros(10)), num_boost_round=1)
        device = 'cuda'
        print("   ✓ GPU (CUDA) available - training will be fast!")
    except Exception:
        device = 'cpu'
        print("   ✓ Using CPU (no GPU detected)")

    # Load Atlanta-only data
    print("\n[2/6] Loading Atlanta-only splits...")
    data_dir = base_dir / "data/processed/atlanta/xgboost_data"

    train_df = pl.read_parquet(data_dir / "train.parquet")
    val_df = pl.read_parquet(data_dir / "val.parquet")
    test_df = pl.read_parquet(data_dir / "test.parquet")

    print(f"   ✓ Train: {len(train_df):,} samples")
    print(f"   ✓ Val: {len(val_df):,} samples")
    print(f"   ✓ Test: {len(test_df):,} samples")

    # Prepare features
    print("\n[3/6] Preparing features...")
    print("   Training set:")
    X_train, feature_names = prepare_features(train_df, is_training=True)
    y_train = train_df['label'].to_numpy()

    print("   Validation set:")
    X_val, _ = prepare_features(val_df, is_training=False)
    y_val = val_df['label'].to_numpy()

    print("   Test set:")
    X_test, _ = prepare_features(test_df, is_training=False)
    y_test = test_df['label'].to_numpy()

    # Create group sizes for ranking
    print("\n   Creating query groups...")
    train_groups = train_df.group_by(['user_id', 'src_gmap_id', 'src_ts']).agg(pl.count().alias('count'))['count'].to_numpy()
    val_groups = val_df.group_by(['user_id', 'src_gmap_id', 'src_ts']).agg(pl.count().alias('count'))['count'].to_numpy()
    test_groups = test_df.group_by(['user_id', 'src_gmap_id', 'src_ts']).agg(pl.count().alias('count'))['count'].to_numpy()

    print(f"   ✓ Train queries: {len(train_groups):,}")
    print(f"   ✓ Val queries: {len(val_groups):,}")
    print(f"   ✓ Test queries: {len(test_groups):,}")

    # Create DMatrix
    print("\n   Creating DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    dtrain.set_group(train_groups)
    dval.set_group(val_groups)
    dtest.set_group(test_groups)

    # Training parameters
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@10',
        'tree_method': 'hist',
        'device': device,
        'max_depth': 8,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'seed': 42
    }

    print("\n[4/6] Training XGBoost model...")
    print(f"   Parameters: {params}")

    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=10
    )

    best_iteration = model.best_iteration
    print(f"\n   ✓ Training completed!")
    print(f"   ✓ Best iteration: {best_iteration}")
    print(f"   ✓ Best val nDCG@10: {evals_result['val']['ndcg@10'][best_iteration]:.4f}")

    # Save model
    print("\n[5/6] Saving model...")
    model_dir = base_dir / "models/atlanta"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_ranker.json"
    model.save_model(str(model_path))
    print(f"   ✓ Model saved to: {model_path}")

    # Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    y_pred_test = model.predict(dtest)

    test_metrics = calculate_metrics(y_test, y_pred_test, test_groups)

    print(f"\n   📊 Test Set Metrics:")
    print(f"   ✓ Recall@5:  {test_metrics['Recall@5']:.4f}")
    print(f"   ✓ Recall@10: {test_metrics['Recall@10']:.4f}")
    print(f"   ✓ Recall@20: {test_metrics['Recall@20']:.4f}")
    print(f"   ✓ MRR:       {test_metrics['MRR']:.4f}")
    print(f"   ✓ nDCG@10:   {test_metrics['nDCG@10']:.4f}")

    # Save metrics
    metrics_path = model_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n   ✓ Metrics saved to: {metrics_path}")

    # Generate predictions on test set
    print("\n   Generating top-10 predictions...")

    # Group by query
    test_df = test_df.with_columns(pl.lit(y_pred_test).alias('pred_score'))

    # Get top-10 for each query
    predictions = test_df.sort(['user_id', 'src_gmap_id', 'src_ts', 'pred_score'], descending=[False, False, False, True])
    predictions = predictions.group_by(['user_id', 'src_gmap_id', 'src_ts']).head(10)

    print(f"   ✓ Generated {len(predictions):,} predictions")

    # Save predictions
    pred_dir = model_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    predictions.write_parquet(pred_dir / "test_predictions.parquet")
    print(f"   ✓ Predictions saved to: {pred_dir / 'test_predictions.parquet'}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ ATLANTA MODEL TRAINING COMPLETE")
    print("=" * 80)

    print(f"\n📁 Output Files:")
    print(f"   • Model: {model_path}")
    print(f"   • Metrics: {metrics_path}")
    print(f"   • Predictions: {pred_dir / 'test_predictions.parquet'}")

    print(f"\n📊 Performance Summary:")
    print(f"   • Training samples: {len(train_df):,}")
    print(f"   • Test Recall@10: {test_metrics['Recall@10']:.4f}")
    print(f"   • Test MRR: {test_metrics['MRR']:.4f}")
    print(f"   • Test nDCG@10: {test_metrics['nDCG@10']:.4f}")

    print(f"\n💡 Comparison to Original (Leaked) Model:")
    print(f"   • Original Recall@10: 1.0000 (100% - data leakage)")
    print(f"   • Atlanta-only Recall@10: {test_metrics['Recall@10']:.4f} (realistic)")
    print(f"   • The new metrics are LOWER but VALID")
    print(f"   • This represents true model performance")

    print("\n" + "=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
