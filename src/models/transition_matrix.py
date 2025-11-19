"""
Transition Matrix Generator
Creates sparse transition probability matrices for visualization.
"""

import polars as pl
import numpy as np
import json
from pathlib import Path
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data.features import add_all_features


def build_transition_matrix(model: xgb.Booster, biz_df: pl.DataFrame, 
                           feature_names: List[str], reviews_df: Optional[pl.DataFrame] = None,
                           top_k: int = 10, max_candidates: int = 100) -> pl.DataFrame:
    """
    Build transition probability matrix using trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        biz_df: Business DataFrame with restaurant metadata
        feature_names: List of feature names used in model
        reviews_df: Reviews DataFrame (optional, for enhanced features)
        top_k: Number of top predictions per source restaurant
        max_candidates: Maximum number of candidate destinations per source
        
    Returns:
        DataFrame with transition probabilities
    """
    print("\n" + "=" * 80)
    print("BUILDING TRANSITION MATRIX")
    print("=" * 80)
    
    # Get list of source restaurants (sample for efficiency)
    source_restaurants = biz_df.select("gmap_id").to_series().to_list()
    
    # For efficiency, sample restaurants if too many
    if len(source_restaurants) > 1000:
        import random
        random.seed(42)
        source_restaurants = random.sample(source_restaurants, 1000)
        print(f"  Sampling {len(source_restaurants)} source restaurants for efficiency")
    
    print(f"  Processing {len(source_restaurants)} source restaurants")
    print(f"  Max candidates per source: {max_candidates}")
    print(f"  Top-K predictions: {top_k}")
    
    all_transitions = []
    
    for i, src_gmap_id in enumerate(source_restaurants):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(source_restaurants)} ({i/len(source_restaurants)*100:.1f}%)")
        
        # Get candidate destinations
        candidates = get_candidate_destinations(src_gmap_id, biz_df, max_candidates)
        
        if len(candidates) == 0:
            continue
        
        # Create synthetic pairs for prediction
        pairs_for_prediction = create_prediction_pairs(src_gmap_id, candidates, biz_df)
        
        if len(pairs_for_prediction) == 0:
            continue
        
        # Add features to pairs
        try:
            pairs_with_features = add_all_features(pairs_for_prediction, biz_df, reviews_df)
        except Exception as e:
            print(f"    Warning: Failed to add features for {src_gmap_id}: {e}")
            continue
        
        # Prepare features for prediction
        try:
            X_pred = prepare_features_for_prediction(pairs_with_features, feature_names)
        except Exception as e:
            print(f"    Warning: Failed to prepare features for {src_gmap_id}: {e}")
            continue
        
        # Make predictions
        try:
            predictions = model.predict(X_pred)
            
            # Get top-K predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            for rank, idx in enumerate(top_indices):
                dst_gmap_id = candidates[idx]
                probability = float(predictions[idx])
                
                all_transitions.append({
                    'src_gmap_id': src_gmap_id,
                    'dst_gmap_id': dst_gmap_id,
                    'transition_probability': probability,
                    'rank': rank + 1
                })
                
        except Exception as e:
            print(f"    Warning: Failed to predict for {src_gmap_id}: {e}")
            continue
    
    print(f"\n  Generated {len(all_transitions)} transitions")
    
    # Convert to DataFrame
    if len(all_transitions) == 0:
        print("  Warning: No transitions generated!")
        return pl.DataFrame({
            'src_gmap_id': [],
            'dst_gmap_id': [],
            'transition_probability': [],
            'rank': []
        })
    
    transition_df = pl.DataFrame(all_transitions)
    
    # Add restaurant metadata
    transition_df = transition_df.join(
        biz_df.select(["gmap_id", "name", "category_main", "avg_rating"]).rename({
            "name": "src_name",
            "category_main": "src_category",
            "avg_rating": "src_rating"
        }),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    ).join(
        biz_df.select(["gmap_id", "name", "category_main", "avg_rating"]).rename({
            "name": "dst_name", 
            "category_main": "dst_category",
            "avg_rating": "dst_rating"
        }),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    print(f"  ✓ Transition matrix built with {len(transition_df)} entries")
    return transition_df


def get_candidate_destinations(src_gmap_id: str, biz_df: pl.DataFrame, 
                              max_candidates: int = 100) -> List[str]:
    """
    Get candidate destination restaurants for a source restaurant.
    
    Args:
        src_gmap_id: Source restaurant ID
        biz_df: Business DataFrame
        max_candidates: Maximum number of candidates
        
    Returns:
        List of candidate destination restaurant IDs
    """
    # Get source restaurant info
    src_info = biz_df.filter(pl.col("gmap_id") == src_gmap_id)
    
    if len(src_info) == 0:
        return []
    
    src_lat = src_info["lat"].item()
    src_lon = src_info["lon"].item()
    
    # Strategy 1: Get nearby restaurants (within 20km)
    nearby_restaurants = biz_df.filter(
        (pl.col("gmap_id") != src_gmap_id) &  # Exclude self
        (
            ((pl.col("lat") - src_lat) ** 2 + (pl.col("lon") - src_lon) ** 2) < 0.04  # Approx 20km
        )
    ).select("gmap_id").to_series().to_list()
    
    # Strategy 2: Get restaurants from relative_results if available
    relative_results = []
    try:
        relative_results_data = src_info["relative_results"].item()
        if relative_results_data and isinstance(relative_results_data, list):
            relative_results = [r for r in relative_results_data if r != src_gmap_id][:50]
    except Exception:
        pass
    
    # Strategy 3: Get popular restaurants (high review count)
    popular_restaurants = biz_df.filter(
        pl.col("gmap_id") != src_gmap_id
    ).sort("num_reviews", descending=True).head(50).select("gmap_id").to_series().to_list()
    
    # Combine strategies (prioritize nearby, then relative_results, then popular)
    candidates = []
    
    # Add nearby restaurants first
    candidates.extend(nearby_restaurants[:max_candidates//2])
    
    # Add relative results
    for r in relative_results:
        if r not in candidates and len(candidates) < max_candidates:
            candidates.append(r)
    
    # Add popular restaurants to fill remaining slots
    for r in popular_restaurants:
        if r not in candidates and len(candidates) < max_candidates:
            candidates.append(r)
    
    return candidates[:max_candidates]


def create_prediction_pairs(src_gmap_id: str, candidate_dst_ids: List[str], 
                           biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Create synthetic pairs for prediction.
    
    Args:
        src_gmap_id: Source restaurant ID
        candidate_dst_ids: List of candidate destination IDs
        biz_df: Business DataFrame
        
    Returns:
        DataFrame with synthetic pairs
    """
    from datetime import datetime, timedelta
    
    # Create synthetic timestamps (use current time as base)
    base_time = datetime.now()
    src_time = base_time
    dst_time = base_time + timedelta(hours=2)  # Assume 2-hour gap
    
    # Create pairs
    pairs_data = []
    for dst_gmap_id in candidate_dst_ids:
        pairs_data.append({
            'src_gmap_id': src_gmap_id,
            'dst_gmap_id': dst_gmap_id,
            'user_id': 'synthetic_user',  # Synthetic user for prediction
            'src_ts': src_time,
            'dst_ts': dst_time,
            'label': 0  # Will be ignored during prediction
        })
    
    if len(pairs_data) == 0:
        return pl.DataFrame()
    
    pairs_df = pl.DataFrame(pairs_data)
    
    # Join with business data to get coordinates and other info
    pairs_df = pairs_df.join(
        biz_df.select(["gmap_id", "lat", "lon", "avg_rating", "price_bucket", "category_main"]).rename({
            "lat": "src_lat",
            "lon": "src_lon", 
            "avg_rating": "src_rating",
            "price_bucket": "src_price",
            "category_main": "src_category_main"
        }),
        left_on="src_gmap_id",
        right_on="gmap_id",
        how="left"
    ).join(
        biz_df.select(["gmap_id", "lat", "lon", "avg_rating", "price_bucket", "category_main"]).rename({
            "lat": "dst_lat",
            "lon": "dst_lon",
            "avg_rating": "dst_rating", 
            "price_bucket": "dst_price",
            "category_main": "dst_category_main"
        }),
        left_on="dst_gmap_id",
        right_on="gmap_id",
        how="left"
    )
    
    return pairs_df


def prepare_features_for_prediction(pairs_df: pl.DataFrame, feature_names: List[str]) -> np.ndarray:
    """
    Prepare features for XGBoost prediction.
    
    Args:
        pairs_df: DataFrame with features
        feature_names: List of expected feature names
        
    Returns:
        Feature matrix for prediction
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Handle categorical features (one-hot encode)
    categorical_features = ['distance_bucket', 'direction', 'delta_hours_bucket', 'dst_meal_type', 
                           'cuisine_pair_type', 'user_visit_frequency_bucket']
    
    processed_df = pairs_df.clone()
    
    # One-hot encode categorical features
    for cat_feature in categorical_features:
        if cat_feature in processed_df.columns:
            # Get unique values and create dummy columns
            unique_values = processed_df[cat_feature].unique().to_list()
            for value in unique_values:
                if value is not None:
                    col_name = f"{cat_feature}_{value}"
                    processed_df = processed_df.with_columns([
                        (pl.col(cat_feature) == value).alias(col_name)
                    ])
    
    # Select only the features that exist in both the DataFrame and feature_names
    available_features = []
    for feature in feature_names:
        if feature in processed_df.columns:
            available_features.append(feature)
    
    if len(available_features) == 0:
        raise ValueError("No matching features found between DataFrame and feature_names")
    
    # Extract feature matrix
    feature_matrix = processed_df.select(available_features).to_numpy()
    
    # Handle missing values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    return feature_matrix


def export_transition_matrix(transition_matrix_df: pl.DataFrame, output_dir: Path):
    """
    Export transition matrix in multiple formats.
    
    Args:
        transition_matrix_df: Transition matrix DataFrame
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("EXPORTING TRANSITION MATRIX")
    print("=" * 80)
    
    # Create output directory
    matrix_dir = output_dir / "transition_matrix"
    matrix_dir.mkdir(exist_ok=True)
    
    # Export as Parquet (full data)
    print("\n[1/4] Exporting full matrix as Parquet...")
    parquet_path = matrix_dir / "full_transition_matrix.parquet"
    transition_matrix_df.write_parquet(parquet_path, compression="snappy")
    print(f"  ✓ {parquet_path}")
    print(f"  Size: {parquet_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Export top-K as JSON (for dashboard)
    print("\n[2/4] Exporting top-K as JSON...")
    json_data = {}
    
    for row in transition_matrix_df.iter_rows(named=True):
        src_id = row['src_gmap_id']
        if src_id not in json_data:
            json_data[src_id] = []
        
        json_data[src_id].append({
            'dst_gmap_id': row['dst_gmap_id'],
            'dst_name': row.get('dst_name', ''),
            'dst_category': row.get('dst_category', ''),
            'probability': row['transition_probability'],
            'rank': row['rank']
        })
    
    # Sort by rank for each source
    for src_id in json_data:
        json_data[src_id].sort(key=lambda x: x['rank'])
    
    json_path = matrix_dir / "transition_matrix_topk.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ {json_path}")
    print(f"  Size: {json_path.stat().st_size / 1024:.1f} KB")
    
    # Export summary statistics
    print("\n[3/4] Generating summary statistics...")
    stats = {
        'total_transitions': len(transition_matrix_df),
        'unique_sources': transition_matrix_df['src_gmap_id'].n_unique(),
        'unique_destinations': transition_matrix_df['dst_gmap_id'].n_unique(),
        'avg_probability': transition_matrix_df['transition_probability'].mean(),
        'max_probability': transition_matrix_df['transition_probability'].max(),
        'min_probability': transition_matrix_df['transition_probability'].min(),
        'top_transitions': transition_matrix_df.sort('transition_probability', descending=True).head(10).to_dicts()
    }
    
    stats_path = matrix_dir / "summary_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ {stats_path}")
    
    # Export sparse matrix format (for graph analysis)
    print("\n[4/4] Exporting sparse matrix...")
    try:
        import scipy.sparse as sp
        
        # Create mapping from gmap_id to index
        all_restaurants = list(set(
            transition_matrix_df['src_gmap_id'].to_list() + 
            transition_matrix_df['dst_gmap_id'].to_list()
        ))
        id_to_idx = {gmap_id: idx for idx, gmap_id in enumerate(all_restaurants)}
        
        # Create sparse matrix
        n_restaurants = len(all_restaurants)
        row_indices = [id_to_idx[gmap_id] for gmap_id in transition_matrix_df['src_gmap_id'].to_list()]
        col_indices = [id_to_idx[gmap_id] for gmap_id in transition_matrix_df['dst_gmap_id'].to_list()]
        data = transition_matrix_df['transition_probability'].to_list()
        
        sparse_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_restaurants, n_restaurants))
        
        # Save sparse matrix and ID mapping
        sparse_path = matrix_dir / "sparse_matrix.npz"
        sp.save_npz(sparse_path, sparse_matrix)
        
        mapping_path = matrix_dir / "id_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump({
                'id_to_idx': id_to_idx,
                'idx_to_id': {idx: gmap_id for gmap_id, idx in id_to_idx.items()}
            }, f, indent=2)
        
        print(f"  ✓ {sparse_path}")
        print(f"  ✓ {mapping_path}")
        
    except ImportError:
        print("  ⚠ scipy not available, skipping sparse matrix export")
    
    print(f"\n✓ Transition matrix exported to {matrix_dir}")
    print(f"  Total transitions: {len(transition_matrix_df):,}")
    print(f"  Unique sources: {transition_matrix_df['src_gmap_id'].n_unique():,}")
    print(f"  Unique destinations: {transition_matrix_df['dst_gmap_id'].n_unique():,}")


def load_model_and_generate_matrix(model_path: Path, data_dir: Path, output_dir: Path):
    """
    Load trained model and generate transition matrix.
    
    Args:
        model_path: Path to trained XGBoost model
        data_dir: Path to data directory
        output_dir: Path to output directory
    """
    print("\n" + "=" * 80)
    print("TRANSITION MATRIX GENERATION")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = xgb.Booster()
    model.load_model(str(model_path))
    print("  ✓ Model loaded")
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    biz_df = pl.read_parquet(data_dir / "biz_ga.parquet")
    
    try:
        reviews_df = pl.read_parquet(data_dir / "reviews_ga.parquet")
        print(f"  ✓ Loaded {len(reviews_df):,} reviews")
    except Exception:
        reviews_df = None
        print("  ⚠ Reviews data not available, using basic features only")
    
    print(f"  ✓ Loaded {len(biz_df):,} businesses")
    
    # Get feature names from model (if available) or use default
    try:
        feature_names = model.feature_names
        print(f"  ✓ Using {len(feature_names)} features from model")
    except Exception:
        # Use default feature names from xgboost_ranker.py
        from src.models.xgboost_ranker import NUMERICAL_FEATURES, BOOLEAN_FEATURES
        feature_names = NUMERICAL_FEATURES + BOOLEAN_FEATURES
        print(f"  ⚠ Using default {len(feature_names)} features")
    
    # Build transition matrix
    transition_matrix_df = build_transition_matrix(
        model, biz_df, feature_names, reviews_df, top_k=10, max_candidates=100
    )
    
    # Export results
    export_transition_matrix(transition_matrix_df, output_dir)
    
    print("\n✓✓✓ TRANSITION MATRIX GENERATION COMPLETE ✓✓✓")


if __name__ == "__main__":
    # Example usage
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / "data" / "processed" / "ga" / "models" / "xgboost_ranker.json"
    data_dir = base_dir / "data" / "processed" / "ga"
    output_dir = base_dir / "data" / "processed" / "ga"
    
    load_model_and_generate_matrix(model_path, data_dir, output_dir)
