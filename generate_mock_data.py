#!/usr/bin/env python3
"""
Generate Mock Dataset for Visualization Team

Creates realistic mock data based on the actual data structure from our
preprocessing pipeline. This allows the visualization team to work in parallel
while model training is in progress.

Output:
    data/mock/
    ├── xgboost_predictions.parquet  # XGBoost model predictions
    ├── lstm_predictions.parquet     # LSTM model predictions
    ├── businesses.parquet            # Business metadata
    └── flows.parquet                # User flow data

Usage:
    python generate_mock_data.py [--size SIZE]
    
    --size: Number of mock samples (default: 1000)
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Atlanta bounds (for visualization)
ATLANTA_BOUNDS = {
    "lat_min": 33.6,
    "lat_max": 34.0,
    "lon_min": -84.6,
    "lon_max": -84.2,
}

# Food categories (from our actual vocabulary)
CATEGORIES = [
    "american", "asian", "bakery", "bar", "bbq", "breakfast", "brewery",
    "burger", "cafe", "chinese", "coffee", "dessert", "fast_food",
    "ice_cream", "italian", "mexican", "nightclub", "pizza", "pub",
    "restaurant", "seafood", "steakhouse", "sushi", "wine_bar"
]

# Real Atlanta restaurant names (examples)
RESTAURANT_NAMES = [
    "The Varsity", "Mary Mac's Tea Room", "Busy Bee Cafe", "Fox Bros. Bar-B-Q",
    "Antico Pizza", "Bone's Restaurant", "South City Kitchen", "Canoe",
    "Bacchanalia", "Miller Union", "Staplehouse", "Gunshow", "Kimball House",
    "The Optimist", "Empire State South", "Ticonderoga Club", "BeetleCat",
    "Grana", "Superica", "Little Bear", "Muchacho", "Bread & Butterfly",
    "Pancake Social", "West Egg Cafe", "Flying Biscuit Cafe", "Home grown",
    "Waffle House", "Chick-fil-A", "The General Muir", "Krog Street Market",
    "Ponce City Market", "Krog Bar", "Fred's Meat & Bread", "Gu's Dumplings",
    "Nan Thai Fine Dining", "Taqueria del Sol", "JCT Kitchen & Bar",
    "Bocado", "Holeman and Finch", "Ration and Dram", "Two Urban Licks",
    "Aria", "Nikolai's Roof", "Rathbun's", "Kevin Rathbun Steak",
    "Marcel", "Ecco Midtown", "Lure", "Kyma", "Alma Cocina"
]

# Atlanta neighborhoods
NEIGHBORHOODS = [
    "Midtown", "Buckhead", "Virginia-Highland", "Inman Park", "Old Fourth Ward",
    "Poncey-Highland", "Little Five Points", "East Atlanta", "West Midtown",
    "Downtown", "Decatur", "Brookhaven", "Sandy Springs", "Dunwoody"
]


def generate_business_metadata(n_businesses: int = 100) -> pl.DataFrame:
    """Generate mock business metadata."""
    print(f"\n[Generating {n_businesses} businesses...]")
    
    businesses = []
    for i in range(n_businesses):
        # Generate unique gmap_id (similar format to real data)
        gmap_id = f"0x88f{random.randint(100000, 999999):06x}:{random.randint(100000, 999999):06x}"
        
        # Random location within Atlanta bounds
        lat = random.uniform(ATLANTA_BOUNDS["lat_min"], ATLANTA_BOUNDS["lat_max"])
        lon = random.uniform(ATLANTA_BOUNDS["lon_min"], ATLANTA_BOUNDS["lon_max"])
        
        # Random category
        category = random.choice(CATEGORIES)
        
        # Random name (or use real names)
        if i < len(RESTAURANT_NAMES):
            name = RESTAURANT_NAMES[i]
        else:
            name = f"{random.choice(['The', 'Mama', 'Papa', 'Uncle', 'Aunt'])} {random.choice(['Dragon', 'Phoenix', 'Tiger', 'Bear', 'Eagle'])}'s {category.title()}"
        
        # Random rating (skewed towards higher ratings)
        rating = np.clip(np.random.normal(4.1, 0.6), 1.0, 5.0)
        
        # Random number of reviews (log-normal distribution)
        num_reviews = int(np.random.lognormal(3.5, 1.5))
        
        # Random price bucket (1-4, with nulls)
        price_bucket = random.choice([1, 1, 2, 2, 2, 3, 3, 4, None])
        
        # Random closed status (most are open)
        is_closed = random.random() < 0.15
        
        # Generate some relative results (similar businesses)
        n_relatives = random.randint(5, 20)
        relative_results = [
            f"0x88f{random.randint(100000, 999999):06x}:{random.randint(100000, 999999):06x}"
            for _ in range(n_relatives)
        ]
        
        businesses.append({
            "gmap_id": gmap_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "category_main": category,
            "category_all": [category] + random.sample(CATEGORIES, k=random.randint(0, 2)),
            "avg_rating": rating,
            "num_reviews": num_reviews,
            "price_bucket": price_bucket,
            "is_closed": is_closed,
            "relative_results": relative_results,
        })
    
    df = pl.DataFrame(businesses)
    print(f"  ✓ Generated {len(df)} businesses")
    print(f"  Categories: {df['category_main'].n_unique()}")
    print(f"  Avg rating: {df['avg_rating'].mean():.2f}")
    
    return df


def generate_user_flows(businesses: pl.DataFrame, n_flows: int = 500) -> pl.DataFrame:
    """Generate mock user flow data (consecutive visits)."""
    print(f"\n[Generating {n_flows} user flows...]")
    
    business_ids = businesses["gmap_id"].to_list()
    business_categories = dict(zip(businesses["gmap_id"], businesses["category_main"]))
    business_coords = dict(zip(businesses["gmap_id"], zip(businesses["lat"], businesses["lon"])))
    
    flows = []
    n_users = n_flows // 5  # Each user has ~5 flows on average
    
    for user_idx in range(n_users):
        user_id = f"user_{user_idx:06d}"
        n_user_flows = random.randint(3, 10)
        
        # Start time (random date in 2020-2021)
        current_time = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 730))
        
        for _ in range(n_user_flows):
            # Pick two different businesses
            src_id, dst_id = random.sample(business_ids, k=2)
            
            # Time delta (0.5 to 48 hours)
            delta_hours = random.choice([
                random.uniform(0.5, 2),    # Same day
                random.uniform(2, 6),      # Few hours
                random.uniform(12, 24),    # Next day
                random.uniform(24, 48),    # 1-2 days
            ])
            
            dst_time = current_time + timedelta(hours=delta_hours)
            
            # Get coordinates
            src_lat, src_lon = business_coords[src_id]
            dst_lat, dst_lon = business_coords[dst_id]
            
            # Calculate distance (haversine approximation)
            lat_diff = (dst_lat - src_lat) * 111  # km per degree
            lon_diff = (dst_lon - src_lon) * 111 * np.cos(np.radians(src_lat))
            distance_km = np.sqrt(lat_diff**2 + lon_diff**2)
            
            flows.append({
                "user_id": user_id,
                "src_gmap_id": src_id,
                "dst_gmap_id": dst_id,
                "src_ts": current_time,
                "dst_ts": dst_time,
                "delta_hours": delta_hours,
                "src_category": business_categories[src_id],
                "dst_category": business_categories[dst_id],
                "src_lat": src_lat,
                "src_lon": src_lon,
                "dst_lat": dst_lat,
                "dst_lon": dst_lon,
                "distance_km": distance_km,
            })
            
            # Update current time for next flow
            current_time = dst_time + timedelta(hours=random.uniform(1, 72))
    
    df = pl.DataFrame(flows[:n_flows])
    print(f"  ✓ Generated {len(df)} flows")
    print(f"  Unique users: {df['user_id'].n_unique()}")
    print(f"  Avg distance: {df['distance_km'].mean():.2f} km")
    print(f"  Avg time delta: {df['delta_hours'].mean():.2f} hours")
    
    return df


def generate_xgboost_predictions(flows: pl.DataFrame, businesses: pl.DataFrame) -> pl.DataFrame:
    """Generate mock XGBoost predictions (top-K recommendations)."""
    print(f"\n[Generating XGBoost predictions...]")
    
    business_ids = businesses["gmap_id"].to_list()
    business_names = dict(zip(businesses["gmap_id"], businesses["name"]))
    business_categories = dict(zip(businesses["gmap_id"], businesses["category_main"]))
    business_ratings = dict(zip(businesses["gmap_id"], businesses["avg_rating"]))
    
    predictions = []
    
    for flow in flows.iter_rows(named=True):
        user_id = flow["user_id"]
        src_id = flow["src_gmap_id"]
        actual_dst_id = flow["dst_gmap_id"]
        
        # Generate top-K predictions (K=10)
        # Include the actual destination with high probability
        if random.random() < 0.7:  # 70% recall@10
            candidates = [actual_dst_id] + random.sample(
                [b for b in business_ids if b != src_id and b != actual_dst_id], 
                k=9
            )
        else:
            candidates = random.sample(
                [b for b in business_ids if b != src_id], 
                k=10
            )
        
        # Shuffle and assign scores (descending)
        random.shuffle(candidates)
        scores = sorted([random.uniform(0.3, 0.95) for _ in range(10)], reverse=True)
        
        for rank, (dst_id, score) in enumerate(zip(candidates, scores), start=1):
            predictions.append({
                "user_id": user_id,
                "src_gmap_id": src_id,
                "dst_gmap_id": dst_id,
                "dst_name": business_names.get(dst_id, "Unknown"),
                "dst_category": business_categories.get(dst_id, "restaurant"),
                "dst_rating": business_ratings.get(dst_id, 4.0),
                "score": score,
                "rank": rank,
                "is_actual": dst_id == actual_dst_id,
                "model": "xgboost",
            })
    
    df = pl.DataFrame(predictions)
    print(f"  ✓ Generated {len(df)} predictions")
    print(f"  Recall@10: {df.filter(pl.col('is_actual')).height / flows.height:.2%}")
    print(f"  Avg score: {df['score'].mean():.3f}")
    
    return df


def generate_lstm_predictions(flows: pl.DataFrame, businesses: pl.DataFrame) -> pl.DataFrame:
    """Generate mock LSTM predictions (category-level and business-level)."""
    print(f"\n[Generating LSTM predictions...]")
    
    business_ids = businesses["gmap_id"].to_list()
    business_names = dict(zip(businesses["gmap_id"], businesses["name"]))
    business_categories = dict(zip(businesses["gmap_id"], businesses["category_main"]))
    business_ratings = dict(zip(businesses["gmap_id"], businesses["avg_rating"]))
    
    predictions = []
    
    for flow in flows.iter_rows(named=True):
        user_id = flow["user_id"]
        actual_dst_id = flow["dst_gmap_id"]
        actual_category = flow["dst_category"]
        
        # Category-level predictions (top-5 categories)
        if random.random() < 0.6:  # 60% accuracy
            top_categories = [actual_category] + random.sample(
                [c for c in CATEGORIES if c != actual_category], 
                k=4
            )
        else:
            top_categories = random.sample(CATEGORIES, k=5)
        
        random.shuffle(top_categories)
        cat_probs = sorted([random.uniform(0.1, 0.8) for _ in range(5)], reverse=True)
        cat_probs = [p / sum(cat_probs) for p in cat_probs]  # Normalize to sum to 1
        
        for rank, (category, prob) in enumerate(zip(top_categories, cat_probs), start=1):
            predictions.append({
                "user_id": user_id,
                "prediction_type": "category",
                "predicted_category": category,
                "predicted_business_id": None,
                "predicted_business_name": None,
                "probability": prob,
                "rank": rank,
                "is_actual": category == actual_category,
                "model": "lstm_category",
            })
        
        # Business-level predictions (top-10 businesses)
        if random.random() < 0.5:  # 50% recall@10
            top_businesses = [actual_dst_id] + random.sample(
                [b for b in business_ids if b != actual_dst_id], 
                k=9
            )
        else:
            top_businesses = random.sample(business_ids, k=10)
        
        random.shuffle(top_businesses)
        biz_probs = sorted([random.uniform(0.05, 0.4) for _ in range(10)], reverse=True)
        biz_probs = [p / sum(biz_probs) for p in biz_probs]  # Normalize
        
        for rank, (biz_id, prob) in enumerate(zip(top_businesses, biz_probs), start=1):
            predictions.append({
                "user_id": user_id,
                "prediction_type": "business",
                "predicted_category": business_categories.get(biz_id, "restaurant"),
                "predicted_business_id": biz_id,
                "predicted_business_name": business_names.get(biz_id, "Unknown"),
                "probability": prob,
                "rank": rank,
                "is_actual": biz_id == actual_dst_id,
                "model": "lstm_business",
            })
    
    df = pl.DataFrame(predictions)
    
    # Calculate metrics
    cat_preds = df.filter(pl.col("prediction_type") == "category")
    biz_preds = df.filter(pl.col("prediction_type") == "business")
    
    print(f"  ✓ Generated {len(df)} predictions")
    print(f"  Category accuracy: {cat_preds.filter(pl.col('is_actual')).height / flows.height:.2%}")
    print(f"  Business Recall@10: {biz_preds.filter(pl.col('is_actual')).height / flows.height:.2%}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate mock data for visualization team")
    parser.add_argument("--size", type=int, default=1000, help="Number of mock flows (default: 1000)")
    parser.add_argument("--businesses", type=int, default=100, help="Number of businesses (default: 100)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MOCK DATA GENERATOR FOR VISUALIZATION TEAM")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Businesses: {args.businesses}")
    print(f"  Flows: {args.size}")
    print(f"  Location: Atlanta, GA")
    
    # Create output directory
    output_dir = Path("data/mock")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    businesses = generate_business_metadata(n_businesses=args.businesses)
    flows = generate_user_flows(businesses, n_flows=args.size)
    xgb_predictions = generate_xgboost_predictions(flows, businesses)
    lstm_predictions = generate_lstm_predictions(flows, businesses)
    
    # Save to parquet files
    print("\n[Saving files...]")
    businesses.write_parquet(output_dir / "businesses.parquet")
    print(f"  ✓ Saved businesses.parquet ({len(businesses)} rows)")
    
    flows.write_parquet(output_dir / "flows.parquet")
    print(f"  ✓ Saved flows.parquet ({len(flows)} rows)")
    
    xgb_predictions.write_parquet(output_dir / "xgboost_predictions.parquet")
    print(f"  ✓ Saved xgboost_predictions.parquet ({len(xgb_predictions)} rows)")
    
    lstm_predictions.write_parquet(output_dir / "lstm_predictions.parquet")
    print(f"  ✓ Saved lstm_predictions.parquet ({len(lstm_predictions)} rows)")
    
    # Also save as JSON for easier inspection (pretty-printed)
    print("\n[Saving JSON samples...]")
    import json as json_lib
    
    # Convert to JSON with pretty printing
    businesses.head(20).write_json(output_dir / "businesses_sample.json")
    with open(output_dir / "businesses_sample.json", "r") as f:
        data = json_lib.load(f)
    with open(output_dir / "businesses_sample.json", "w") as f:
        json_lib.dump(data, f, indent=2)
    
    flows.head(20).write_json(output_dir / "flows_sample.json")
    with open(output_dir / "flows_sample.json", "r") as f:
        data = json_lib.load(f)
    with open(output_dir / "flows_sample.json", "w") as f:
        json_lib.dump(data, f, indent=2)
    
    xgb_predictions.head(50).write_json(output_dir / "xgboost_predictions_sample.json")
    with open(output_dir / "xgboost_predictions_sample.json", "r") as f:
        data = json_lib.load(f)
    with open(output_dir / "xgboost_predictions_sample.json", "w") as f:
        json_lib.dump(data, f, indent=2)
    
    lstm_predictions.head(50).write_json(output_dir / "lstm_predictions_sample.json")
    with open(output_dir / "lstm_predictions_sample.json", "r") as f:
        data = json_lib.load(f)
    with open(output_dir / "lstm_predictions_sample.json", "w") as f:
        json_lib.dump(data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  - businesses.parquet ({len(businesses)} rows)")
    print(f"  - flows.parquet ({len(flows)} rows)")
    print(f"  - xgboost_predictions.parquet ({len(xgb_predictions)} rows)")
    print(f"  - lstm_predictions.parquet ({len(lstm_predictions)} rows)")
    print(f"  - README.md (documentation)")
    print(f"  - *_sample.json (JSON samples for inspection)")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.parquet"))
    print(f"\nTotal size: {total_size / 1024 / 1024:.2f} MB")
    
    print("\nMock data generation complete!")
    print("\nVisualization team can now use this data for development.")
    print("The structure matches the real data that will be produced by model training.")


if __name__ == "__main__":
    main()

