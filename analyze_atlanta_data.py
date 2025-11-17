#!/usr/bin/env python3
"""
Analyze Atlanta Dataset Size
Helps determine if Atlanta data is sufficient for test set or needs to be excluded from training.
"""

import polars as pl
from pathlib import Path

# Atlanta bounds from your mock data generator
ATLANTA_BOUNDS = {
    "lat_min": 33.6,
    "lat_max": 34.0,
    "lon_min": -84.6,
    "lon_max": -84.2,
}

def analyze_atlanta_data():
    """Analyze the size and characteristics of Atlanta data"""
    
    data_dir = Path("data/processed/ga")
    
    # Load business data
    print("Loading business data...")
    biz_df = pl.read_parquet(data_dir / "biz_ga.parquet")
    
    # Filter Atlanta businesses
    atlanta_biz = biz_df.filter(
        (pl.col("lat") >= ATLANTA_BOUNDS["lat_min"]) &
        (pl.col("lat") <= ATLANTA_BOUNDS["lat_max"]) &
        (pl.col("lon") >= ATLANTA_BOUNDS["lon_min"]) &
        (pl.col("lon") <= ATLANTA_BOUNDS["lon_max"])
    )
    
    print(f"\n{'='*80}")
    print("ATLANTA DATA ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"📊 BUSINESSES:")
    print(f"   Total Georgia: {len(biz_df):,}")
    print(f"   Atlanta area: {len(atlanta_biz):,} ({len(atlanta_biz)/len(biz_df)*100:.1f}%)")
    
    # Load LSTM training data to check user coverage
    print("\n📝 Loading LSTM training data...")
    lstm_dir = data_dir / "lstm_data"
    
    # Check train/val/test splits
    for split in ["train", "val", "test"]:
        split_file = lstm_dir / f"business_{split}.parquet"
        if split_file.exists():
            split_df = pl.read_parquet(split_file)
            
            # Get unique users
            n_users = split_df["user_id"].n_unique()
            n_examples = len(split_df)
            
            # Get businesses visited in this split (target column contains business IDs)
            visited_biz = set(split_df["target"].unique())
            
            # Check overlap with Atlanta
            atlanta_biz_ids = set(atlanta_biz["gmap_id"])
            atlanta_overlap = len(visited_biz & atlanta_biz_ids)
            
            print(f"\n{split.upper()} Split:")
            print(f"   Users: {n_users:,}")
            print(f"   Examples: {n_examples:,}")
            print(f"   Unique businesses visited: {len(visited_biz):,}")
            print(f"   Atlanta businesses visited: {atlanta_overlap:,} ({atlanta_overlap/len(visited_biz)*100:.1f}%)")
    
    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}\n")
    
    atlanta_pct = len(atlanta_biz) / len(biz_df) * 100
    
    if atlanta_pct < 5:
        print("✅ OPTION 1 (RECOMMENDED): Use Atlanta as test set")
        print("   - Atlanta is a small subset (<5% of data)")
        print("   - Sufficient for visualization purposes")
        print("   - Train on rest of Georgia data")
        print("   - Use Atlanta for final inference/visualization only")
        
    elif atlanta_pct < 15:
        print("⚠️  OPTION 2: Exclude Atlanta from training")
        print("   - Atlanta is 5-15% of data (moderate size)")
        print("   - Re-split remaining Georgia data (70/15/15)")
        print("   - Keep Atlanta completely separate for inference")
        print("   - May reduce training data significantly")
        
    else:
        print("❌ OPTION 3: Include Atlanta in training")
        print("   - Atlanta is >15% of data (too large to exclude)")
        print("   - Risk of overfitting is acceptable given data size")
        print("   - Use standard 70/15/15 split including Atlanta")
        print("   - Filter to Atlanta bounds during visualization phase")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    analyze_atlanta_data()

