"""
Create proper Atlanta-only train/val/test splits.

This script:
1. Filters to Atlanta-only restaurant pairs
2. Performs temporal split (70/15/15)
3. Ensures no data leakage
4. Saves new splits for XGBoost training
"""

import polars as pl
from pathlib import Path
from datetime import datetime

def main():
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("CREATING ATLANTA-ONLY TRAIN/VAL/TEST SPLITS")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    atlanta_biz = pl.read_parquet(base_dir / "data/viz/atlanta_businesses.parquet")

    # Load original full data
    original_train = pl.read_parquet(base_dir / "data/processed/ga/xgboost_data/train.parquet")
    original_val = pl.read_parquet(base_dir / "data/processed/ga/xgboost_data/val.parquet")
    original_test = pl.read_parquet(base_dir / "data/processed/ga/xgboost_data/test.parquet")

    print(f"   ✓ Atlanta businesses: {len(atlanta_biz):,}")
    print(f"   ✓ Original train: {len(original_train):,}")
    print(f"   ✓ Original val: {len(original_val):,}")
    print(f"   ✓ Original test: {len(original_test):,}")

    # Filter to Atlanta-only
    print("\n[2/5] Filtering to Atlanta-only pairs...")
    atlanta_biz_ids = set(atlanta_biz['gmap_id'].to_list())

    # Combine all data first, then re-split temporally
    all_data = pl.concat([original_train, original_val, original_test])
    print(f"   ✓ Combined all splits: {len(all_data):,} samples")

    # Filter to Atlanta-only (both src and dst in Atlanta)
    atlanta_data = all_data.filter(
        pl.col('src_gmap_id').is_in(list(atlanta_biz_ids)) &
        pl.col('dst_gmap_id').is_in(list(atlanta_biz_ids))
    )
    print(f"   ✓ Atlanta-only samples: {len(atlanta_data):,}")
    print(f"   ✓ Reduction: {(1 - len(atlanta_data)/len(all_data))*100:.1f}%")

    # Sort by timestamp to ensure temporal ordering
    print("\n[3/5] Sorting by timestamp for temporal split...")
    atlanta_data = atlanta_data.sort('src_ts')

    # Calculate split indices
    total_samples = len(atlanta_data)
    train_end = int(total_samples * 0.70)
    val_end = train_end + int(total_samples * 0.15)

    print(f"   ✓ Total samples: {total_samples:,}")
    print(f"   ✓ Train end index: {train_end:,} (70%)")
    print(f"   ✓ Val end index: {val_end:,} (85%)")

    # Split the data
    print("\n[4/5] Creating new splits...")
    atlanta_train = atlanta_data[:train_end]
    atlanta_val = atlanta_data[train_end:val_end]
    atlanta_test = atlanta_data[val_end:]

    print(f"   ✓ Train: {len(atlanta_train):,} samples")
    print(f"   ✓ Val: {len(atlanta_val):,} samples")
    print(f"   ✓ Test: {len(atlanta_test):,} samples")

    # Verify temporal ordering
    train_max_ts = atlanta_train['src_ts'].max()
    val_min_ts = atlanta_val['src_ts'].min()
    val_max_ts = atlanta_val['src_ts'].max()
    test_min_ts = atlanta_test['src_ts'].min()

    # Convert to seconds if needed (timestamps might be datetime objects or milliseconds)
    def to_datetime_str(ts):
        if isinstance(ts, datetime):
            return ts.strftime('%Y-%m-%d')
        else:
            # Assume milliseconds
            return datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d')

    print(f"\n   Temporal verification:")
    print(f"   Train: earliest to {to_datetime_str(train_max_ts)}")
    print(f"   Val:   {to_datetime_str(val_min_ts)} to {to_datetime_str(val_max_ts)}")
    print(f"   Test:  {to_datetime_str(test_min_ts)} onwards")

    # Check for business overlap (should be 100% - same businesses, different time periods)
    train_businesses = set(atlanta_train['src_gmap_id'].unique().to_list() + atlanta_train['dst_gmap_id'].unique().to_list())
    val_businesses = set(atlanta_val['src_gmap_id'].unique().to_list() + atlanta_val['dst_gmap_id'].unique().to_list())
    test_businesses = set(atlanta_test['src_gmap_id'].unique().to_list() + atlanta_test['dst_gmap_id'].unique().to_list())

    overlap_train_test = len(train_businesses.intersection(test_businesses))

    print(f"\n   Business overlap:")
    print(f"   Train businesses: {len(train_businesses):,}")
    print(f"   Val businesses: {len(val_businesses):,}")
    print(f"   Test businesses: {len(test_businesses):,}")
    print(f"   Train-Test overlap: {overlap_train_test:,} ({overlap_train_test/len(test_businesses)*100:.1f}%)")
    print(f"   ⚠️  Note: Business overlap is EXPECTED in temporal splits")
    print(f"   ✓  What matters: Model sees DIFFERENT time periods, not different businesses")

    # Check label distribution
    print(f"\n   Label distribution:")
    for split_name, split_data in [("Train", atlanta_train), ("Val", atlanta_val), ("Test", atlanta_test)]:
        pos = split_data.filter(pl.col('label') == 1)
        neg = split_data.filter(pl.col('label') == 0)
        print(f"   {split_name}: {len(pos):,} pos ({len(pos)/len(split_data)*100:.1f}%), {len(neg):,} neg ({len(neg)/len(split_data)*100:.1f}%)")

    # Save new splits
    print("\n[5/5] Saving Atlanta-only splits...")
    output_dir = base_dir / "data/processed/atlanta/xgboost_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    atlanta_train.write_parquet(output_dir / "train.parquet")
    atlanta_val.write_parquet(output_dir / "val.parquet")
    atlanta_test.write_parquet(output_dir / "test.parquet")

    print(f"   ✓ Saved to: {output_dir}")
    print(f"   ✓ train.parquet: {len(atlanta_train):,} samples")
    print(f"   ✓ val.parquet: {len(atlanta_val):,} samples")
    print(f"   ✓ test.parquet: {len(atlanta_test):,} samples")

    # Summary
    print("\n" + "=" * 80)
    print("✅ ATLANTA-ONLY SPLITS CREATED SUCCESSFULLY")
    print("=" * 80)

    print(f"\n📊 Summary:")
    print(f"   • Total Atlanta samples: {total_samples:,}")
    print(f"   • Training: {len(atlanta_train):,} (70%)")
    print(f"   • Validation: {len(atlanta_val):,} (15%)")
    print(f"   • Testing: {len(atlanta_test):,} (15%)")
    print(f"   • Atlanta restaurants: {len(atlanta_biz_ids):,}")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Run: python retrain_atlanta_xgboost.py")
    print(f"   2. This will train a new model on Atlanta-only data")
    print(f"   3. Expect more realistic metrics (not 100% anymore)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()