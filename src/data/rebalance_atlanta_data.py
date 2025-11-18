#!/usr/bin/env python3
"""
Rebalance Train/Val/Test Split with Atlanta Data

Problem: Current split has all Atlanta users in test (4.9M), non-Atlanta in train/val (1.8M/987K)
Solution: Split Atlanta users into train/val/test while keeping user sequences intact

Author: Team 15
Date: November 2025
"""

import polars as pl
from pathlib import Path
import numpy as np

def rebalance_data(
    data_dir: Path,
    atlanta_train_ratio: float = 0.60,
    atlanta_val_ratio: float = 0.20,
    atlanta_test_ratio: float = 0.20,
    random_seed: int = 42
):
    """
    Rebalance data by splitting Atlanta users while preserving sequences.

    Args:
        data_dir: Path to lstm_data directory
        atlanta_train_ratio: Proportion of Atlanta users for training
        atlanta_val_ratio: Proportion of Atlanta users for validation
        atlanta_test_ratio: Proportion of Atlanta users for testing
        random_seed: Random seed for reproducibility
    """

    print("=" * 80)
    print("REBALANCING TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    # Set random seed
    np.random.seed(random_seed)

    # Load current data
    print("\n1. Loading current data splits...")
    train_df = pl.read_parquet(data_dir / "business_train.parquet")
    val_df = pl.read_parquet(data_dir / "business_val.parquet")
    test_df = pl.read_parquet(data_dir / "business_test.parquet")

    print(f"   Current train: {len(train_df):,} examples")
    print(f"   Current val:   {len(val_df):,} examples")
    print(f"   Current test:  {len(test_df):,} examples (ALL Atlanta)")

    # Get unique Atlanta users
    print("\n2. Extracting unique Atlanta users...")
    atlanta_users = test_df['user_id'].unique().to_list()
    print(f"   Found {len(atlanta_users):,} unique Atlanta users")

    # Shuffle and split Atlanta users (not examples!)
    print("\n3. Splitting Atlanta USERS into train/val/test...")
    np.random.shuffle(atlanta_users)

    n_users = len(atlanta_users)
    n_train = int(n_users * atlanta_train_ratio)
    n_val = int(n_users * atlanta_val_ratio)

    atlanta_train_users = set(atlanta_users[:n_train])
    atlanta_val_users = set(atlanta_users[n_train:n_train + n_val])
    atlanta_test_users = set(atlanta_users[n_train + n_val:])

    print(f"   Atlanta train users: {len(atlanta_train_users):,} ({atlanta_train_ratio:.0%})")
    print(f"   Atlanta val users:   {len(atlanta_val_users):,} ({atlanta_val_ratio:.0%})")
    print(f"   Atlanta test users:  {len(atlanta_test_users):,} ({atlanta_test_ratio:.0%})")

    # Split test_df by user (preserves ALL examples for each user)
    print("\n4. Splitting Atlanta examples by user...")
    atlanta_train_examples = test_df.filter(pl.col('user_id').is_in(atlanta_train_users))
    atlanta_val_examples = test_df.filter(pl.col('user_id').is_in(atlanta_val_users))
    atlanta_test_examples = test_df.filter(pl.col('user_id').is_in(atlanta_test_users))

    print(f"   Atlanta train examples: {len(atlanta_train_examples):,}")
    print(f"   Atlanta val examples:   {len(atlanta_val_examples):,}")
    print(f"   Atlanta test examples:  {len(atlanta_test_examples):,}")

    # Combine with existing non-Atlanta data
    print("\n5. Combining with existing non-Atlanta data...")
    new_train = pl.concat([train_df, atlanta_train_examples])
    new_val = pl.concat([val_df, atlanta_val_examples])
    new_test = atlanta_test_examples  # Only Atlanta test users

    print(f"   New train: {len(new_train):,} examples (+{len(atlanta_train_examples):,})")
    print(f"   New val:   {len(new_val):,} examples (+{len(atlanta_val_examples):,})")
    print(f"   New test:  {len(new_test):,} examples (-{len(test_df) - len(new_test):,})")

    # Verify no user appears in multiple splits
    print("\n6. Verifying user separation...")
    train_users = set(new_train['user_id'].unique().to_list())
    val_users = set(new_val['user_id'].unique().to_list())
    test_users = set(new_test['user_id'].unique().to_list())

    train_val_overlap = train_users & val_users
    train_test_overlap = train_users & test_users
    val_test_overlap = val_users & test_users

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("   ERROR: User overlap detected!")
        print(f"   Train-Val overlap: {len(train_val_overlap)}")
        print(f"   Train-Test overlap: {len(train_test_overlap)}")
        print(f"   Val-Test overlap: {len(val_test_overlap)}")
        raise ValueError("User sequences split across datasets!")

    print("   ✓ No user overlap - sequences preserved!")

    # Save rebalanced data
    print("\n7. Saving rebalanced data...")
    output_dir = data_dir / "rebalanced"
    output_dir.mkdir(exist_ok=True)

    new_train.write_parquet(output_dir / "business_train.parquet")
    new_val.write_parquet(output_dir / "business_val.parquet")
    new_test.write_parquet(output_dir / "business_test.parquet")

    print(f"   Saved to: {output_dir}/")

    # Print final statistics
    print("\n" + "=" * 80)
    print("REBALANCING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Split:")
    print(f"  Train: {len(new_train):,} examples from {len(train_users):,} users")
    print(f"  Val:   {len(new_val):,} examples from {len(val_users):,} users")
    print(f"  Test:  {len(new_test):,} examples from {len(test_users):,} users")

    print(f"\nTrain/Test Ratio: {len(new_train) / len(new_test):.2f}x")
    print(f"  (Before: {len(train_df) / len(test_df):.2f}x)")

    # Show geographic distribution
    print(f"\nGeographic Distribution in Training:")
    non_atlanta_train = len(train_df)
    atlanta_train = len(atlanta_train_examples)
    total_train = len(new_train)
    print(f"  Non-Atlanta: {non_atlanta_train:,} ({non_atlanta_train/total_train:.1%})")
    print(f"  Atlanta:     {atlanta_train:,} ({atlanta_train/total_train:.1%})")

    print("\n✓ Use 'data/processed/ga/lstm_data/rebalanced/' in notebooks")
    print("=" * 80)

    return {
        'train': new_train,
        'val': new_val,
        'test': new_test,
        'stats': {
            'train_examples': len(new_train),
            'val_examples': len(new_val),
            'test_examples': len(new_test),
            'train_users': len(train_users),
            'val_users': len(val_users),
            'test_users': len(test_users),
        }
    }


if __name__ == "__main__":
    # Run rebalancing
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ga" / "lstm_data"

    print("\nThis will create a rebalanced train/val/test split.")
    print("Atlanta users will be split 60/20/20 for train/val/test.")
    print("All examples for each user will stay together (sequences preserved).\n")

    rebalance_data(
        data_dir=data_dir,
        atlanta_train_ratio=0.60,
        atlanta_val_ratio=0.20,
        atlanta_test_ratio=0.20,
        random_seed=42
    )
