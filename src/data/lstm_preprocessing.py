#!/usr/bin/env python3
"""
Phase A5: LSTM-Specific Preprocessing

This script prepares LSTM training data by:
1. Building vocabularies (category-level and business-level)
2. Applying sequence windowing to create (input, target) pairs
3. Padding/truncating sequences to fixed length

Input:
    - data/processed/ga/lstm_data/train.parquet
    - data/processed/ga/lstm_data/val.parquet
    - data/processed/ga/lstm_data/test.parquet
    - data/processed/ga/biz_ga.parquet

Output:
    - data/processed/ga/lstm_data/category_train.parquet
    - data/processed/ga/lstm_data/category_val.parquet
    - data/processed/ga/lstm_data/category_test.parquet
    - data/processed/ga/lstm_data/business_train.parquet
    - data/processed/ga/lstm_data/business_val.parquet
    - data/processed/ga/lstm_data/business_test.parquet
    - data/processed/ga/lstm_data/category_vocab.json
    - data/processed/ga/lstm_data/business_vocab.json

Author: Team 15
Date: October 2025
"""

import polars as pl
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

# Configuration
MAX_SEQ_LEN = 20  # Maximum sequence length for LSTM
TOP_K_BUSINESSES = 20000  # Top K most popular businesses for business-level vocab
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
LSTM_DATA_DIR = BASE_DIR / "data" / "processed" / "ga" / "lstm_data"


def build_category_vocabulary(train_df: pl.DataFrame) -> Dict[str, int]:
    """
    Build category-level vocabulary.
    
    Args:
        train_df: Training sequences DataFrame
        
    Returns:
        Dictionary mapping category names to indices
    """
    print("\n[1/6] Building category vocabulary...")
    
    # Get unique categories from training data
    unique_categories = train_df.select("category_main").unique().to_series().to_list()
    unique_categories = sorted([cat for cat in unique_categories if cat is not None])
    
    print(f"  Found {len(unique_categories)} unique food categories")
    
    # Create vocabulary: <PAD>=0, <UNK>=1, then categories
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, category in enumerate(unique_categories, start=2):
        vocab[category] = idx
    
    print(f"  Category vocabulary size: {len(vocab)} tokens")
    print(f"  Categories: {unique_categories[:10]}...")
    
    return vocab


def build_business_vocabulary(train_df: pl.DataFrame, top_k: int = TOP_K_BUSINESSES) -> Dict[str, int]:
    """
    Build business-level vocabulary with top-K most popular businesses.
    
    Args:
        train_df: Training sequences DataFrame
        top_k: Number of top businesses to include
        
    Returns:
        Dictionary mapping business IDs to indices
    """
    print(f"\n[2/6] Building business vocabulary (top-{top_k})...")
    
    # Count business frequencies
    business_counts = (
        train_df
        .group_by("gmap_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    
    total_businesses = len(business_counts)
    print(f"  Total unique businesses in training: {total_businesses:,}")
    
    # Take top-K
    top_businesses = business_counts.head(top_k).select("gmap_id").to_series().to_list()
    
    # Create vocabulary: <PAD>=0, <UNK>=1, then top-K businesses
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, business_id in enumerate(top_businesses, start=2):
        vocab[business_id] = idx
    
    # Calculate coverage
    top_k_visits = business_counts.head(top_k).select("count").sum().item()
    total_visits = business_counts.select("count").sum().item()
    coverage = (top_k_visits / total_visits) * 100
    
    print(f"  Business vocabulary size: {len(vocab)} tokens")
    print(f"  Coverage: {coverage:.2f}% of training visits")
    print(f"  Businesses mapped to <UNK>: {total_businesses - top_k:,}")
    
    return vocab


def apply_sequence_windowing(
    sequences_df: pl.DataFrame,
    vocab: Dict[str, int],
    vocab_type: str,  # "category" or "business"
    max_seq_len: int = MAX_SEQ_LEN
) -> pl.DataFrame:
    """
    Apply sequence windowing to create (input, target) training examples.
    
    For each user sequence [A, B, C, D, E], creates:
        - Input=[A], Target=B
        - Input=[A, B], Target=C
        - Input=[A, B, C], Target=D
        - Input=[A, B, C, D], Target=E
    
    Args:
        sequences_df: User sequences DataFrame
        vocab: Vocabulary mapping (category or business)
        vocab_type: "category" or "business"
        max_seq_len: Maximum input sequence length
        
    Returns:
        DataFrame with windowed sequences
    """
    print(f"\n  Applying sequence windowing for {vocab_type}-level LSTM...")
    
    # Determine which column to use
    id_col = "category_main" if vocab_type == "category" else "gmap_id"
    
    # Convert to pandas for easier windowing (Polars doesn't have great windowing support yet)
    df_pandas = sequences_df.to_pandas()
    
    windowed_data = []
    
    # Group by user
    for user_id, group in df_pandas.groupby("user_id"):
        # Sort by sequence index
        group = group.sort_values("seq_idx")
        
        # Get the sequence of IDs
        sequence = group[id_col].tolist()
        
        # Skip if sequence is too short (need at least 2 items: 1 input + 1 target)
        if len(sequence) < 2:
            continue
        
        # Create windows
        for i in range(1, len(sequence)):
            # Input: all items up to position i (truncate to max_seq_len)
            input_seq = sequence[max(0, i - max_seq_len):i]
            
            # Target: item at position i
            target = sequence[i]
            
            # Map to vocabulary indices
            input_indices = [vocab.get(item, vocab[UNK_TOKEN]) for item in input_seq]
            target_index = vocab.get(target, vocab[UNK_TOKEN])
            
            # Pad input sequence to max_seq_len
            padding_needed = max_seq_len - len(input_indices)
            if padding_needed > 0:
                input_indices = [vocab[PAD_TOKEN]] * padding_needed + input_indices
            
            windowed_data.append({
                "user_id": user_id,
                "input_seq": input_indices,
                "target": target_index,
                "seq_len": len(input_seq)  # Original length before padding
            })
    
    # Convert back to Polars
    windowed_df = pl.DataFrame(windowed_data)
    
    print(f"    Created {len(windowed_df):,} training examples from {df_pandas['user_id'].nunique():,} users")
    print(f"    Avg examples per user: {len(windowed_df) / df_pandas['user_id'].nunique():.1f}")
    
    return windowed_df


def process_split(
    split_name: str,
    sequences_df: pl.DataFrame,
    category_vocab: Dict[str, int],
    business_vocab: Dict[str, int],
    max_seq_len: int = MAX_SEQ_LEN
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Process a single data split (train/val/test) for both category and business levels.
    
    Args:
        split_name: "train", "val", or "test"
        sequences_df: User sequences DataFrame
        category_vocab: Category vocabulary
        business_vocab: Business vocabulary
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (category_windowed_df, business_windowed_df)
    """
    print(f"\n[Processing {split_name.upper()} split]")
    print(f"  Input: {len(sequences_df):,} visits from {sequences_df['user_id'].n_unique():,} users")
    
    # Apply windowing for category-level
    category_windowed = apply_sequence_windowing(
        sequences_df, category_vocab, "category", max_seq_len
    )
    
    # Apply windowing for business-level
    business_windowed = apply_sequence_windowing(
        sequences_df, business_vocab, "business", max_seq_len
    )
    
    return category_windowed, business_windowed


def main():
    """Main preprocessing pipeline."""
    
    print("=" * 80)
    print("PHASE A5: LSTM-SPECIFIC PREPROCESSING")
    print("=" * 80)
    
    # Load data
    print("\n[Loading data...]")
    train_df = pl.read_parquet(LSTM_DATA_DIR / "train.parquet")
    val_df = pl.read_parquet(LSTM_DATA_DIR / "val.parquet")
    test_df = pl.read_parquet(LSTM_DATA_DIR / "test.parquet")
    
    print(f"  Train: {len(train_df):,} visits from {train_df['user_id'].n_unique():,} users")
    print(f"  Val: {len(val_df):,} visits from {val_df['user_id'].n_unique():,} users")
    print(f"  Test: {len(test_df):,} visits from {test_df['user_id'].n_unique():,} users")
    
    # Build vocabularies (only from training data)
    category_vocab = build_category_vocabulary(train_df)
    business_vocab = build_business_vocabulary(train_df, TOP_K_BUSINESSES)
    
    # Process each split
    print("\n" + "=" * 80)
    print("APPLYING SEQUENCE WINDOWING")
    print("=" * 80)
    
    category_train, business_train = process_split("train", train_df, category_vocab, business_vocab)
    category_val, business_val = process_split("val", val_df, category_vocab, business_vocab)
    category_test, business_test = process_split("test", test_df, category_vocab, business_vocab)
    
    # Save processed data
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)
    
    print("\n[Saving category-level data...]")
    category_train.write_parquet(LSTM_DATA_DIR / "category_train.parquet")
    category_val.write_parquet(LSTM_DATA_DIR / "category_val.parquet")
    category_test.write_parquet(LSTM_DATA_DIR / "category_test.parquet")
    print(f"  ✓ Saved category_train.parquet ({len(category_train):,} examples)")
    print(f"  ✓ Saved category_val.parquet ({len(category_val):,} examples)")
    print(f"  ✓ Saved category_test.parquet ({len(category_test):,} examples)")
    
    print("\n[Saving business-level data...]")
    business_train.write_parquet(LSTM_DATA_DIR / "business_train.parquet")
    business_val.write_parquet(LSTM_DATA_DIR / "business_val.parquet")
    business_test.write_parquet(LSTM_DATA_DIR / "business_test.parquet")
    print(f"  ✓ Saved business_train.parquet ({len(business_train):,} examples)")
    print(f"  ✓ Saved business_val.parquet ({len(business_val):,} examples)")
    print(f"  ✓ Saved business_test.parquet ({len(business_test):,} examples)")
    
    # Save vocabularies
    print("\n[Saving vocabularies...]")
    with open(LSTM_DATA_DIR / "category_vocab.json", "w") as f:
        json.dump(category_vocab, f, indent=2)
    print(f"  ✓ Saved category_vocab.json ({len(category_vocab)} tokens)")
    
    with open(LSTM_DATA_DIR / "business_vocab.json", "w") as f:
        json.dump(business_vocab, f, indent=2)
    print(f"  ✓ Saved business_vocab.json ({len(business_vocab)} tokens)")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\n📊 CATEGORY-LEVEL LSTM DATA:")
    print(f"  Vocabulary size: {len(category_vocab)} tokens")
    print(f"  Train examples: {len(category_train):,}")
    print(f"  Val examples: {len(category_val):,}")
    print(f"  Test examples: {len(category_test):,}")
    print(f"  Total examples: {len(category_train) + len(category_val) + len(category_test):,}")
    print(f"  Max sequence length: {MAX_SEQ_LEN}")
    
    print("\n📊 BUSINESS-LEVEL LSTM DATA:")
    print(f"  Vocabulary size: {len(business_vocab)} tokens")
    print(f"  Train examples: {len(business_train):,}")
    print(f"  Val examples: {len(business_val):,}")
    print(f"  Test examples: {len(business_test):,}")
    print(f"  Total examples: {len(business_train) + len(business_val) + len(business_test):,}")
    print(f"  Max sequence length: {MAX_SEQ_LEN}")
    
    # Calculate file sizes
    category_size = sum([
        (LSTM_DATA_DIR / f"category_{split}.parquet").stat().st_size
        for split in ["train", "val", "test"]
    ]) / (1024 * 1024)
    
    business_size = sum([
        (LSTM_DATA_DIR / f"business_{split}.parquet").stat().st_size
        for split in ["train", "val", "test"]
    ]) / (1024 * 1024)
    
    print(f"\n💾 STORAGE:")
    print(f"  Category-level data: {category_size:.1f} MB")
    print(f"  Business-level data: {business_size:.1f} MB")
    print(f"  Vocabularies: <0.1 MB")
    print(f"  Total: {category_size + business_size:.1f} MB")
    
    print("\n" + "=" * 80)
    print("✓✓✓ PHASE A5 COMPLETE ✓✓✓")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Use category_*.parquet for category-level LSTM training (Phase B2)")
    print("  - Use business_*.parquet for business-level LSTM training (Phase B2)")
    print("  - Load vocabularies for token-to-index mapping during training")
    print("=" * 80)


if __name__ == "__main__":
    main()

