# Consolidated Preprocessing Scripts Guide

**Date**: October 26, 2025  
**Purpose**: Simplify preprocessing pipeline execution

---

## Overview

Instead of running 5+ individual scripts for preprocessing, we now have **2 consolidated scripts** that run the entire pipeline end-to-end:

1. **`preprocess_xgboost.py`** - Complete XGBoost preprocessing
2. **`preprocess_lstm.py`** - Complete LSTM preprocessing
3. **`cleanup_intermediate_files.py`** - Remove intermediate files

---

## Quick Start

### For XGBoost Training

```bash
# Run complete preprocessing pipeline
python preprocess_xgboost.py

# Expected time: ~83 minutes
# Output: data/processed/ga/xgboost_data/
```

### For LSTM Training

```bash
# Run complete preprocessing pipeline
python preprocess_lstm.py

# Expected time: ~50 minutes
# Output: data/processed/ga/lstm_data/
```

### Cleanup Intermediate Files

```bash
# Remove intermediate files (frees ~1.76 GB)
python cleanup_intermediate_files.py

# Keeps only final training files
```

---

## What Each Script Does

### `preprocess_xgboost.py`

**Pipeline**: A1 → A2 → A2.5 → A3 → A4

**Phases**:
1. **A1**: Data Ingestion & Normalization (~10 min)
   - Reads raw JSON files
   - Normalizes categories, timestamps, prices
   - Filters to Georgia bounds
   - Outputs: `biz_ga.parquet`, `reviews_ga.parquet`

2. **A2**: User Sequence Derivation (~5 min)
   - Creates user visit sequences
   - Generates consecutive visit pairs
   - Outputs: `user_sequences_ga.parquet`, `pairs_ga.parquet`

3. **A2.5**: Data Quality Filtering (~8 min)
   - Re-categorizes "restaurant" businesses
   - Filters to users with 5+ visits
   - Removes batch reviews (≤0.2h gaps)
   - Outputs: `user_sequences_filtered_ga.parquet`, `pairs_filtered_ga.parquet`

4. **A3**: Feature Engineering (~45 min)
   - Adds 47 features (spatial, temporal, quality, price, category)
   - Generates negative samples (3.1:1 ratio)
   - Outputs: `features_ga.parquet`

5. **A4**: Temporal Data Splitting (~3 min)
   - Splits chronologically (70/15/15)
   - Outputs: `xgboost_data/train.parquet`, `val.parquet`, `test.parquet`

**Final Output**:
```
data/processed/ga/xgboost_data/
├── train.parquet (117 MB) - 2.85M samples
├── val.parquet (25 MB) - 610K samples
├── test.parquet (25 MB) - 610K samples
└── biz_ga.parquet (4.8 MB) - Business metadata
```

---

### `preprocess_lstm.py`

**Pipeline**: A1 → A2 → A2.5 → A4 → A5

**Phases**:
1. **A1**: Data Ingestion & Normalization (~10 min)
   - Same as XGBoost

2. **A2**: User Sequence Derivation (~5 min)
   - Same as XGBoost

3. **A2.5**: Data Quality Filtering (~8 min)
   - Sequences only (no pairs)
   - Filters to users with 5+ visits

4. **A4**: Temporal Data Splitting (~3 min)
   - LSTM only (by user's last visit)
   - Splits users 70/15/15

5. **A5**: LSTM-Specific Preprocessing (~12 min)
   - Builds vocabularies (category: 26, business: 20K)
   - Applies sequence windowing (6.73M examples)
   - Pads/truncates to max length 20

**Final Output**:
```
data/processed/ga/lstm_data/
├── category_train.parquet (27 MB) - 3.86M examples
├── category_val.parquet (8 MB) - 1.20M examples
├── category_test.parquet (11 MB) - 1.66M examples
├── business_train.parquet (27 MB) - 3.86M examples
├── business_val.parquet (8 MB) - 1.20M examples
├── business_test.parquet (10 MB) - 1.66M examples
├── category_vocab.json (<1 KB) - 26 tokens
├── business_vocab.json (964 KB) - 20,002 tokens
└── biz_ga.parquet (4.8 MB) - Business metadata
```

---

### `cleanup_intermediate_files.py`

**Purpose**: Remove intermediate preprocessing files to save disk space

**Files Removed** (~1.76 GB):
- `reviews_ga.parquet` (787 MB)
- `user_sequences_ga.parquet` (257 MB)
- `user_sequences_filtered_ga.parquet` (155 MB)
- `pairs_ga.parquet` (194 MB)
- `pairs_filtered_ga.parquet` (50 MB)
- `features_ga.parquet` (199 MB)
- `lstm_data/train.parquet` (90 MB)
- `lstm_data/val.parquet` (27 MB)
- `lstm_data/test.parquet` (37 MB)

**Files Kept** (263.5 MB):
- All files in `xgboost_data/`
- `lstm_data/category_*.parquet`
- `lstm_data/business_*.parquet`
- `lstm_data/*_vocab.json`
- `biz_ga.parquet` (root and copies)

**Usage**:
```bash
python cleanup_intermediate_files.py

# Interactive prompt:
# "Proceed with cleanup? (yes/no): "
```

---

## Comparison: Old vs. New Workflow

### Old Workflow (5+ Commands)

```bash
# Step 1: Ingestion
python src/data/ingest.py

# Step 2: Sequences
python src/data/sequences.py

# Step 3: Quality filtering
python src/data/filter_quality.py

# Step 4: Feature engineering (XGBoost only)
python src/data/features.py

# Step 5: Splitting
python src/data/split_data.py

# Step 6: LSTM preprocessing (LSTM only)
python src/data/lstm_preprocessing.py

# Total: 6 commands, easy to forget steps
```

### New Workflow (1 Command)

```bash
# For XGBoost:
python preprocess_xgboost.py

# For LSTM:
python preprocess_lstm.py

# Total: 1 command per model type
```

---

## Benefits

### ✅ Simplicity
- **Before**: Run 5-6 scripts in correct order
- **After**: Run 1 script

### ✅ Reproducibility
- Single script = entire pipeline
- No risk of missing steps
- Consistent preprocessing

### ✅ Efficiency
- Automatic pipeline execution
- No manual intervention needed
- Progress tracking included

### ✅ Disk Space
- Cleanup script removes 1.76 GB
- Keeps only essential files
- Optimized for training

### ✅ Onboarding
- New team members: just run one script
- Clear documentation
- Easy to understand

---

## Technical Details

### How It Works

Both consolidated scripts import the individual phase modules:

```python
from src.data import ingest, sequences, filter_quality, features, split_data, lstm_preprocessing

def main():
    ingest.main()
    sequences.main()
    filter_quality.main()
    # ... etc
```

### Modified Functions

We added helper functions to the individual scripts:

**`split_data.py`**:
- `split_xgboost_only()` - Split only XGBoost data
- `split_lstm_only()` - Split only LSTM data

**`filter_quality.py`**:
- `filter_sequences_only()` - Filter only sequences (for LSTM)

These allow the consolidated scripts to run only the necessary parts of each phase.

---

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure you're in the project root directory:
```bash
cd /Users/istantheman/Forkast
python preprocess_xgboost.py
```

### Issue: Missing raw data files

**Error**: `FileNotFoundError: data/raw/meta-Georgia.json`

**Solution**: Place raw data files in `data/raw/`:
- `review-Georgia.json` (7.2 GB)
- `meta-Georgia.json` (168 MB)

### Issue: Out of memory

**Error**: `MemoryError` or system slowdown

**Solution**: 
- Close other applications
- Recommended: 16GB+ RAM
- Consider running on a server

### Issue: Cleanup script removes wrong files

**Solution**: The script asks for confirmation before deleting. Review the list carefully before typing "yes".

---

## Advanced Usage

### Running Individual Phases

If you need to run only specific phases, use the individual scripts:

```bash
# Just ingestion
python src/data/ingest.py

# Just feature engineering
python src/data/features.py

# etc.
```

### Customizing Parameters

Edit the consolidated scripts to change parameters:

```python
# In preprocess_lstm.py
MAX_SEQ_LEN = 30  # Change from 20 to 30
TOP_K_BUSINESSES = 30000  # Change from 20K to 30K
```

### Debugging

Add `--debug` flag (if implemented) or add print statements:

```python
# In preprocess_xgboost.py
print(f"Debug: Loaded {len(df)} rows")
```

---

## File Structure After Preprocessing

```
Forkast/
├── data/
│   ├── raw/
│   │   ├── meta-Georgia.json (168 MB)
│   │   └── review-Georgia.json (7.2 GB)
│   └── processed/
│       └── ga/
│           ├── biz_ga.parquet (4.8 MB) ✓ KEEP
│           ├── xgboost_data/
│           │   ├── train.parquet (117 MB) ✓ USE
│           │   ├── val.parquet (25 MB) ✓ USE
│           │   ├── test.parquet (25 MB) ✓ USE
│           │   └── biz_ga.parquet (4.8 MB) ✓ USE
│           └── lstm_data/
│               ├── category_train.parquet (27 MB) ✓ USE
│               ├── category_val.parquet (8 MB) ✓ USE
│               ├── category_test.parquet (11 MB) ✓ USE
│               ├── business_train.parquet (27 MB) ✓ USE
│               ├── business_val.parquet (8 MB) ✓ USE
│               ├── business_test.parquet (10 MB) ✓ USE
│               ├── category_vocab.json (<1 KB) ✓ USE
│               ├── business_vocab.json (964 KB) ✓ USE
│               └── biz_ga.parquet (4.8 MB) ✓ USE
├── preprocess_xgboost.py ✓ NEW
├── preprocess_lstm.py ✓ NEW
└── cleanup_intermediate_files.py ✓ NEW
```

---

## Next Steps After Preprocessing

### 1. Verify Output Files

```bash
ls -lh data/processed/ga/xgboost_data/
ls -lh data/processed/ga/lstm_data/
```

### 2. Clean Up (Optional)

```bash
python cleanup_intermediate_files.py
```

### 3. Start Model Training

**XGBoost**:
```bash
python src/models/xgboost_ranker.py
```

**LSTM**:
```bash
python src/models/lstm_predictor.py
```

---

## FAQ

**Q: Can I run both preprocessing scripts at once?**  
A: No, they both process the same raw data. Run them sequentially or on different machines.

**Q: How long does preprocessing take?**  
A: XGBoost: ~83 minutes, LSTM: ~50 minutes (on recommended hardware)

**Q: Can I resume if preprocessing fails?**  
A: Currently no. You'll need to restart. Consider adding checkpointing if needed.

**Q: Do I need to run both scripts?**  
A: Only if you want to train both models. Run only the one you need.

**Q: What if I want to change preprocessing parameters?**  
A: Edit the individual phase scripts in `src/data/` or modify the consolidated scripts.

---

## Summary

| Script | Purpose | Time | Output Size |
|--------|---------|------|-------------|
| `preprocess_xgboost.py` | XGBoost pipeline | ~83 min | 172.5 MB |
| `preprocess_lstm.py` | LSTM pipeline | ~50 min | 91 MB |
| `cleanup_intermediate_files.py` | Remove intermediates | <1 min | Frees 1.76 GB |

**Total final size**: 263.5 MB (from 7.4 GB raw data = 28:1 compression)

---

**Ready to preprocess?** 🚀

```bash
python preprocess_xgboost.py  # or
python preprocess_lstm.py
```

