# Forkast: Georgia Restaurant Flow Prediction & Visualization

Predict restaurant visit flows for Georgia (full state) and surface interactive insights for Atlanta.

## Project Structure

```
Forkast/
├── data/
│   ├── raw/              # Google Local JSON dumps (reviews + metadata)
│   ├── processed/        # Parquet artifacts for GA + ATL
│   └── processed/predictions_cache/  # DuckDB + API caches
├── src/                  # Data prep + modeling scripts
├── notebooks/            # Exploration + evaluation
├── services/             # FastAPI service powering the map
├── visualization/        # Leaflet/D3 dashboard (static assets)
└── outputs/              # CSV used by the dashboard
```

## 1. Data Preprocessing & Model Training Pipeline

**Setup Environment:**

```bash
# Create virtual environment
python3 -m venv dva_env
source dva_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Download Raw Data:**

From the [Google Local dataset](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/#complete-data), download and place these files in `data/raw/`:
- `review-Georgia.json` (reviews dataset)
- `meta-Georgia.json` (business metadata)

Also download `meta-Georgia.json` to `outputs/` (needed for the visualization API).

---

### Step 1: XGBoost Pipeline

**1a. Preprocess Data for XGBoost**

```bash
python preprocess_xgboost.py
```

**What this does:**
- Phase A1: Ingests raw JSON data and normalizes it
- Phase A2: Generates restaurant pair transitions from user sequences
- Phase A2.5: Filters sequences by quality (minimum visits, time gaps)
- Phase A3: Engineers features (distance, time, ratings, categories, etc.)
- Phase A4: Splits data geographically (non-Atlanta train/val → Atlanta test)

**Output:** `data/processed/ga/xgboost_data/`
- `train.parquet`, `val.parquet`, `test.parquet` (feature-engineered pairs)
- `biz_ga.parquet` (business metadata)

**1b. Train XGBoost Model**

```bash
jupyter notebook notebooks/04_xgboost_full_pipeline.ipynb
```

**What this does:**
- Loads preprocessed train/val/test splits
- Trains XGBoost ranking model with `rank:pairwise` objective
- Evaluates using Recall@K, MRR, nDCG@K metrics
- Generates predictions for Atlanta test set
- Exports predictions to CSV format

**Output:** 
- Trained model (saved in notebook)
- `outputs/atlanta_xgboost_predictions_with_meta.csv` (generated in notebook)

---

### Step 2: LSTM Pipeline

**2a. Preprocess Data for LSTM (Atlanta-Specific)**

```bash
python preprocess_lstm_atlanta.py
```

**What this does:**
- Phase A1: Ingests raw JSON data
- Phase A2: Generates user-level sequences of restaurant visits
- Phase A2.5: Filters sequences by quality (5+ visits, >0.2 hour gaps)
- Phase A4: Splits data with Atlanta-specific strategy (non-Atlanta users train/val → Atlanta users test)
- Phase A5: LSTM-specific preprocessing (windowing, vocabularies, class weights)

**Output:** `data/processed/ga/lstm_data/`
- `train.parquet`, `val.parquet`, `test.parquet` (raw splits)
- `business_train/val/test.parquet` (windowed sequences)
- `category_train/val/test.parquet` (windowed category sequences)
- `business_vocab.json` (20,002 tokens)
- `category_vocab.json` (26 tokens)
- `category_class_weights.json` (for handling class imbalance)
- `atlanta_business_ids.json`
- `biz_ga.parquet`


**2b. Train LSTM Model**

```bash
jupyter notebook notebooks/lstm_business_training.ipynb
```

**What this does:**
- Loads windowed business sequences
- Trains 2-layer LSTM (256 hidden units, 128-dim embeddings)
- Uses Adam optimizer with early stopping
- Evaluates on Atlanta test set
- Generates top-K predictions

**Output:**
- `models/business_lstm/best_model.pt` (trained model checkpoint)
- `data/processed/ga/lstm_data/rebalanced/atlanta_business_predictions.parquet` (predictions)

**2c. Generate LSTM Predictions CSV**

```bash
python generate_business_predictions_csv.py
```

**What this does:**
- Loads LSTM predictions from Parquet
- Maps business indices back to Google Maps IDs using vocabulary
- Joins with business metadata
- Formats columns to match visualization API requirements
- Exports to CSV

**Output:** `outputs/atlanta_business_predictions_with_meta.csv`

---

### Step 3: Verify Outputs

Before running the dashboard, ensure these files exist:

```bash
# Check file sizes (should be ~1-2 GB each)
ls -lh outputs/atlanta_business_predictions_with_meta.csv
ls -lh outputs/atlanta_xgboost_predictions_with_meta.csv
ls -lh outputs/meta-Georgia.json
```

**Alternative: Download Pre-computed Predictions**

If you don't want to run the full pipeline, download pre-computed predictions:
- [LSTM predictions CSV](https://drive.google.com/file/d/1vCr1mpk47gX_fkZzSEno70bhkP7kSDi3/view?usp=sharing) → `outputs/atlanta_business_predictions_with_meta.csv`
- [XGBoost predictions CSV](https://drive.google.com/file/d/1yh2ZPqjEW61AJN3mF1wwVCOv5F-hvRdY/view?usp=sharing) → `outputs/atlanta_xgboost_predictions_with_meta.csv`
- [meta-Georgia.json](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/#complete-data) → `outputs/meta-Georgia.json`

---

## 2. Visualization Dashboard

### Required files

| File | Where to place | Source |
| --- | --- | --- |
| `outputs/atlanta_business_predictions_with_meta.csv` | `outputs/` | Download the latest build from [Drive](https://drive.google.com/file/d/1vCr1mpk47gX_fkZzSEno70bhkP7kSDi3/view?usp=sharing). |
| `outputs/atlanta_xgboost_predictions_with_meta.csv` | `outputs/` | Download the latest build from [Drive](https://drive.google.com/file/d/1yh2ZPqjEW61AJN3mF1wwVCOv5F-hvRdY/view?usp=sharing). |
| `outputs/meta-Georgia.json` | `outputs/` (API reads it directly) | Same [Google Local dataset link](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/#complete-data) as the raw files. |

### Run locally

```bash
# 1. API (FastAPI + DuckDB). First boot may take a few minutes:
uvicorn services.predictions_api_simple:app --host 127.0.0.1 --port 9000

# 2. Open another terminal. Static assets (Leaflet/D3 front-end):
cd visualization
python3 -m http.server 8000

# 3. Visit the dashboard
open http://localhost:8000/dashboard_simple.html
```
