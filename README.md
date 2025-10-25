# Forkast: Georgia Restaurant Visit Flow Prediction & Visualization

A data visualization project that predicts restaurant visit sequences using XGBoost and LSTM models, trained on Georgia Google Maps data and visualized for Atlanta.

## Project Structure

```
Forkast/
├── data/
│   ├── raw/                    # Raw JSON data (7.2GB reviews + 168MB metadata)
│   ├── processed/
│   │   ├── ga/                 # Full Georgia processed data (Parquet)
│   │   └── atl/                # Atlanta subset for visualization
├── src/
│   ├── data/                   # Data processing pipeline
│   ├── models/                 # Model training (XGBoost, LSTM)
│   ├── viz/                    # Dash dashboard
│   └── utils/                  # Utilities (geo, metrics, category mapping)
├── notebooks/                  # Jupyter notebooks for development
├── models/                     # Trained model artifacts
├── outputs/                    # Predictions for visualization
└── dashboard/                  # Static assets for Dash
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv dva_env
source dva_env/bin/activate  # On Windows: dva_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Files

Place the raw data files in `data/raw/`:
- `review-Georgia.json` (7.2GB)
- `meta-Georgia.json` (168MB)

## Pipeline Phases

### Phase A: Data Processing

**A1. Data Ingestion & Normalization**
```bash
python src/data/ingest.py
```
Or use the notebook: `notebooks/01_data_ingestion.ipynb`

Converts raw JSON to cleaned Parquet format with:
- Timestamp normalization
- Category mapping (~25 families)
- Geographic filtering (Georgia bounds)
- Deduplication

**A2. User Sequence Derivation**
```bash
python src/data/sequences.py
```
Creates user visit sequences and consecutive pairs.

**A3. Feature Engineering**
```bash
python src/data/features.py
```
Generates spatial, semantic, popularity, quality, price, and temporal features with hybrid negative sampling.

### Phase B: Model Training

**B1. XGBoost Ranking**
```bash
python src/models/xgboost_ranker.py
```
Trains business-level ranking model with Recall@K, MRR, nDCG evaluation.

**B2. LSTM Sequential Prediction**
```bash
python src/models/lstm_predictor.py
```
Trains both category-level and business-level sequence models.

### Phase C: Atlanta Filtering

```bash
python src/data/atlanta_filter.py
```
Filters data and predictions to Atlanta bounds and exports JSON for dashboard.

### Phase D: Visualization Dashboard

```bash
python src/viz/app.py
```
Launches interactive Dash dashboard with:
- Geo flow map (Mapbox)
- Category flow (Sankey diagram)
- Business panel with predictions

### Phase E: Evaluation

Model evaluation, ablation studies, and documentation generation.

## Key Features

- **Memory-efficient processing**: Uses Polars for streaming 7GB+ datasets
- **Hybrid negative sampling**: Combines random (10km radius) + Google's relative_results
- **Dual LSTM models**: Category-level (25 tokens) + Business-level (top-50k vocab)
- **Interactive visualization**: Dash dashboard with multiple views and filters
- **Comprehensive metrics**: Recall@K, MRR, nDCG@5, AUC

## Requirements

- Python 3.12+
- 16GB+ RAM recommended for full pipeline
- GPU recommended for LSTM training (optional, CPU works)

## Development

Use Jupyter notebooks in `notebooks/` for prototyping and exploration.
