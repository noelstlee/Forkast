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

#### Option 1: Consolidated Pipelines (Recommended)

**All-in-One (XGBoost + LSTM + Cleanup)**:
```bash
python preprocess_all.py              # With cleanup prompt
python preprocess_all.py --cleanup    # Auto cleanup
python preprocess_all.py --no-cleanup # Skip cleanup
```
Runs complete pipeline: A1 → A2 → A2.5 → A3 → A4 → A5 + cleanup (~98 minutes)

Output: Both `xgboost_data/` and `lstm_data/` ready for training

**Individual Model Pipelines**:

For XGBoost only:
```bash
python preprocess_xgboost.py
```
Runs: A1 → A2 → A2.5 → A3 → A4 (~83 minutes)

For LSTM only:
```bash
python preprocess_lstm.py
```
Runs: A1 → A2 → A2.5 → A4 → A5 (~50 minutes)

**Cleanup Only**:
```bash
python cleanup_intermediate_files.py
```
Removes intermediate files (~1.82 GB freed)

#### Option 2: Individual Phase Scripts

**A1. Data Ingestion & Normalization**
```bash
python src/data/ingest.py
```
Converts raw JSON to cleaned Parquet format

**A2. User Sequence Derivation**
```bash
python src/data/sequences.py
```
Creates user visit sequences and consecutive pairs

**A2.5. Data Quality Filtering**
```bash
python src/data/filter_quality.py
```
Improves data quality (re-categorization, filtering)

**A3. Feature Engineering**
```bash
python src/data/features.py
```
Generates 47 features with hybrid negative sampling

**A4. Temporal Data Splitting**
```bash
python src/data/split_data.py
```
Splits data chronologically (70/15/15)

**A5. LSTM-Specific Preprocessing**
```bash
python src/data/lstm_preprocessing.py
```
Prepares LSTM data (vocabulary, windowing, padding)

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

## Mock Data for Visualization Team

Generate realistic mock data for parallel development while model training is in progress:

```bash
python generate_mock_data.py --size 1000 --businesses 100
```

**Output** (`data/mock/`):
- `businesses.parquet`: Business metadata (100 Atlanta restaurants)
- `flows.parquet`: User visit flows (1,000 consecutive visits)
- `xgboost_predictions.parquet`: XGBoost top-10 predictions (10,000 rows)
- `lstm_predictions.parquet`: LSTM category/business predictions (15,000 rows)
- `README.md`: Complete documentation and usage examples
- `*_sample.json`: JSON samples for easy inspection

**Features**:
- Realistic Atlanta locations (33.6-34.0°N, -84.6 to -84.2°W)
- Actual restaurant names (The Varsity, Mary Mac's, Fox Bros., etc.)
- Proper data structure matching real model outputs
- Configurable size for different testing needs
- ~0.32 MB total (lightweight for development)

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

## Documentation

- **[DATA_PREPROCESSING_DOCUMENTATION.md](DATA_PREPROCESSING_DOCUMENTATION.md)**: Comprehensive guide to all preprocessing phases, design decisions, intermediate results, and data lineage (1,230 lines)

## Development

Use Jupyter notebooks in `notebooks/` for prototyping and exploration.
