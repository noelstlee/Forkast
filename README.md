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

## 1. Data Preprocessing Pipeline

| Step | Command | Notes |
| --- | --- | --- |
| Create env | `python3 -m venv dva_env && source dva_env/bin/activate` | |
| Install deps | `pip install -r requirements.txt` | Python 3.12, ≥16 GB RAM recommended |
| Download raw data | From the [Google Local dataset](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/#complete-data): place `review-Georgia.json` and `meta-Georgia.json` inside `data/raw/`. |
| Generate processed artifacts | Use the scripts in `src/data/` (e.g., `python preprocess_lstm_atlanta.py`) or the all-in-one helpers in `preprocess_*.py` depending on the experiment. | Review each script’s flags before running; outputs live under `data/processed/`. |
| Export predictions CSV | `python generate_business_predictions_csv.py` produces `outputs/atlanta_business_predictions_with_meta.csv` for the map if you’re not using the shared download. |

## 2. Visualization Dashboard

### Required files

| File | Where to place | Source |
| --- | --- | --- |
| `outputs/atlanta_business_predictions_with_meta.csv` | `outputs/` | Download the latest build from [Drive](https://drive.google.com/file/d/1vCr1mpk47gX_fkZzSEno70bhkP7kSDi3/view?usp=sharing). |
| `outputs/meta-Georgia.json` | `outputs/` (API reads it directly) | Same [Google Local dataset link](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/#complete-data) as the raw files. |

### Run locally

```bash
# 1. API (FastAPI + DuckDB). First boot may take a few minutes:
uvicorn services.predictions_api:app --host 0.0.0.0 --port 9000 --reload

# 2. Open another terminal. Static assets (Leaflet/D3 front-end):
cd visualization
python3 -m http.server 8000

# 3. Visit the dashboard
open http://localhost:8000/dashboard.html
```

**Heads up:** the first API start loads the 2 GB CSV and builds metadata caches, so expect a delay before the health check turns green. Subsequent restarts reuse the warmed DuckDB table.
