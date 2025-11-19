from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[1]

# CSV paths for both models
LSTM_DATA_CSV = REPO_ROOT / "outputs" / "atlanta_business_predictions_with_meta.csv"
XGBOOST_DATA_CSV = REPO_ROOT / "outputs" / "atlanta_xgboost_predictions_with_meta.csv"

META_PATH = REPO_ROOT / "outputs" / "meta-Georgia.json"
CACHE_ROOT = REPO_ROOT / "data" / "processed" / "predictions_cache"
LSTM_VENUES_CACHE_PATH = CACHE_ROOT / "venues_index_lstm.json"
XGBOOST_VENUES_CACHE_PATH = CACHE_ROOT / "venues_index_xgboost.json"
PREDICTIONS_CACHE_DIR = CACHE_ROOT / "predictions_by_source"

MAX_PREDICTIONS = 10

app = FastAPI(title="Forkast Predictions API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Separate connections for each model
_lstm_conn: duckdb.DuckDBPyConnection | None = None
_xgboost_conn: duckdb.DuckDBPyConnection | None = None
_meta_lookup: Dict[str, Dict[str, Any]] | None = None


def ensure_cache_dirs() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_meta_lookup() -> Dict[str, Dict[str, Any]]:
    """Load metadata from the newline-delimited meta-Georgia file."""
    global _meta_lookup
    if _meta_lookup is not None:
        return _meta_lookup

    meta: Dict[str, Dict[str, Any]] = {}
    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gmap_id = payload.get("gmap_id")
                if not gmap_id:
                    continue
                meta[gmap_id] = {
                    "name": payload.get("name"),
                    "address": payload.get("address"),
                    "description": payload.get("description"),
                    "category": payload.get("category"),
                    "avg_rating": payload.get("avg_rating"),
                    "num_of_reviews": payload.get("num_of_reviews"),
                    "price": payload.get("price"),
                    "hours": payload.get("hours"),
                    "misc": payload.get("MISC"),
                    "state": payload.get("state"),
                    "url": payload.get("url"),
                    "relative_results": payload.get("relative_results"),
                }
    _meta_lookup = meta
    return _meta_lookup


def get_meta_for_gmap(gmap_id: str | None) -> Dict[str, Any] | None:
    if not gmap_id:
        return None
    return load_meta_lookup().get(gmap_id)


def attach_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(payload)
    enriched["meta"] = get_meta_for_gmap(payload.get("gmap_id"))
    return enriched


def get_lstm_connection() -> duckdb.DuckDBPyConnection:
    """Get connection for LSTM predictions."""
    global _lstm_conn
    if _lstm_conn is None:
        if not LSTM_DATA_CSV.exists():
            raise FileNotFoundError(f"LSTM predictions not found at {LSTM_DATA_CSV}")
        print("[LSTM] Loading predictions CSV...")
        conn = duckdb.connect(database=":memory:")
        conn.execute("PRAGMA threads=4")
        conn.execute(
            f"""
            CREATE TABLE predictions AS
            SELECT *
            FROM read_csv_auto('{str(LSTM_DATA_CSV)}', HEADER=TRUE, SAMPLE_SIZE=-1)
        """
        )
        conn.execute("CREATE INDEX idx_source ON predictions (source_business_idx)")
        print("[LSTM] Ready!")
        _lstm_conn = conn
    return _lstm_conn


def get_xgboost_connection() -> duckdb.DuckDBPyConnection:
    """Get connection for XGBoost predictions."""
    global _xgboost_conn
    if _xgboost_conn is None:
        if not XGBOOST_DATA_CSV.exists():
            raise FileNotFoundError(f"XGBoost predictions not found at {XGBOOST_DATA_CSV}")
        print("[XGBoost] Loading predictions CSV...")
        conn = duckdb.connect(database=":memory:")
        conn.execute("PRAGMA threads=4")
        conn.execute(
            f"""
            CREATE TABLE predictions AS
            SELECT *
            FROM read_csv_auto('{str(XGBOOST_DATA_CSV)}', HEADER=TRUE, SAMPLE_SIZE=-1)
        """
        )
        conn.execute("CREATE INDEX idx_source ON predictions (source_business_idx)")
        print("[XGBoost] Ready!")
        _xgboost_conn = conn
    return _xgboost_conn


def _fetch_dicts(sql: str, params: List[Any] | None = None, model: str = "lstm") -> List[Dict[str, Any]]:
    """Fetch results from the appropriate model connection."""
    conn = get_lstm_connection() if model == "lstm" else get_xgboost_connection()
    conn.execute(sql, params or [])
    columns = [col[0] for col in conn.description]
    return [dict(zip(columns, row)) for row in conn.fetchall()]


def _normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _normalize_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _apply_softmax(scores: List[float]) -> List[float]:
    """Apply softmax normalization to convert raw scores to probabilities."""
    if not scores:
        return []
    
    # Filter out None values and get valid scores
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return [None] * len(scores)
    
    # Subtract max for numerical stability
    max_score = max(valid_scores)
    exp_scores = [math.exp(s - max_score) for s in valid_scores]
    sum_exp = sum(exp_scores)
    
    # Calculate probabilities
    probabilities = [exp / sum_exp for exp in exp_scores]
    
    # Map back to original list (preserving None positions)
    result = []
    prob_idx = 0
    for s in scores:
        if s is not None:
            result.append(probabilities[prob_idx])
            prob_idx += 1
        else:
            result.append(None)
    
    return result


def build_venues_index(model: str = "lstm") -> List[Dict[str, Any]]:
    """Build venues index for a specific model."""
    print(f"[{model.upper()}] Building venues index...")
    
    query = """
        SELECT DISTINCT
            source_business_idx AS business_idx,
            source_gmap_id AS gmap_id,
            source_name AS name,
            source_lat AS lat,
            source_lon AS lon,
            source_category_main AS category_main,
            source_category_all AS category_all,
            source_avg_rating AS avg_rating,
            source_num_reviews AS num_reviews,
            source_price_bucket AS price_bucket,
            source_is_closed AS is_closed
        FROM predictions
        WHERE source_lat IS NOT NULL
          AND source_lon IS NOT NULL
    """
    
    rows = _fetch_dicts(query, model=model)
    print(f"[{model.upper()}] Found {len(rows)} unique venues")
    
    venues: List[Dict[str, Any]] = []
    for row in rows:
        business_idx = _normalize_int(row["business_idx"])
        if business_idx is None:
            continue
        venues.append({
            "business_idx": business_idx,
            "gmap_id": row["gmap_id"],
            "name": row["name"],
            "lat": _normalize_float(row["lat"]),
            "lon": _normalize_float(row["lon"]),
            "category_main": row.get("category_main"),
            "category_all": row.get("category_all"),
            "avg_rating": _normalize_float(row.get("avg_rating")),
            "num_reviews": _normalize_int(row.get("num_reviews")),
            "price_bucket": row.get("price_bucket"),
            "is_closed": _normalize_bool(row.get("is_closed")),
        })
    
    # Cache the venues
    cache_path = LSTM_VENUES_CACHE_PATH if model == "lstm" else XGBOOST_VENUES_CACHE_PATH
    cache_path.write_text(json.dumps(venues))
    print(f"[{model.upper()}] Cached {len(venues)} venues to {cache_path.name}")
    
    return venues


def get_venues_index(model: str = "lstm") -> List[Dict[str, Any]]:
    """Get venues index for a model, from cache if available."""
    ensure_cache_dirs()
    cache_path = LSTM_VENUES_CACHE_PATH if model == "lstm" else XGBOOST_VENUES_CACHE_PATH
    
    if cache_path.exists():
        print(f"[{model.upper()}] Loading venues from cache")
        return json.loads(cache_path.read_text())
    
    print(f"[{model.upper()}] Cache not found, building venues index...")
    return build_venues_index(model=model)


def _fetch_predictions_from_db(business_idx: int, model: str = "lstm") -> List[Dict[str, Any]]:
    """Fetch predictions for a business from the specified model."""
    query = """
        SELECT
            prediction_id,
            rank,
            score_raw,
            score_share,
            score_cum_share,
            score_z,
            dest_business_idx,
            dest_gmap_id,
            dest_name,
            dest_lat,
            dest_lon,
            dest_category_main,
            dest_category_all,
            dest_avg_rating,
            dest_num_reviews,
            dest_price_bucket,
            dest_is_closed,
            link_distance_km,
            same_main_category
        FROM predictions
        WHERE source_business_idx = ?
        ORDER BY rank ASC
        LIMIT ?
    """
    
    rows = _fetch_dicts(query, [business_idx, MAX_PREDICTIONS], model=model)
    
    predictions: List[Dict[str, Any]] = []
    for row in rows:
        predictions.append({
            "prediction_id": _normalize_int(row.get("prediction_id")),
            "rank": _normalize_int(row.get("rank")),
            "score_raw": _normalize_float(row.get("score_raw")),
            "score_share": _normalize_float(row.get("score_share")),
            "score_cum_share": _normalize_float(row.get("score_cum_share")),
            "score_z": _normalize_float(row.get("score_z")),
            "dest_business_idx": _normalize_int(row.get("dest_business_idx")),
            "dest_gmap_id": row.get("dest_gmap_id"),
            "dest_name": row.get("dest_name"),
            "dest_lat": _normalize_float(row.get("dest_lat")),
            "dest_lon": _normalize_float(row.get("dest_lon")),
            "dest_category_main": row.get("dest_category_main"),
            "dest_category_all": row.get("dest_category_all"),
            "dest_avg_rating": _normalize_float(row.get("dest_avg_rating")),
            "dest_num_reviews": _normalize_int(row.get("dest_num_reviews")),
            "dest_price_bucket": row.get("dest_price_bucket"),
            "dest_is_closed": _normalize_bool(row.get("dest_is_closed")),
            "link_distance_km": _normalize_float(row.get("link_distance_km")),
            "same_main_category": _normalize_bool(row.get("same_main_category")),
        })
    
    # Apply softmax normalization to XGBoost predictions to get higher percentages
    if model == "xgboost" and predictions:
        raw_scores = [pred.get("score_raw") for pred in predictions]
        softmax_probs = _apply_softmax(raw_scores)
        
        # Update score_share with softmax probabilities (as percentages)
        for i, prob in enumerate(softmax_probs):
            if prob is not None:
                predictions[i]["score_share"] = prob
        
        # Recalculate cumulative share
        cum_sum = 0.0
        for pred in predictions:
            if pred.get("score_share") is not None:
                cum_sum += pred["score_share"]
                pred["score_cum_share"] = cum_sum
    
    return predictions


@app.on_event("startup")
def bootstrap_cache() -> None:
    """Initialize connections and pre-build venue caches at startup."""
    print("[Startup] Initializing...")
    load_meta_lookup()
    print("[Startup] Metadata loaded")
    
    # Pre-build venue caches for both models
    print("[Startup] Building venue caches...")
    try:
        get_lstm_connection()
        if not LSTM_VENUES_CACHE_PATH.exists():
            print("[Startup] Building LSTM venue cache (first time only)...")
            build_venues_index("lstm")
        else:
            print("[Startup] LSTM venue cache exists")
    except Exception as e:
        print(f"[Startup] Warning: Could not initialize LSTM: {e}")
    
    try:
        get_xgboost_connection()
        if not XGBOOST_VENUES_CACHE_PATH.exists():
            print("[Startup] Building XGBoost venue cache (first time only)...")
            build_venues_index("xgboost")
        else:
            print("[Startup] XGBoost venue cache exists")
    except Exception as e:
        print(f"[Startup] Warning: Could not initialize XGBoost: {e}")
    
    print("[Startup] Ready!")


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/venues")
def list_venues(model: str = Query("lstm", regex="^(lstm|xgboost)$")) -> List[Dict[str, Any]]:
    """Get venues for the specified model."""
    return get_venues_index(model=model)


@app.get("/venues/{business_idx}")
def get_venue(
    business_idx: int,
    model: str = Query("lstm", regex="^(lstm|xgboost)$")
) -> Dict[str, Any]:
    """Get a specific venue."""
    venues = get_venues_index(model=model)
    venue = next((v for v in venues if v["business_idx"] == business_idx), None)
    if not venue:
        raise HTTPException(status_code=404, detail="Venue not found")
    return attach_meta(venue)


@app.get("/venues/{business_idx}/predictions")
def get_predictions(
    business_idx: int,
    model: str = Query("lstm", regex="^(lstm|xgboost)$")
) -> Dict[str, Any]:
    """Get predictions for a venue from the specified model."""
    venues = get_venues_index(model=model)
    venue = next((v for v in venues if v["business_idx"] == business_idx), None)
    if not venue:
        raise HTTPException(status_code=404, detail="Venue not found")
    
    predictions = _fetch_predictions_from_db(business_idx, model=model)
    
    for pred in predictions:
        pred["dest_meta"] = get_meta_for_gmap(pred.get("dest_gmap_id"))
    
    return {
        "source": attach_meta(venue),
        "predictions": predictions,
        "model": model,
        "metadata": {
            "max_results": MAX_PREDICTIONS,
            "count": len(predictions),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.predictions_api_simple:app", host="0.0.0.0", port=9000, reload=True)

