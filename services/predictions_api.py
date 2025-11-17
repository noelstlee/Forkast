from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "outputs" / "atlanta_business_predictions_with_meta.csv"
META_PATH = REPO_ROOT / "outputs" / "meta-Georgia.json"
CACHE_ROOT = REPO_ROOT / "data" / "processed" / "predictions_cache"
VENUES_CACHE_PATH = CACHE_ROOT / "venues_index.json"
PREDICTIONS_CACHE_DIR = CACHE_ROOT / "predictions_by_source"

MAX_PREDICTIONS = 10

app = FastAPI(title="Forkast Predictions API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_duckdb_conn: duckdb.DuckDBPyConnection | None = None
_venue_lookup: Dict[int, Dict[str, Any]] | None = None
_meta_lookup: Dict[str, Dict[str, Any]] | None = None


def ensure_cache_dirs() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_meta_lookup() -> Dict[str, Dict[str, Any]]:
    """
    Load metadata from the newline-delimited meta-Georgia file and keep
    only the fields useful for the dashboard.
    """
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


def get_connection() -> duckdb.DuckDBPyConnection:
    global _duckdb_conn
    if _duckdb_conn is None:
        if not DATA_CSV.exists():
            raise FileNotFoundError(
                f"Prediction dataset not found at {DATA_CSV}. Please export it first."
            )
        conn = duckdb.connect(database=":memory:")
        conn.execute("PRAGMA threads=4")
        conn.execute("DROP TABLE IF EXISTS predictions")
        conn.execute(
            f"""
            CREATE TABLE predictions AS
            SELECT *
            FROM read_csv_auto(
                '{str(DATA_CSV)}',
                HEADER=TRUE,
                SAMPLE_SIZE=-1
            )
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_business ON predictions (source_business_idx)"
        )
        _duckdb_conn = conn
    return _duckdb_conn


def _fetch_dicts(sql: str, params: List[Any] | None = None) -> List[Dict[str, Any]]:
    conn = get_connection()
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


def build_venues_index() -> List[Dict[str, Any]]:
    global _venue_lookup
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
    rows = _fetch_dicts(query)
    venues: List[Dict[str, Any]] = []
    for row in rows:
        business_idx = _normalize_int(row["business_idx"])
        if business_idx is None:
            continue
        venues.append(
            {
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
            }
        )
    VENUES_CACHE_PATH.write_text(json.dumps(venues))
    _venue_lookup = None
    return venues


def get_venues_index() -> List[Dict[str, Any]]:
    ensure_cache_dirs()
    if VENUES_CACHE_PATH.exists():
        return json.loads(VENUES_CACHE_PATH.read_text())
    return build_venues_index()


def _cache_prediction_path(business_idx: int) -> Path:
    return PREDICTIONS_CACHE_DIR / f"{business_idx}.json"


def _fetch_predictions_from_db(business_idx: int) -> List[Dict[str, Any]]:
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
    rows = _fetch_dicts(query, [business_idx, MAX_PREDICTIONS])
    predictions: List[Dict[str, Any]] = []
    for row in rows:
        prediction_id = _normalize_int(row["prediction_id"])
        rank = _normalize_int(row["rank"])
        dest_business_idx = _normalize_int(row.get("dest_business_idx"))
        predictions.append(
            {
                "prediction_id": prediction_id,
                "rank": rank,
                "score_raw": _normalize_float(row.get("score_raw")),
                "score_share": _normalize_float(row.get("score_share")),
                "score_cum_share": _normalize_float(row.get("score_cum_share")),
                "score_z": _normalize_float(row.get("score_z")),
                "dest_business_idx": dest_business_idx,
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
            }
        )
    return predictions


def get_predictions_for_source(business_idx: int) -> List[Dict[str, Any]]:
    ensure_cache_dirs()
    cache_path = _cache_prediction_path(business_idx)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    predictions = _fetch_predictions_from_db(business_idx)
    cache_path.write_text(json.dumps(predictions))
    return predictions


def venue_lookup() -> Dict[int, Dict[str, Any]]:
    global _venue_lookup
    if _venue_lookup is None:
        _venue_lookup = {venue["business_idx"]: venue for venue in get_venues_index()}
    return _venue_lookup


@app.on_event("startup")
def bootstrap_cache() -> None:
    # Ensure DuckDB can open the CSV once at startup and warm the venues cache.
    get_connection()
    get_venues_index()
    load_meta_lookup()


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/venues")
def list_venues() -> List[Dict[str, Any]]:
    return get_venues_index()


@app.get("/venues/{business_idx}")
def get_venue(business_idx: int) -> Dict[str, Any]:
    venue = venue_lookup().get(business_idx)
    if not venue:
        raise HTTPException(status_code=404, detail="Venue not found")
    return attach_meta(venue)


@app.get("/venues/{business_idx}/predictions")
def get_predictions(business_idx: int) -> Dict[str, Any]:
    venue = venue_lookup().get(business_idx)
    if not venue:
        raise HTTPException(status_code=404, detail="Venue not found")
    predictions = get_predictions_for_source(business_idx)
    for pred in predictions:
        pred["dest_meta"] = get_meta_for_gmap(pred.get("dest_gmap_id"))
    return {
        "source": attach_meta(venue),
        "predictions": predictions,
        "metadata": {
            "max_results": MAX_PREDICTIONS,
            "count": len(predictions),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.predictions_api:app", host="0.0.0.0", port=8000, reload=True)

