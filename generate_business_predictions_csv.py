#!/usr/bin/env python3
"""Generate a visualization-ready CSV of LSTM business predictions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import polars as pl


EARTH_RADIUS_KM = 6371.0088
SPECIAL_TOKENS = {"<PAD>", "<UNK>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine LSTM predictions with business metadata into a flat CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("data/processed/ga/lstm_data/rebalanced/atlanta_business_predictions.parquet"),
        help="Parquet file containing LSTM business predictions",
    )
    parser.add_argument(
        "--business-vocab-path",
        type=Path,
        default=Path("data/processed/ga/lstm_data/business_vocab.json"),
        help="JSON vocabulary mapping Google Maps IDs to integer indices",
    )
    parser.add_argument(
        "--biz-meta-path",
        type=Path,
        default=Path("data/processed/ga/biz_ga.parquet"),
        help="Parquet file containing business metadata for Georgia",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/atlanta_business_predictions_with_meta.csv"),
        help="Destination CSV path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top predictions per business to keep",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum raw prediction score required to keep a row",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Keep rows where the source business resolves to <UNK>/<PAD>",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "gzip"],
        default="none",
        help="CSV compression to apply at write time",
    )
    return parser.parse_args()


def load_business_vocab(path: Path) -> pl.DataFrame:
    """Return a mapping of integer business indices to Google Maps IDs."""

    if not path.exists():
        raise FileNotFoundError(f"Business vocabulary not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        vocab_data = json.load(fh)

    pairs: list[tuple[int, str]] = []
    for gmap_id, index in vocab_data.items():
        try:
            idx_int = int(index)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid index '{index}' for gmap_id '{gmap_id}'") from exc
        pairs.append((idx_int, gmap_id))

    pairs.sort(key=lambda item: item[0])

    return pl.DataFrame(
        {
            "business_idx": [p[0] for p in pairs],
            "gmap_id": [p[1] for p in pairs],
        }
    )


def build_prediction_pairs(
    predictions_path: Path,
    top_k: int,
    min_score: float,
) -> pl.DataFrame:
    """Explode the predictions parquet into a row-per-destination table."""

    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction parquet not found: {predictions_path}")

    predictions = (
        pl.read_parquet(predictions_path)
        .with_row_index("prediction_id")
        .rename({"target_business_idx": "source_business_idx"})
        .explode(["predicted_business_indices", "prediction_scores"])
        .rename(
            {
                "predicted_business_indices": "dest_business_idx",
                "prediction_scores": "score_raw",
            }
        )
        .with_columns(
            [
                pl.col("prediction_id").cum_count().over("prediction_id").alias("rank"),
                pl.col("score_raw").cast(pl.Float64()),
                pl.sum("score_raw").over("prediction_id").alias("score_sum"),
                pl.mean("score_raw").over("prediction_id").alias("score_mean"),
                pl.col("score_raw").std(ddof=0).over("prediction_id").alias("score_std"),
            ]
        )
    )

    if top_k > 0:
        predictions = predictions.filter(pl.col("rank") <= top_k)

    if min_score > 0:
        predictions = predictions.filter(pl.col("score_raw") >= min_score)

    predictions = predictions.with_columns(
        [
            pl.when(pl.col("score_sum") > 0)
            .then(pl.col("score_raw") / pl.col("score_sum"))
            .otherwise(0.0)
            .alias("score_share"),
        ]
    ).with_columns(
        [
            pl.col("score_share")
            .cum_sum()
            .over("prediction_id")
            .alias("score_cum_share"),
            pl.when(pl.col("score_std") > 0)
            .then((pl.col("score_raw") - pl.col("score_mean")) / pl.col("score_std"))
            .otherwise(None)
            .alias("score_z"),
        ]
    )

    predictions = predictions.with_columns(
        [
            pl.col("rank").cast(pl.Int32),
            pl.col("score_share").round(9),
            pl.col("score_cum_share").round(9),
            pl.col("score_z").round(6),
        ]
    )

    return predictions.drop(["score_sum", "score_mean", "score_std"])


def prepare_business_metadata(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Business metadata parquet not found: {path}")

    return pl.read_parquet(path).select(
        [
            pl.col("gmap_id"),
            pl.col("name"),
            pl.col("lat").cast(pl.Float64()),
            pl.col("lon").cast(pl.Float64()),
            pl.col("category_main"),
            pl.col("category_all").list.join("|").alias("category_all"),
            pl.col("avg_rating").cast(pl.Float64()),
            pl.col("num_reviews"),
            pl.col("price_bucket").cast(pl.Float64()),
            pl.col("is_closed"),
        ]
    )


def rename_metadata(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return df.rename(
        {
            "gmap_id": f"{prefix}_gmap_id",
            "name": f"{prefix}_name",
            "lat": f"{prefix}_lat",
            "lon": f"{prefix}_lon",
            "category_main": f"{prefix}_category_main",
            "category_all": f"{prefix}_category_all",
            "avg_rating": f"{prefix}_avg_rating",
            "num_reviews": f"{prefix}_num_reviews",
            "price_bucket": f"{prefix}_price_bucket",
            "is_closed": f"{prefix}_is_closed",
        }
    )


def haversine_km_expr(lat1: str, lon1: str, lat2: str, lon2: str) -> pl.Expr:
    lat1_rad = pl.col(lat1).cast(pl.Float64()).radians()
    lon1_rad = pl.col(lon1).cast(pl.Float64()).radians()
    lat2_rad = pl.col(lat2).cast(pl.Float64()).radians()
    lon2_rad = pl.col(lon2).cast(pl.Float64()).radians()

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a_expr = (dlat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)
    central_angle = pl.min_horizontal(pl.lit(1.0), a_expr).sqrt().arcsin() * 2

    return (EARTH_RADIUS_KM * central_angle).alias("link_distance_km")


def add_metadata(
    pairs: pl.DataFrame,
    vocab_df: pl.DataFrame,
    biz_meta: pl.DataFrame,
    include_unknown: bool,
) -> pl.DataFrame:
    source_map = vocab_df.rename(
        {
            "business_idx": "source_business_idx",
            "gmap_id": "source_gmap_id",
        }
    )
    dest_map = vocab_df.rename(
        {
            "business_idx": "dest_business_idx",
            "gmap_id": "dest_gmap_id",
        }
    )

    enriched = (
        pairs.join(source_map, on="source_business_idx", how="left")
        .join(dest_map, on="dest_business_idx", how="left")
    )

    if not include_unknown:
        enriched = enriched.filter(
            (~pl.col("source_gmap_id").is_null())
            & (~pl.col("source_gmap_id").is_in(list(SPECIAL_TOKENS)))
        )

    source_meta = rename_metadata(biz_meta, "source")
    dest_meta = rename_metadata(biz_meta, "dest")

    enriched = enriched.join(source_meta, on="source_gmap_id", how="left").join(
        dest_meta, on="dest_gmap_id", how="left"
    )

    enriched = enriched.filter(
        (~pl.col("dest_gmap_id").is_null())
        & (~pl.col("dest_gmap_id").is_in(list(SPECIAL_TOKENS)))
    )

    if enriched.is_empty():
        return enriched

    enriched = enriched.with_columns(
        [
            pl.col("prediction_id").cum_count().over("prediction_id").alias("rank"),
            pl.sum("score_raw").over("prediction_id").alias("score_sum"),
            pl.mean("score_raw").over("prediction_id").alias("score_mean"),
            pl.col("score_raw").std(ddof=0).over("prediction_id").alias("score_std"),
        ]
    )

    enriched = enriched.with_columns(
        [
            pl.when(pl.col("score_sum") > 0)
            .then(pl.col("score_raw") / pl.col("score_sum"))
            .otherwise(0.0)
            .alias("score_share"),
            pl.col("score_share")
            .cum_sum()
            .over("prediction_id")
            .alias("score_cum_share"),
            pl.when(pl.col("score_std") > 0)
            .then((pl.col("score_raw") - pl.col("score_mean")) / pl.col("score_std"))
            .otherwise(None)
            .alias("score_z"),
        ]
    ).with_columns(
        [
            pl.col("rank").cast(pl.Int32),
            pl.col("score_share").round(9),
            pl.col("score_cum_share").round(9),
            pl.col("score_z").round(6),
        ]
    )

    enriched = enriched.drop(["score_sum", "score_mean", "score_std"])

    enriched = enriched.with_columns(
        [
            pl.col("source_name").is_not_null().alias("source_has_metadata"),
            pl.col("dest_name").is_not_null().alias("dest_has_metadata"),
            (pl.col("source_business_idx") == pl.col("dest_business_idx")).alias(
                "is_self_prediction"
            ),
            pl.when(
                pl.col("source_category_main").is_not_null()
                & pl.col("dest_category_main").is_not_null()
            )
            .then(pl.col("source_category_main") == pl.col("dest_category_main"))
            .otherwise(False)
            .alias("same_main_category"),
            pl.concat_str(
                [
                    pl.col("source_category_main").fill_null("unknown"),
                    pl.lit("→"),
                    pl.col("dest_category_main").fill_null("unknown"),
                ]
            ).alias("category_pair_key"),
            haversine_km_expr(
                "source_lat",
                "source_lon",
                "dest_lat",
                "dest_lon",
            ),
        ]
    )

    return enriched


def select_column_order(df: pl.DataFrame) -> pl.DataFrame:
    ordered_cols: Iterable[str] = [
        "prediction_id",
        "rank",
        "score_raw",
        "score_share",
        "score_cum_share",
        "score_z",
        "source_business_idx",
        "source_gmap_id",
        "source_name",
        "source_lat",
        "source_lon",
        "source_category_main",
        "source_category_all",
        "source_avg_rating",
        "source_num_reviews",
        "source_price_bucket",
        "source_is_closed",
        "dest_business_idx",
        "dest_gmap_id",
        "dest_name",
        "dest_lat",
        "dest_lon",
        "dest_category_main",
        "dest_category_all",
        "dest_avg_rating",
        "dest_num_reviews",
        "dest_price_bucket",
        "dest_is_closed",
        "link_distance_km",
        "same_main_category",
        "is_self_prediction",
        "source_has_metadata",
        "dest_has_metadata",
        "category_pair_key",
    ]

    extra_cols = [col for col in df.columns if col not in ordered_cols]
    return df.select([*(col for col in ordered_cols if col in df.columns), *extra_cols])


def main() -> None:
    args = parse_args()

    pairs = build_prediction_pairs(args.predictions_path, args.top_k, args.min_score)
    vocab_df = load_business_vocab(args.business_vocab_path)
    biz_meta = prepare_business_metadata(args.biz_meta_path)

    enriched = add_metadata(pairs, vocab_df, biz_meta, args.include_unknown)
    enriched = select_column_order(enriched)

    output_path = args.output_path
    if args.compression == "gzip" and not output_path.name.endswith(".gz"):
        output_path = output_path.with_name(output_path.name + ".gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.compression == "gzip":
        with gzip.open(output_path, "wt", encoding="utf-8", newline="") as fh:
            enriched.write_csv(fh)
    else:
        enriched.write_csv(output_path)

    total_rows = enriched.height
    unique_sources = enriched.select(pl.col("source_business_idx").n_unique()).item()
    unique_destinations = enriched.select(pl.col("dest_business_idx").n_unique()).item()

    print(f"Wrote {total_rows:,} rows to {output_path}")
    print(f"Unique source businesses: {unique_sources:,}")
    print(f"Unique destination businesses: {unique_destinations:,}")


if __name__ == "__main__":
    main()

