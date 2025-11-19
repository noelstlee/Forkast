"""
Filter XGBoost predictions and business data to Atlanta metropolitan area.

This script prepares visualization-ready datasets for the Dash dashboard:
1. Filters businesses to Atlanta bounds
2. Filters predictions to Atlanta-only flows
3. Creates flow aggregations for visualization
4. Exports to Parquet and JSON formats

Input:
- models/predictions/xgboost_predictions.parquet
- data/processed/ga/xgboost_data/biz_ga.parquet

Output:
- data/viz/atlanta_businesses.parquet
- data/viz/atlanta_flows.parquet
- data/viz/atlanta_category_flows.parquet
- data/viz/atlanta_top_predictions.parquet
- data/viz/*.json (JSON exports for Dash)
"""

import polars as pl
import json
from pathlib import Path
from typing import Dict, List


# Atlanta metropolitan area bounds
ATLANTA_BOUNDS = {
    'lat_min': 33.6,   # Southern boundary
    'lat_max': 34.0,   # Northern boundary
    'lon_min': -84.6,  # Western boundary
    'lon_max': -84.2   # Eastern boundary
}


def filter_atlanta_businesses(biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter businesses to Atlanta metropolitan area.

    Args:
        biz_df: Business DataFrame with lat/lon columns

    Returns:
        Filtered DataFrame with only Atlanta businesses
    """
    print("\n" + "=" * 80)
    print("FILTERING BUSINESSES TO ATLANTA AREA")
    print("=" * 80)

    print(f"\nTotal businesses: {len(biz_df):,}")

    atlanta_biz = biz_df.filter(
        (pl.col('lat') >= ATLANTA_BOUNDS['lat_min']) &
        (pl.col('lat') <= ATLANTA_BOUNDS['lat_max']) &
        (pl.col('lon') >= ATLANTA_BOUNDS['lon_min']) &
        (pl.col('lon') <= ATLANTA_BOUNDS['lon_max'])
    )

    print(f"Atlanta businesses: {len(atlanta_biz):,}")
    print(f"Percentage kept: {len(atlanta_biz)/len(biz_df)*100:.1f}%")

    # Print category distribution
    print("\nTop 10 categories in Atlanta:")
    category_dist = atlanta_biz.group_by('category_main').agg(pl.len().alias('count')).sort('count', descending=True)
    for row in category_dist.head(10).iter_rows(named=True):
        print(f"  {row['category_main']:20s} {row['count']:4d}")

    return atlanta_biz


def filter_atlanta_predictions(preds_df: pl.DataFrame, atlanta_business_ids: set) -> pl.DataFrame:
    """
    Filter predictions to only Atlanta-to-Atlanta flows.

    Args:
        preds_df: Predictions DataFrame
        atlanta_business_ids: Set of business IDs in Atlanta

    Returns:
        Filtered predictions with only Atlanta flows
    """
    print("\n" + "=" * 80)
    print("FILTERING PREDICTIONS TO ATLANTA FLOWS")
    print("=" * 80)

    print(f"\nTotal predictions: {len(preds_df):,}")

    # Filter to flows where BOTH source and destination are in Atlanta
    atlanta_preds = preds_df.filter(
        pl.col('src_gmap_id').is_in(list(atlanta_business_ids)) &
        pl.col('dst_gmap_id').is_in(list(atlanta_business_ids))
    )

    print(f"Atlanta-only flows: {len(atlanta_preds):,}")
    print(f"Percentage kept: {len(atlanta_preds)/len(preds_df)*100:.1f}%")
    print(f"Unique queries: {atlanta_preds['query_id'].n_unique():,}")

    return atlanta_preds


def create_flow_aggregations(preds_df: pl.DataFrame, biz_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate predictions into business-to-business flows.

    Creates flow strength based on:
    - Number of predicted transitions
    - Average model scores
    - Distance statistics

    Args:
        preds_df: Predictions DataFrame
        biz_df: Business DataFrame

    Returns:
        Aggregated flows DataFrame
    """
    print("\n" + "=" * 80)
    print("CREATING FLOW AGGREGATIONS")
    print("=" * 80)

    # Aggregate by source-destination pairs
    flows = preds_df.group_by(['src_gmap_id', 'dst_gmap_id']).agg([
        pl.len().alias('flow_count'),
        pl.col('score').mean().alias('avg_score'),
        pl.col('distance_km').mean().alias('avg_distance'),
        pl.col('label').sum().alias('actual_visits'),
        pl.col('rank').filter(pl.col('label') == 1).mean().alias('avg_rank_when_actual')
    ])

    # Join with business info for source
    flows = flows.join(
        biz_df.select(['gmap_id', 'name', 'lat', 'lon', 'category_main', 'avg_rating']),
        left_on='src_gmap_id',
        right_on='gmap_id',
        how='left'
    ).rename({
        'name': 'src_name',
        'lat': 'src_lat',
        'lon': 'src_lon',
        'category_main': 'src_category',
        'avg_rating': 'src_rating'
    })

    # Join with business info for destination
    flows = flows.join(
        biz_df.select(['gmap_id', 'name', 'lat', 'lon', 'category_main', 'avg_rating']),
        left_on='dst_gmap_id',
        right_on='gmap_id',
        how='left'
    ).rename({
        'name': 'dst_name',
        'lat': 'dst_lat',
        'lon': 'dst_lon',
        'category_main': 'dst_category',
        'avg_rating': 'dst_rating'
    })

    # Sort by flow count
    flows = flows.sort('flow_count', descending=True)

    print(f"\nTotal flows: {len(flows):,}")
    print(f"Flows with actual visits: {(flows['actual_visits'] > 0).sum():,}")

    print("\nTop 10 strongest flows:")
    for i, row in enumerate(flows.head(10).iter_rows(named=True), 1):
        print(f"  {i:2d}. {row['src_name'][:25]:25s} → {row['dst_name'][:25]:25s} "
              f"({row['flow_count']:3d} flows, {row['avg_distance']:.1f} km)")

    return flows


def create_category_flows(preds_df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate flows by category transitions.

    Args:
        preds_df: Predictions DataFrame with category columns

    Returns:
        Category-level flows for Sankey diagram
    """
    print("\n" + "=" * 80)
    print("CREATING CATEGORY FLOW AGGREGATIONS")
    print("=" * 80)

    category_flows = preds_df.group_by(['src_category_main', 'dst_category_main']).agg([
        pl.len().alias('flow_count'),
        pl.col('score').mean().alias('avg_score'),
        pl.col('label').sum().alias('actual_visits'),
        pl.col('distance_km').mean().alias('avg_distance')
    ]).sort('flow_count', descending=True)

    print(f"\nTotal category transitions: {len(category_flows):,}")

    print("\nTop 10 category transitions:")
    for i, row in enumerate(category_flows.head(10).iter_rows(named=True), 1):
        same = "→" if row['src_category_main'] != row['dst_category_main'] else "↻"
        print(f"  {i:2d}. {row['src_category_main']:15s} {same} {row['dst_category_main']:15s} "
              f"({row['flow_count']:4d} flows)")

    return category_flows


def create_top_predictions_per_business(preds_df: pl.DataFrame, biz_df: pl.DataFrame, top_k: int = 10) -> pl.DataFrame:
    """
    For each business, get top-K predicted next destinations.

    Args:
        preds_df: Predictions DataFrame
        biz_df: Business DataFrame
        top_k: Number of top predictions to keep per business

    Returns:
        Top-K predictions per source business
    """
    print("\n" + "=" * 80)
    print(f"CREATING TOP-{top_k} PREDICTIONS PER BUSINESS")
    print("=" * 80)

    # Get average predictions per source business
    top_preds = preds_df.group_by(['src_gmap_id', 'dst_gmap_id']).agg([
        pl.col('score').mean().alias('avg_score'),
        pl.len().alias('prediction_count'),
        pl.col('label').sum().alias('actual_count')
    ])

    # Get top-K per source
    top_preds = top_preds.sort(['src_gmap_id', 'avg_score'], descending=[False, True])
    top_preds = top_preds.with_columns([
        pl.col('avg_score').rank(method='ordinal', descending=True).over('src_gmap_id').alias('rank')
    ])

    top_preds = top_preds.filter(pl.col('rank') <= top_k)

    # Join with business names
    top_preds = top_preds.join(
        biz_df.select(['gmap_id', 'name', 'category_main', 'avg_rating']),
        left_on='src_gmap_id',
        right_on='gmap_id',
        how='left'
    ).rename({
        'name': 'src_name',
        'category_main': 'src_category',
        'avg_rating': 'src_rating'
    })

    top_preds = top_preds.join(
        biz_df.select(['gmap_id', 'name', 'category_main', 'avg_rating']),
        left_on='dst_gmap_id',
        right_on='gmap_id',
        how='left'
    ).rename({
        'name': 'dst_name',
        'category_main': 'dst_category',
        'avg_rating': 'dst_rating'
    })

    print(f"\nTotal predictions: {len(top_preds):,}")
    print(f"Unique source businesses: {top_preds['src_gmap_id'].n_unique():,}")

    return top_preds


def export_to_json(df: pl.DataFrame, output_path: Path, orient: str = 'records'):
    """
    Export DataFrame to JSON format for Dash.

    Args:
        df: DataFrame to export
        output_path: Output JSON file path
        orient: JSON orientation ('records' or 'table')
    """
    # Convert to pandas for easier JSON export
    df_pandas = df.to_pandas()

    # Convert timestamps to strings
    for col in df_pandas.columns:
        if df_pandas[col].dtype == 'datetime64[ns]':
            df_pandas[col] = df_pandas[col].astype(str)

    # Convert list/array columns to native Python lists
    for col in df_pandas.columns:
        if df_pandas[col].dtype == 'object':
            # Check if first non-null value is a list or array
            first_val = df_pandas[col].dropna().iloc[0] if len(df_pandas[col].dropna()) > 0 else None
            if first_val is not None and hasattr(first_val, '__iter__') and not isinstance(first_val, str):
                df_pandas[col] = df_pandas[col].apply(lambda x: list(x) if x is not None and hasattr(x, '__iter__') else x)

    # Export to JSON
    if orient == 'records':
        data = df_pandas.to_dict(orient='records')
    else:
        data = df_pandas.to_dict(orient='list')

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✓ Exported to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def main():
    """Main filtering pipeline."""
    base_dir = Path(__file__).parent

    # Input paths
    predictions_path = base_dir / "models" / "predictions" / "xgboost_predictions.parquet"
    business_path = base_dir / "data" / "processed" / "ga" / "xgboost_data" / "biz_ga.parquet"

    # Output directory
    viz_dir = base_dir / "data" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ATLANTA DATA FILTERING FOR VISUALIZATION")
    print("=" * 80)
    print(f"\nAtlanta bounds:")
    print(f"  Latitude:  {ATLANTA_BOUNDS['lat_min']} to {ATLANTA_BOUNDS['lat_max']}")
    print(f"  Longitude: {ATLANTA_BOUNDS['lon_min']} to {ATLANTA_BOUNDS['lon_max']}")
    print(f"\nInput files:")
    print(f"  Predictions: {predictions_path}")
    print(f"  Businesses:  {business_path}")
    print(f"\nOutput directory: {viz_dir}")

    # Load data
    print("\n[Loading data...]")
    biz_df = pl.read_parquet(business_path)
    preds_df = pl.read_parquet(predictions_path)
    print(f"  ✓ Loaded {len(biz_df):,} businesses")
    print(f"  ✓ Loaded {len(preds_df):,} predictions")

    # 1. Filter businesses to Atlanta
    atlanta_biz = filter_atlanta_businesses(biz_df)
    atlanta_business_ids = set(atlanta_biz['gmap_id'].to_list())

    # 2. Filter predictions to Atlanta flows
    atlanta_preds = filter_atlanta_predictions(preds_df, atlanta_business_ids)

    # 3. Create flow aggregations
    flows = create_flow_aggregations(atlanta_preds, atlanta_biz)

    # 4. Create category flows
    category_flows = create_category_flows(atlanta_preds)

    # 5. Create top predictions per business
    top_preds = create_top_predictions_per_business(atlanta_preds, atlanta_biz, top_k=10)

    # Save Parquet files
    print("\n" + "=" * 80)
    print("SAVING PARQUET FILES")
    print("=" * 80)

    atlanta_biz.write_parquet(viz_dir / "atlanta_businesses.parquet", compression="snappy")
    print(f"  ✓ atlanta_businesses.parquet ({len(atlanta_biz):,} rows)")

    flows.write_parquet(viz_dir / "atlanta_flows.parquet", compression="snappy")
    print(f"  ✓ atlanta_flows.parquet ({len(flows):,} rows)")

    category_flows.write_parquet(viz_dir / "atlanta_category_flows.parquet", compression="snappy")
    print(f"  ✓ atlanta_category_flows.parquet ({len(category_flows):,} rows)")

    top_preds.write_parquet(viz_dir / "atlanta_top_predictions.parquet", compression="snappy")
    print(f"  ✓ atlanta_top_predictions.parquet ({len(top_preds):,} rows)")

    # Save JSON files for Dash
    print("\n" + "=" * 80)
    print("SAVING JSON FILES")
    print("=" * 80)

    export_to_json(atlanta_biz, viz_dir / "atlanta_businesses.json")
    export_to_json(flows.head(1000), viz_dir / "atlanta_flows_top1000.json")  # Top 1000 flows
    export_to_json(category_flows, viz_dir / "atlanta_category_flows.json")
    export_to_json(top_preds.head(5000), viz_dir / "atlanta_top_predictions.json")  # Sample

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary = {
        "atlanta_bounds": ATLANTA_BOUNDS,
        "total_businesses": len(atlanta_biz),
        "total_flows": len(flows),
        "total_predictions": len(atlanta_preds),
        "unique_queries": atlanta_preds['query_id'].n_unique(),
        "category_count": atlanta_biz['category_main'].n_unique(),
        "avg_flow_distance_km": flows['avg_distance'].mean(),
        "top_categories": category_flows.head(10).select(['src_category_main', 'dst_category_main', 'flow_count']).to_dicts()
    }

    with open(viz_dir / "atlanta_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAtlanta businesses: {summary['total_businesses']:,}")
    print(f"Atlanta flows: {summary['total_flows']:,}")
    print(f"Atlanta predictions: {summary['total_predictions']:,}")
    print(f"Unique categories: {summary['category_count']}")
    print(f"Average flow distance: {summary['avg_flow_distance_km']:.2f} km")

    print("\n✓✓✓ ATLANTA FILTERING COMPLETE ✓✓✓")
    print(f"\nVisualization-ready data saved to: {viz_dir}")
    print("\nFiles created:")
    print("  - atlanta_businesses.parquet / .json")
    print("  - atlanta_flows.parquet / .json")
    print("  - atlanta_category_flows.parquet / .json")
    print("  - atlanta_top_predictions.parquet / .json")
    print("  - atlanta_summary.json")


if __name__ == "__main__":
    main()