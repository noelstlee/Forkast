# Visualization Team Quick Start Guide

This guide helps the visualization team get started with mock data while model training is in progress.

## Quick Start

### 1. Generate Mock Data

```bash
# Activate virtual environment
source dva_env/bin/activate  # On Windows: dva_env\Scripts\activate

# Generate mock data (default: 1000 flows, 100 businesses)
python generate_mock_data.py

# Or customize the size
python generate_mock_data.py --size 2000 --businesses 200
```

### 2. Explore the Data

All data is in `data/mock/`:

```
data/mock/
├── businesses.parquet          # Business metadata
├── flows.parquet               # User visit flows
├── xgboost_predictions.parquet # XGBoost predictions
├── lstm_predictions.parquet    # LSTM predictions
├── README.md                   # Detailed documentation
└── *_sample.json              # JSON samples for inspection
```

### 3. Load Data in Python

#### Using Polars (Recommended)

```python
import polars as pl

# Load all datasets
businesses = pl.read_parquet("data/mock/businesses.parquet")
flows = pl.read_parquet("data/mock/flows.parquet")
xgb_predictions = pl.read_parquet("data/mock/xgboost_predictions.parquet")
lstm_predictions = pl.read_parquet("data/mock/lstm_predictions.parquet")

# Quick exploration
print(f"Businesses: {len(businesses)}")
print(f"Flows: {len(flows)}")
print(f"XGBoost predictions: {len(xgb_predictions)}")
print(f"LSTM predictions: {len(lstm_predictions)}")
```

#### Using Pandas

```python
import pandas as pd

# Load all datasets
businesses = pd.read_parquet("data/mock/businesses.parquet")
flows = pd.read_parquet("data/mock/flows.parquet")
xgb_predictions = pd.read_parquet("data/mock/xgboost_predictions.parquet")
lstm_predictions = pd.read_parquet("data/mock/lstm_predictions.parquet")
```

---

## Data Structure Overview

### 1. `businesses.parquet` (Business Metadata)

| Column | Type | Description |
|--------|------|-------------|
| `gmap_id` | str | Unique Google Maps ID |
| `name` | str | Restaurant name |
| `lat`, `lon` | float | Coordinates (Atlanta: 33.6-34.0°N, -84.6 to -84.2°W) |
| `category_main` | str | Primary food category |
| `category_all` | list[str] | All categories |
| `avg_rating` | float | Average rating (1-5) |
| `num_reviews` | int | Number of reviews |
| `price_bucket` | int | Price level (1-4, or null) |
| `is_closed` | bool | Whether business is closed |
| `relative_results` | list[str] | Similar businesses |

**Example:**
```python
# Get all pizza places
pizza_places = businesses.filter(pl.col("category_main") == "pizza")

# Get highly rated restaurants (4.5+)
top_rated = businesses.filter(pl.col("avg_rating") >= 4.5)
```

### 2. `flows.parquet` (User Visit Flows)

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | Anonymous user ID |
| `src_gmap_id` | str | Source business ID |
| `dst_gmap_id` | str | Destination business ID |
| `src_ts` | datetime | Source visit timestamp |
| `dst_ts` | datetime | Destination visit timestamp |
| `delta_hours` | float | Time between visits (hours) |
| `src_category` | str | Source category |
| `dst_category` | str | Destination category |
| `src_lat`, `src_lon` | float | Source coordinates |
| `dst_lat`, `dst_lon` | float | Destination coordinates |
| `distance_km` | float | Distance between visits (km) |

**Example:**
```python
# Get flows from pizza to burger
pizza_to_burger = flows.filter(
    (pl.col("src_category") == "pizza") &
    (pl.col("dst_category") == "burger")
)

# Get flows within 5km
nearby_flows = flows.filter(pl.col("distance_km") <= 5)
```

### 3. `xgboost_predictions.parquet` (XGBoost Predictions)

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | User ID |
| `src_gmap_id` | str | Source business |
| `dst_gmap_id` | str | Predicted destination |
| `dst_name` | str | Predicted business name |
| `dst_category` | str | Predicted category |
| `dst_rating` | float | Predicted business rating |
| `score` | float | Prediction score (0-1) |
| `rank` | int | Rank in top-10 (1-10) |
| `is_actual` | bool | Whether this was the actual next visit |
| `model` | str | Model name ("xgboost") |

**Example:**
```python
# Get top-3 predictions for a specific flow
first_flow = flows[0]
top_3 = xgb_predictions.filter(
    (pl.col("user_id") == first_flow["user_id"]) &
    (pl.col("src_gmap_id") == first_flow["src_gmap_id"])
).sort("rank").head(3)

# Calculate Recall@5
recall_at_5 = xgb_predictions.filter(
    (pl.col("rank") <= 5) & 
    (pl.col("is_actual") == True)
).height / flows.height
print(f"Recall@5: {recall_at_5:.2%}")
```

### 4. `lstm_predictions.parquet` (LSTM Predictions)

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | str | User ID |
| `prediction_type` | str | "category" or "business" |
| `predicted_category` | str | Predicted food category |
| `predicted_business_id` | str | Predicted business ID (business-level only) |
| `predicted_business_name` | str | Predicted business name (business-level only) |
| `probability` | float | Prediction probability (softmax output) |
| `rank` | int | Rank in top-K |
| `is_actual` | bool | Whether this was the actual next visit/category |
| `model` | str | Model name ("lstm_category" or "lstm_business") |

**Example:**
```python
# Get category-level predictions
category_preds = lstm_predictions.filter(
    pl.col("prediction_type") == "category"
)

# Get business-level predictions
business_preds = lstm_predictions.filter(
    pl.col("prediction_type") == "business"
)

# Get top-1 category prediction for each flow
top_1_categories = category_preds.filter(pl.col("rank") == 1)
```

---

## Visualization Examples

### 1. Geographic Flow Map (Mapbox)

```python
import plotly.graph_objects as go

# Prepare data for flow lines
flow_data = []
for row in flows.iter_rows(named=True):
    flow_data.append({
        'src_lat': row['src_lat'],
        'src_lon': row['src_lon'],
        'dst_lat': row['dst_lat'],
        'dst_lon': row['dst_lon'],
        'category': row['dst_category'],
    })

# Create Mapbox figure
fig = go.Figure()

# Add flow lines
for flow in flow_data:
    fig.add_trace(go.Scattermapbox(
        lon=[flow['src_lon'], flow['dst_lon']],
        lat=[flow['src_lat'], flow['dst_lat']],
        mode='lines',
        line=dict(width=2, color='blue'),
        opacity=0.5,
        hoverinfo='skip',
    ))

# Add business markers
fig.add_trace(go.Scattermapbox(
    lon=businesses['lon'],
    lat=businesses['lat'],
    mode='markers',
    marker=dict(size=8, color='red'),
    text=businesses['name'],
    hoverinfo='text',
))

# Update layout
fig.update_layout(
    mapbox=dict(
        style='open-street-map',
        center=dict(lat=33.8, lon=-84.4),  # Atlanta center
        zoom=10,
    ),
    height=600,
    margin=dict(l=0, r=0, t=0, b=0),
)

fig.show()
```

### 2. Category Sankey Diagram

```python
import plotly.graph_objects as go

# Count transitions
transitions = flows.group_by(['src_category', 'dst_category']).agg(
    pl.count().alias('count')
).sort('count', descending=True)

# Get unique categories
categories = sorted(set(flows['src_category'].unique()) | set(flows['dst_category'].unique()))
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=categories,
    ),
    link=dict(
        source=[category_to_idx[row['src_category']] for row in transitions.iter_rows(named=True)],
        target=[category_to_idx[row['dst_category']] for row in transitions.iter_rows(named=True)],
        value=[row['count'] for row in transitions.iter_rows(named=True)],
    )
)])

fig.update_layout(
    title="Restaurant Category Flow",
    height=600,
)

fig.show()
```

### 3. Prediction Comparison Panel

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get predictions for first flow
first_flow = flows[0]
user_id = first_flow['user_id']
src_id = first_flow['src_gmap_id']

# XGBoost top-5
xgb_top5 = xgb_predictions.filter(
    (pl.col('user_id') == user_id) &
    (pl.col('src_gmap_id') == src_id) &
    (pl.col('rank') <= 5)
).sort('rank')

# LSTM category top-5
lstm_cat_top5 = lstm_predictions.filter(
    (pl.col('user_id') == user_id) &
    (pl.col('prediction_type') == 'category') &
    (pl.col('rank') <= 5)
).sort('rank')

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('XGBoost Top-5', 'LSTM Category Top-5'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

# XGBoost bars
fig.add_trace(
    go.Bar(
        x=xgb_top5['dst_name'],
        y=xgb_top5['score'],
        name='XGBoost',
        marker_color=['green' if is_actual else 'blue' for is_actual in xgb_top5['is_actual']],
    ),
    row=1, col=1
)

# LSTM bars
fig.add_trace(
    go.Bar(
        x=lstm_cat_top5['predicted_category'],
        y=lstm_cat_top5['probability'],
        name='LSTM',
        marker_color=['green' if is_actual else 'orange' for is_actual in lstm_cat_top5['is_actual']],
    ),
    row=1, col=2
)

fig.update_layout(height=400, showlegend=False)
fig.show()
```

---

## Common Tasks

### Task 1: Filter by Date Range

```python
# Filter flows from September 2021
sept_flows = flows.filter(
    (pl.col('src_ts') >= '2021-09-01') &
    (pl.col('src_ts') < '2021-10-01')
)
```

### Task 2: Get Top Categories

```python
# Count destination categories
top_categories = flows.group_by('dst_category').agg(
    pl.count().alias('count')
).sort('count', descending=True)

print(top_categories.head(10))
```

### Task 3: Calculate Model Metrics

```python
# XGBoost Recall@K
for k in [1, 3, 5, 10]:
    recall = xgb_predictions.filter(
        (pl.col('rank') <= k) & 
        (pl.col('is_actual') == True)
    ).height / flows.height
    print(f"XGBoost Recall@{k}: {recall:.2%}")

# LSTM Category Accuracy
category_correct = lstm_predictions.filter(
    (pl.col('prediction_type') == 'category') &
    (pl.col('rank') == 1) &
    (pl.col('is_actual') == True)
).height

total_flows = flows.height
accuracy = category_correct / total_flows
print(f"LSTM Category Accuracy: {accuracy:.2%}")
```

### Task 4: Join Predictions with Business Info

```python
# Add business details to XGBoost predictions
xgb_with_details = xgb_predictions.join(
    businesses,
    left_on='dst_gmap_id',
    right_on='gmap_id',
    how='left'
)

# Now you have lat/lon/rating/etc for each prediction
print(xgb_with_details.head())
```

---

## Data Schema Reference

### Food Categories (24 total)

```python
CATEGORIES = [
    "american", "asian", "bakery", "bar", "bbq", "breakfast", "brewery",
    "burger", "cafe", "chinese", "coffee", "dessert", "fast_food",
    "ice_cream", "italian", "mexican", "nightclub", "pizza", "pub",
    "restaurant", "seafood", "steakhouse", "sushi", "wine_bar"
]
```

### Atlanta Bounds

```python
ATLANTA_BOUNDS = {
    "lat_min": 33.6,
    "lat_max": 34.0,
    "lon_min": -84.6,
    "lon_max": -84.2,
}
```

---

## Notes & Considerations

### 1. Mock Data Limitations

- **Not real user behavior**: Transitions are randomly generated
- **Simplified patterns**: Real data will have more complex patterns
- **Fixed metrics**: Recall/accuracy are artificially set (~70% XGBoost, ~60% LSTM)
- **Limited businesses**: Only 100 businesses (real data has 27K+)

### 2. What Will Change with Real Data

- **More businesses**: 27,710 businesses in Georgia
- **More flows**: Millions of user flows
- **Better predictions**: Real model performance (may be higher or lower)
- **Richer metadata**: More business attributes (hours, menu items, etc.)
- **Temporal patterns**: Real seasonality and trends

### 3. What Won't Change

- **Data structure**: Column names and types will remain the same
- **File format**: Parquet files with same schema
- **Prediction format**: Top-K predictions with scores/probabilities
- **Geographic bounds**: Atlanta coordinates (33.6-34.0°N, -84.6 to -84.2°W)

---

## Troubleshooting

### Issue: "No module named 'polars'"

**Solution**: Activate the virtual environment
```bash
source dva_env/bin/activate  # On Windows: dva_env\Scripts\activate
```

### Issue: "File not found"

**Solution**: Make sure you're in the project root directory
```bash
cd /path/to/Forkast
python generate_mock_data.py
```

### Issue: Need more/less data

**Solution**: Adjust the size parameters
```bash
# Smaller dataset (faster, for quick tests)
python generate_mock_data.py --size 100 --businesses 20

# Larger dataset (more realistic)
python generate_mock_data.py --size 5000 --businesses 500
```

