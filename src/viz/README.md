# Atlanta Restaurant Flow Dashboard

Interactive visualization dashboard for exploring predicted restaurant visit patterns in Atlanta metropolitan area.

## Features

### 📍 **Flow Map** (Mapbox)
- Geographic visualization of restaurant-to-restaurant flows
- Color-coded by distance:
  - 🟢 Green: <2km (very close)
  - 🟡 Yellow: 2-5km (nearby)
  - 🟠 Orange: 5-10km (medium)
  - 🔴 Red: >10km (far)
- Line width scaled by flow count
- Interactive hover for flow details

### 🔀 **Category Sankey Diagram**
- Visualizes transitions between food categories
- Blue links: Same category (loyalty)
- Orange links: Different category (exploration)
- Shows top 20 category transitions

### 📊 **Top Sources Bar Chart**
- Top 10 restaurants by outgoing flow count
- Identifies popular starting points

### 🔍 **Restaurant Search**
- Search 6,887 Atlanta restaurants
- View restaurant details (rating, reviews, location)
- See top-10 predicted next destinations
- XGBoost model scores

### 🎛️ **Interactive Filters**
- **Category Filter**: Focus on specific food types
- **Max Distance**: Limit flows by distance (0-50km)
- **Min Flow Count**: Show only strong connections (1-20)
- **Top N Flows**: Display 50/100/200/500 flows

### 📈 **Live Statistics**
- Total flows (dynamically filtered)
- Total transitions
- Average flow distance
- Actual visits count

## Quick Start

### Option 1: Command Line
```bash
cd /path/to/Forkast
source dva_env/bin/activate
python src/viz/app.py
```

### Option 2: Shell Script
```bash
cd /path/to/Forkast
./run_dashboard.sh
```

### Option 3: Direct Python
```python
python -m src.viz.app
```

Then open browser to: **http://localhost:8050**

## Data Requirements

The dashboard requires filtered Atlanta data. If not already created, run:

```bash
python filter_atlanta.py
```

This creates:
- `data/viz/atlanta_businesses.parquet` (6,887 restaurants)
- `data/viz/atlanta_flows.parquet` (106,783 flows)
- `data/viz/atlanta_category_flows.parquet` (635 transitions)
- `data/viz/atlanta_top_predictions.parquet` (37,362 predictions)
- `data/viz/atlanta_summary.json` (metadata)

## Usage Examples

### Exploring American Restaurants
1. Set **Category Filter** → "american"
2. Adjust **Max Distance** → 5km (local flows)
3. Set **Min Flow Count** → 5 (strong connections)
4. Map shows american restaurant clusters and transitions

### Finding Category Loyalty
1. Observe **Sankey Diagram**
2. Blue thick links = high loyalty (e.g., burger→burger)
3. Orange links = exploration (e.g., american→mexican)

### Restaurant Predictions
1. Use **Restaurant Search** dropdown
2. Type restaurant name
3. View top-10 predicted next destinations
4. See XGBoost model confidence scores

### Identifying Hubs
1. Check **Top Sources Bar Chart**
2. Highest bars = most outgoing flows
3. These are popular "starting point" restaurants

## Dashboard Structure

```
src/viz/app.py
├── Data Loading (lines 42-67)
│   └── Loads 4 parquet files + summary JSON
│
├── Helper Functions (lines 70-218)
│   ├── create_flow_map()      # Mapbox visualization
│   ├── create_sankey_diagram() # Category flows
│   └── create_bar_chart()      # Top sources
│
├── Layout (lines 227-445)
│   ├── Header with stats
│   ├── Filter controls (4 inputs)
│   ├── Flow map (main viz)
│   ├── Sankey + Bar charts
│   ├── Statistics panel
│   └── Restaurant search
│
└── Callbacks (lines 448-575)
    ├── update_visualizations()  # Filter-driven updates
    └── display_restaurant_details()  # Search results
```

## Performance Notes

- **Initial Load**: ~2-3 seconds (loading 106K flows)
- **Filter Update**: <1 second (client-side filtering)
- **Map Rendering**: Scales with top_n (50-500 flows)
- **Memory Usage**: ~200-300 MB

**Tip**: For better performance, keep **Top N Flows** ≤ 200

## Customization

### Change Map Center
```python
# In create_flow_map(), update:
center=dict(lat=33.8, lon=-84.4),  # Downtown Atlanta
zoom=10
```

### Adjust Color Scheme
```python
# In create_flow_map(), modify distance colors:
if row['avg_distance'] < 2:
    color = 'your_color'  # Change colors here
```

### Add More Filters
```python
# In layout section, add new dcc.Dropdown or dcc.Slider
# Then update callbacks with new Input()
```

## Troubleshooting

### Dashboard won't start
```bash
# Check if data exists
ls data/viz/

# If missing, run filtering
python filter_atlanta.py
```

### Port 8050 already in use
```python
# In app.py, change port:
app.run_server(debug=True, port=8051)  # Use 8051 instead
```

### Slow rendering
```python
# Reduce top_n in filters
top_n = 50  # Instead of 500
```

### Missing restaurants in search
```python
# In layout, increase search limit:
businesses_df.head(500)  # Change to 1000 or more
```

## Data Sources

- **XGBoost Model**: `models/xgboost_ranker.json`
- **Predictions**: `models/predictions/xgboost_predictions.parquet`
- **Business Data**: `data/processed/ga/xgboost_data/biz_ga.parquet`
- **Filtered Data**: `data/viz/*.parquet`

## Technologies Used

- **Dash** 3.2.0 - Web framework
- **Plotly** 6.3.1 - Visualizations
- **Dash Bootstrap Components** - UI styling
- **Polars** - Fast data loading
- **Mapbox** - Geographic maps

## Screenshots

### Flow Map
Shows 100 strongest restaurant flows in Atlanta with distance-based coloring.

### Sankey Diagram
Reveals category loyalty patterns (burger→burger dominates).

### Statistics Panel
Live-updating metrics based on current filters.

## Next Steps

### Enhancements
- [ ] Add date range filter (if temporal data available)
- [ ] Show user paths (multi-hop sequences)
- [ ] Compare XGBoost vs actual visit patterns
- [ ] Add heatmap layer for restaurant density
- [ ] Export filtered data to CSV

### Advanced Features
- [ ] User demographics overlay
- [ ] Peak hours visualization
- [ ] Category recommendations engine
- [ ] A/B test different model predictions

## License

Part of Forkast project - Atlanta restaurant visit prediction system.

## Contact

For issues or questions, please check the main project README.