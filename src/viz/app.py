"""
Atlanta Restaurant Visit Flow Dashboard

Interactive Dash application for visualizing predicted restaurant visit patterns
in Atlanta metropolitan area using XGBoost model predictions.

Features:
- Mapbox flow map showing restaurant-to-restaurant transitions
- Sankey diagram for category-level flows
- Interactive filters (category, distance, flow strength)
- Restaurant search and details panel
- Top predictions explorer

Run with: python src/viz/app.py
Access at: http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import json
from pathlib import Path
import numpy as np


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all visualization datasets."""
    base_dir = Path(__file__).parent.parent.parent
    viz_dir = base_dir / "data" / "viz"

    print("Loading data...")
    businesses = pl.read_parquet(viz_dir / "atlanta_businesses.parquet")
    flows = pl.read_parquet(viz_dir / "atlanta_flows.parquet")
    category_flows = pl.read_parquet(viz_dir / "atlanta_category_flows.parquet")
    top_predictions = pl.read_parquet(viz_dir / "atlanta_top_predictions.parquet")

    with open(viz_dir / "atlanta_summary.json", 'r') as f:
        summary = json.load(f)

    print(f"✓ Loaded {len(businesses):,} businesses")
    print(f"✓ Loaded {len(flows):,} flows")
    print(f"✓ Loaded {len(category_flows):,} category flows")

    return businesses, flows, category_flows, top_predictions, summary


# Load data
businesses_df, flows_df, category_flows_df, predictions_df, summary_stats = load_data()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_flow_map(flows, top_n=100, min_flow=1):
    """
    Create Mapbox visualization of restaurant flows.

    Args:
        flows: Flows DataFrame
        top_n: Number of top flows to display
        min_flow: Minimum flow count to include

    Returns:
        Plotly figure
    """
    # Filter and sort flows
    filtered_flows = flows.filter(pl.col('flow_count') >= min_flow).head(top_n)

    fig = go.Figure()

    # Add flow lines
    for row in filtered_flows.iter_rows(named=True):
        # Scale line width by flow count
        line_width = np.log1p(row['flow_count']) * 2

        # Color by distance
        if row['avg_distance'] < 2:
            color = 'green'  # Very close
        elif row['avg_distance'] < 5:
            color = 'yellow'  # Nearby
        elif row['avg_distance'] < 10:
            color = 'orange'  # Medium
        else:
            color = 'red'  # Far

        # Add line
        fig.add_trace(go.Scattermapbox(
            lon=[row['src_lon'], row['dst_lon']],
            lat=[row['src_lat'], row['dst_lat']],
            mode='lines',
            line=dict(width=line_width, color=color),
            opacity=0.6,
            hovertemplate=(
                f"<b>{row['src_name']}</b> → <b>{row['dst_name']}</b><br>"
                f"Flow Count: {row['flow_count']}<br>"
                f"Distance: {row['avg_distance']:.1f} km<br>"
                f"Avg Score: {row['avg_score']:.2f}<br>"
                f"Category: {row['src_category']} → {row['dst_category']}<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # Add source points
    src_points = filtered_flows.unique(subset=['src_gmap_id'])
    fig.add_trace(go.Scattermapbox(
        lon=src_points['src_lon'].to_list(),
        lat=src_points['src_lat'].to_list(),
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.7),
        text=src_points['src_name'].to_list(),
        hovertemplate="<b>%{text}</b><br>Source Restaurant<extra></extra>",
        name='Source',
        showlegend=True
    ))

    # Add destination points
    dst_points = filtered_flows.unique(subset=['dst_gmap_id'])
    fig.add_trace(go.Scattermapbox(
        lon=dst_points['dst_lon'].to_list(),
        lat=dst_points['dst_lat'].to_list(),
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.7),
        text=dst_points['dst_name'].to_list(),
        hovertemplate="<b>%{text}</b><br>Destination Restaurant<extra></extra>",
        name='Destination',
        showlegend=True
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=33.8, lon=-84.4),
            zoom=10
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    return fig


def create_sankey_diagram(category_flows, top_n=20):
    """
    Create Sankey diagram for category transitions.

    Args:
        category_flows: Category flows DataFrame
        top_n: Number of top flows to include

    Returns:
        Plotly figure
    """
    # Get top flows
    top_flows = category_flows.head(top_n)

    # Build unique labels
    labels = []
    label_dict = {}

    for row in top_flows.iter_rows(named=True):
        src = f"{row['src_category_main']} (from)"
        dst = f"{row['dst_category_main']} (to)"

        if src not in label_dict:
            label_dict[src] = len(labels)
            labels.append(src)
        if dst not in label_dict:
            label_dict[dst] = len(labels)
            labels.append(dst)

    # Build links
    sources = []
    targets = []
    values = []
    colors = []

    for row in top_flows.iter_rows(named=True):
        src = f"{row['src_category_main']} (from)"
        dst = f"{row['dst_category_main']} (to)"

        sources.append(label_dict[src])
        targets.append(label_dict[dst])
        values.append(row['flow_count'])

        # Color by same category or different
        if row['src_category_main'] == row['dst_category_main']:
            colors.append('rgba(0, 128, 255, 0.4)')  # Blue for same category
        else:
            colors.append('rgba(255, 128, 0, 0.4)')  # Orange for different

    # Create Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=labels,
            color='lightblue'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])

    fig.update_layout(
        title=f"Top {top_n} Category Transitions",
        font=dict(size=12),
        height=500
    )

    return fig


def create_bar_chart(data, x_col, y_col, title, color=None):
    """Create bar chart."""
    fig = px.bar(
        data.to_pandas() if hasattr(data, 'to_pandas') else data,
        x=x_col,
        y=y_col,
        title=title,
        color=color,
        height=400
    )

    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )

    return fig


# ============================================================================
# DASH APP SETUP
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

app.title = "Atlanta Restaurant Flows"

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🍽️ Atlanta Restaurant Visit Flow Dashboard", className="text-center my-4"),
            html.P(
                f"Visualizing predicted restaurant visit patterns in Atlanta • "
                f"{summary_stats['total_businesses']:,} restaurants • "
                f"{summary_stats['total_flows']:,} flows",
                className="text-center text-muted"
            )
        ])
    ]),

    html.Hr(),

    # Controls Row
    dbc.Row([
        dbc.Col([
            html.Label("Category Filter:", className="fw-bold"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': 'All Categories', 'value': 'all'}] +
                        [{'label': cat.title(), 'value': cat}
                         for cat in sorted(businesses_df['category_main'].unique().to_list())],
                value='all',
                clearable=False
            )
        ], md=3),

        dbc.Col([
            html.Label("Max Distance (km):", className="fw-bold"),
            dcc.Slider(
                id='distance-slider',
                min=0,
                max=50,
                step=5,
                value=50,
                marks={i: f'{i}km' for i in range(0, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=3),

        dbc.Col([
            html.Label("Min Flow Count:", className="fw-bold"),
            dcc.Slider(
                id='min-flow-slider',
                min=1,
                max=20,
                step=1,
                value=1,
                marks={1: '1', 5: '5', 10: '10', 15: '15', 20: '20'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=3),

        dbc.Col([
            html.Label("Top N Flows:", className="fw-bold"),
            dcc.Dropdown(
                id='top-n-dropdown',
                options=[
                    {'label': '50', 'value': 50},
                    {'label': '100', 'value': 100},
                    {'label': '200', 'value': 200},
                    {'label': '500', 'value': 500}
                ],
                value=100,
                clearable=False
            )
        ], md=3)
    ], className="mb-4"),

    # Main Visualizations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📍 Restaurant Flow Map")),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='flow-map'),
                        type='default'
                    ),
                    html.Div([
                        html.Small("🟢 Green: <2km • 🟡 Yellow: 2-5km • 🟠 Orange: 5-10km • 🔴 Red: >10km",
                                 className="text-muted")
                    ], className="text-center mt-2")
                ])
            ])
        ], md=12)
    ], className="mb-4"),

    # Secondary Visualizations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔀 Category Flow (Sankey)")),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='sankey-diagram'),
                        type='default'
                    )
                ])
            ])
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📊 Top Source Restaurants")),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='top-sources-chart'),
                        type='default'
                    )
                ])
            ])
        ], md=6)
    ], className="mb-4"),

    # Statistics Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📈 Flow Statistics")),
                dbc.CardBody([
                    html.Div(id='stats-display')
                ])
            ])
        ], md=12)
    ], className="mb-4"),

    # Restaurant Search & Details
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔍 Restaurant Search")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='restaurant-search',
                        options=[
                            {'label': f"{row['name']} ({row['category_main']})", 'value': row['gmap_id']}
                            for row in businesses_df.head(500).iter_rows(named=True)
                        ],
                        placeholder="Search for a restaurant...",
                        searchable=True
                    ),
                    html.Div(id='restaurant-details', className="mt-3")
                ])
            ])
        ], md=12)
    ]),

    # Footer
    html.Hr(className="mt-5"),
    html.Footer([
        html.P("Built with Dash & Plotly • Data from XGBoost Predictions",
               className="text-center text-muted")
    ])

], fluid=True)


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('flow-map', 'figure'),
     Output('sankey-diagram', 'figure'),
     Output('top-sources-chart', 'figure'),
     Output('stats-display', 'children')],
    [Input('category-filter', 'value'),
     Input('distance-slider', 'value'),
     Input('min-flow-slider', 'value'),
     Input('top-n-dropdown', 'value')]
)
def update_visualizations(category, max_distance, min_flow, top_n):
    """Update all visualizations based on filters."""

    # Filter flows
    filtered_flows = flows_df

    if category != 'all':
        filtered_flows = filtered_flows.filter(
            (pl.col('src_category') == category) |
            (pl.col('dst_category') == category)
        )

    filtered_flows = filtered_flows.filter(
        (pl.col('avg_distance') <= max_distance) &
        (pl.col('flow_count') >= min_flow)
    )

    # Sort by flow count
    filtered_flows = filtered_flows.sort('flow_count', descending=True)

    # Create flow map
    flow_map_fig = create_flow_map(filtered_flows, top_n=top_n, min_flow=min_flow)

    # Create Sankey
    if category == 'all':
        sankey_fig = create_sankey_diagram(category_flows_df, top_n=20)
    else:
        cat_flows_filtered = category_flows_df.filter(
            (pl.col('src_category_main') == category) |
            (pl.col('dst_category_main') == category)
        )
        sankey_fig = create_sankey_diagram(cat_flows_filtered, top_n=20)

    # Top sources
    top_sources = filtered_flows.group_by('src_name').agg([
        pl.col('flow_count').sum().alias('total_flows')
    ]).sort('total_flows', descending=True).head(10)

    top_sources_fig = create_bar_chart(
        top_sources,
        'src_name',
        'total_flows',
        'Top 10 Source Restaurants by Flow Count'
    )

    # Statistics
    stats = html.Div([
        dbc.Row([
            dbc.Col([
                html.H4(f"{len(filtered_flows):,}", className="text-primary"),
                html.P("Total Flows", className="text-muted")
            ], md=3),
            dbc.Col([
                html.H4(f"{filtered_flows['flow_count'].sum():,}", className="text-success"),
                html.P("Total Transitions", className="text-muted")
            ], md=3),
            dbc.Col([
                html.H4(f"{filtered_flows['avg_distance'].mean():.1f} km", className="text-info"),
                html.P("Avg Distance", className="text-muted")
            ], md=3),
            dbc.Col([
                html.H4(f"{filtered_flows['actual_visits'].sum():,}", className="text-warning"),
                html.P("Actual Visits", className="text-muted")
            ], md=3)
        ])
    ])

    return flow_map_fig, sankey_fig, top_sources_fig, stats


@app.callback(
    Output('restaurant-details', 'children'),
    Input('restaurant-search', 'value')
)
def display_restaurant_details(restaurant_id):
    """Display details for selected restaurant."""
    if not restaurant_id:
        return html.P("Select a restaurant to see details and predictions.", className="text-muted")

    # Get restaurant info
    restaurant = businesses_df.filter(pl.col('gmap_id') == restaurant_id)

    if len(restaurant) == 0:
        return html.P("Restaurant not found.", className="text-danger")

    restaurant = restaurant.to_dicts()[0]

    # Get top predictions from this restaurant
    preds = predictions_df.filter(pl.col('src_gmap_id') == restaurant_id).head(10)

    # Build display
    details = html.Div([
        html.H5(restaurant['name'], className="mt-3"),
        html.P([
            html.Strong("Category: "), restaurant['category_main'].title(), html.Br(),
            html.Strong("Rating: "), f"⭐ {restaurant['avg_rating']:.1f}" if restaurant['avg_rating'] > 0 else "No rating", html.Br(),
            html.Strong("Reviews: "), f"{restaurant['num_reviews']:,}", html.Br(),
            html.Strong("Location: "), f"({restaurant['lat']:.4f}, {restaurant['lon']:.4f})"
        ]),

        html.Hr(),

        html.H6("🎯 Top 10 Predicted Next Destinations:"),
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Rank"),
                html.Th("Restaurant"),
                html.Th("Category"),
                html.Th("Score")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row['rank']),
                    html.Td(row['dst_name']),
                    html.Td(row['dst_category'].title()),
                    html.Td(f"{row['avg_score']:.2f}")
                ])
                for row in preds.iter_rows(named=True)
            ])
        ], bordered=True, hover=True, size='sm') if len(preds) > 0 else html.P("No predictions available.", className="text-muted")
    ])

    return details


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 STARTING ATLANTA RESTAURANT FLOW DASHBOARD")
    print("=" * 80)
    print(f"\n📊 Data Loaded:")
    print(f"  • {len(businesses_df):,} businesses")
    print(f"  • {len(flows_df):,} flows")
    print(f"  • {len(category_flows_df):,} category transitions")
    print(f"\n🌐 Opening dashboard at: http://localhost:8050")
    print("  Press Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=8050)