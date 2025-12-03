from dash import html, dcc
import dash_bootstrap_components as dbc

def render_control_panel():
    """右側の操作パネルコンポーネント"""
    return dbc.Card([
        dbc.CardHeader("Intervention Control", className="bg-primary text-white"),
        dbc.CardBody([
            html.H5("Scenario Setting", className="card-title"),
            html.P("交通量 (Traffic) をコントロールした場合の予測："),
            
            # スライダー
            html.Label("Intervention Strength (Traffic)", className="fw-bold mt-3"),
            dcc.Slider(
                id='intervention-slider',
                min=-3, max=3, step=0.5, value=0,
                marks={
                    -3: {'label': 'Empty', 'style': {'color': 'green'}},
                    0: 'Normal',
                    3: {'label': 'Jam', 'style': {'color': 'red'}}
                },
                className="mb-4"
            ),
            html.Hr(),
            # 結果表示エリア
            html.Div([
                html.H6("Expected Delay Impact", className="text-muted text-uppercase"),
                html.H1(id="simulation-result", className="display-4 fw-bold"),
                html.P(id="simulation-detail", className="text-muted")
            ], className="text-center mt-4")
        ])
    ], className="shadow-sm h-100")