from dash import html
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

def render_causal_graph_card(elements):
    """因果グラフを表示するカードコンポーネント"""
    
    graph_stylesheet = [
        {'selector': 'node', 'style': {
            'content': 'data(label)', 'text-valign': 'center', 'color': 'white',
            'text-outline-width': 2, 'text-outline-color': '#888',
            'background-color': '#95a5a6', 'width': 60, 'height': 60
        }},
        {'selector': '.intervention', 'style': {'background-color': '#e67e22', 'line-color': '#e67e22'}},
        {'selector': '.target', 'style': {'background-color': '#e74c3c', 'line-color': '#e74c3c'}},
        {'selector': 'edge', 'style': {
            'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'width': 3
        }}
    ]

    return dbc.Card([
        dbc.CardHeader("Discovered Causal Graph"),
        dbc.CardBody([
            html.P("ノードをドラッグして動かせます。", className="text-muted small"),
            cyto.Cytoscape(
                id='causal-graph',
                elements=elements,
                style={'width': '100%', 'height': '400px'},
                layout={'name': 'breadthfirst', 'roots': ['Rain']},
                stylesheet=graph_stylesheet,
                userZoomingEnabled=False
            )
        ])
    ], className="shadow-sm mb-4")
