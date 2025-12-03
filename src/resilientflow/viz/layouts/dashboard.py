from dash import html
import dash_bootstrap_components as dbc
from ..components.causal_graph import render_causal_graph_card
from ..components.controls import render_control_panel

def create_layout(cytoscape_elements):
    """ダッシュボード全体のレイアウトを構築する"""
    return dbc.Container([
        # ヘッダー
        dbc.Row([
            dbc.Col(html.H2("ResilientFlow: Causal Simulator", className="text-center text-dark mb-4"), width=12)
        ], className="mt-4"),

        dbc.Row([
            # 左側: 因果グラフ
            dbc.Col([
                render_causal_graph_card(cytoscape_elements)
            ], width=12, lg=7),

            # 右側: 操作パネル
            dbc.Col([
                render_control_panel()
            ], width=12, lg=5)
        ])
    ], fluid=True)