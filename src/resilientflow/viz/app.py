# ファイル: src/resilientflow/viz/app.py

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

# srcディレクトリへのパスを通す (推論エンジンをインポートするため)
sys.path.append(str(Path(__file__).parent.parent.parent))

from resilientflow.causal.inference.interventions import CausalInferenceEngine

# --- 1. データとモデルの準備 (アプリ起動時に一度だけ実行) ---
print("[*] Initializing Data and Causal Engine...")

# ダミーデータの生成 (Layer 2のテストと同じロジック)
np.random.seed(42)
n_samples = 500
rain = np.random.randn(n_samples)
traffic = 0.7 * rain + np.random.normal(0, 0.5, n_samples)
delay = 0.5 * traffic + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({"Rain": rain, "Traffic": traffic, "Delay": delay})

# 因果グラフの定義
graph = nx.DiGraph()
graph.add_edges_from([("Rain", "Traffic"), ("Traffic", "Delay")])

# 推論エンジンの初期化 (ここで学習が走ります)
engine = CausalInferenceEngine(graph, df)

# Cytoscape用のグラフ要素データを作成
cytoscape_elements = []
# ノード
for node in graph.nodes:
    cytoscape_elements.append({
        'data': {'id': node, 'label': node},
        'classes': 'target' if node == 'Delay' else 'intervention' if node == 'Traffic' else ''
    })
# エッジ
for edge in graph.edges:
    cytoscape_elements.append({
        'data': {'source': edge[0], 'target': edge[1]}
    })

# --- 2. アプリの設定 ---
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# グラフのスタイル定義 (CSSのようなもの)
graph_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'text-valign': 'center',
            'color': 'white',
            'text-outline-width': 2,
            'text-outline-color': '#888',
            'background-color': '#95a5a6',
            'width': 60,
            'height': 60
        }
    },
    {
        'selector': '.intervention', # Trafficノード用
        'style': {'background-color': '#e67e22', 'line-color': '#e67e22'}
    },
    {
        'selector': '.target', # Delayノード用
        'style': {'background-color': '#e74c3c', 'line-color': '#e74c3c'}
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'width': 3
        }
    }
]

# --- 3. 画面レイアウト ---
app.layout = dbc.Container([
    # ヘッダー
    dbc.Row([
        dbc.Col(html.H2("ResilientFlow: Causal Simulator", className="text-center text-dark mb-4"), width=12)
    ], className="mt-4"),

    dbc.Row([
        # 左側: 因果グラフ (Interactive)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Discovered Causal Graph"),
                dbc.CardBody([
                    html.P("ノードをドラッグして動かせます。", className="text-muted small"),
                    cyto.Cytoscape(
                        id='causal-graph',
                        elements=cytoscape_elements,
                        style={'width': '100%', 'height': '400px'},
                        layout={'name': 'breadthfirst', 'roots': ['Rain']}, # 階層レイアウト
                        stylesheet=graph_stylesheet,
                        userZoomingEnabled=False
                    )
                ])
            ], className="shadow-sm mb-4")
        ], width=12, lg=7),

        # 右側: 操作パネル
        dbc.Col([
            dbc.Card([
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
        ], width=12, lg=5)
    ])
], fluid=True)

# --- 4. インタラクション (Callback) ---
@callback(
    [Output("simulation-result", "children"),
     Output("simulation-detail", "children")],
    [Input("intervention-slider", "value")]
)
def update_simulation(traffic_value):
    """
    スライダーの値が変わるたびに呼び出される関数
    """
    # 推論エンジンを使って「もしTraffic=xだったら」を計算
    predicted_delay = engine.estimate_intervention(
        target_variable="Delay",
        intervention_variable="Traffic",
        intervention_value=float(traffic_value)
    )
    
    # 基準値(0)との差分
    baseline_delay = engine.estimate_intervention("Delay", "Traffic", 0.0)
    diff = predicted_delay - baseline_delay
    
    # 表示用の整形
    result_text = f"{predicted_delay:.2f}"
    
    if diff > 0:
        detail_text = f"Baseline比: +{diff:.2f} (悪化)"
        color_class = "text-danger" # 赤字
    else:
        detail_text = f"Baseline比: {diff:.2f} (改善)"
        color_class = "text-success" # 緑字
        
    # 文字列だけでなく、スタイル(色)も変えたい場合はOutputを増やす必要がありますが
    # 今回はシンプルにテキストのみ返します
    return result_text, detail_text

# --- 5. サーバー起動 ---
if __name__ == "__main__":
    print("[*] Starting Dash Server on http://localhost:8050")
    # debug=Trueにするとコード変更時に自動リロードされます
    app.run(debug=True, host='0.0.0.0', port=8050)