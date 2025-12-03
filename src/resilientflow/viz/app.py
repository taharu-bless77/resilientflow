import dash
from dash import callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import sys

# パス設定
sys.path.append(str(Path(__file__).parent.parent.parent))

from resilientflow.causal.inference.interventions import CausalInferenceEngine
from resilientflow.viz.layouts.dashboard import create_layout

# --- 1. データとモデルの準備 ---
print("[*] Initializing Data and Causal Engine...")

# ダミーデータ生成
np.random.seed(42)
n_samples = 500
rain = np.random.randn(n_samples)
traffic = 0.7 * rain + np.random.normal(0, 0.5, n_samples)
delay = 0.5 * traffic + np.random.normal(0, 0.5, n_samples)
df = pd.DataFrame({"Rain": rain, "Traffic": traffic, "Delay": delay})

# 因果グラフ定義
graph = nx.DiGraph()
graph.add_edges_from([("Rain", "Traffic"), ("Traffic", "Delay")])

# 推論エンジン初期化
engine = CausalInferenceEngine(graph, df)

# Cytoscape用エレメント作成
cytoscape_elements = []
for node in graph.nodes:
    cytoscape_elements.append({
        'data': {'id': node, 'label': node},
        'classes': 'target' if node == 'Delay' else 'intervention' if node == 'Traffic' else ''
    })
for edge in graph.edges:
    cytoscape_elements.append({'data': {'source': edge[0], 'target': edge[1]}})

# --- 2. アプリ設定 ---
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# レイアウトの読み込み
app.layout = create_layout(cytoscape_elements)

# --- 3. コールバック (ロジック) ---
@callback(
    [Output("simulation-result", "children"),
     Output("simulation-detail", "children")],
    [Input("intervention-slider", "value")]
)
def update_simulation(traffic_value):
    predicted_delay = engine.estimate_intervention(
        target_variable="Delay",
        intervention_variable="Traffic",
        intervention_value=float(traffic_value)
    )
    baseline_delay = engine.estimate_intervention("Delay", "Traffic", 0.0)
    diff = predicted_delay - baseline_delay
    
    result_text = f"{predicted_delay:.2f}"
    if diff > 0:
        detail_text = f"Baseline比: +{diff:.2f} (悪化)"
    else:
        detail_text = f"Baseline比: {diff:.2f} (改善)"
        
    return result_text, detail_text

# --- 4. サーバー起動 ---
if __name__ == "__main__":
    print("[*] Starting Dash Server on http://localhost:8050")
    app.run(debug=True, host='0.0.0.0', port=8050)