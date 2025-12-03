# ファイル: run_inference_test.py

import sys
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.causal.inference.interventions import CausalInferenceEngine

def main():
    print("=== ResilientFlow Causal Inference Test ===")

    # 1. ダミーデータの作成（Layer 2前半と同じロジック）
    # 構造: Rain -> Traffic -> Delay
    np.random.seed(42)
    n_samples = 500
    
    rain = np.random.randn(n_samples)
    # 渋滞は雨の影響を受ける (0.7倍)
    traffic = 0.7 * rain + np.random.normal(0, 0.5, n_samples)
    # 遅延は渋滞の影響を受ける (0.5倍)
    delay = 0.5 * traffic + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        "Rain": rain,
        "Traffic": traffic,
        "Delay": delay
    })
    
    print(f"[*] Generated synthetic data: {df.shape}")
    
    # 2. 因果グラフの定義
    # ここでは「発見済み」のグラフ構造を手動で定義して渡します
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("Rain", "Traffic"),
        ("Traffic", "Delay")
    ])
    
    # 3. 推論エンジンの初期化と学習
    # ここでデータとグラフから「数式（モデル）」が作られます
    engine = CausalInferenceEngine(graph, df)
    
    # 4. シミュレーション実行
    # 「もし渋滞(Traffic)をコントロールできたら、遅延(Delay)はどう変わる？」
    
    # シナリオ: 
    #  2.0: 大渋滞が発生した場合
    #  0.0: 渋滞がない場合
    # -2.0: 道がガラガラの場合
    
    engine.compare_scenarios(
        target="Delay", 
        intervention="Traffic", 
        values=[2.0, 0.0, -2.0]
    )

if __name__ == "__main__":
    main()