import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # 純正プロットの代わりにこれを使います
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.causal.discovery.tigramite_engine import TigramiteEngine

# 注意: from tigramite import plotting as tp は削除しました

def generate_synthetic_data(n_samples=200):
    """
    テスト用に因果関係が既知のダミーデータを生成する
    構造: Rain(t-1) -> Traffic(t), Traffic(t) -> Delay(t), Traffic(t-1) -> Delay(t)
    """
    np.random.seed(42)
    data = np.random.randn(n_samples, 3)
    # 0:Rain, 1:Traffic, 2:Delay
    
    # 1. Rain(t-1) -> Traffic(t)
    data[1:, 1] += 0.7 * data[:-1, 0]
    
    # 2. Traffic(t) -> Delay(t)
    data[:, 2] += 0.5 * data[:, 1]
    
    # 3. Traffic(t-1) -> Delay(t)
    data[1:, 2] += 0.3 * data[:-1, 1]
    
    df = pd.DataFrame(data, columns=["Rain_count", "Traffic_Jam_count", "Logistics_Delay_count"])
    df.index = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")
    return df

def plot_causal_graph_custom(val_matrix, p_matrix, var_names, output_path):
    """
    Tigramiteの純正プロットの代わりにNetworkXで描画する関数
    """
    G = nx.DiGraph()
    N = len(var_names)
    alpha = 0.05
    
    # ノードを追加
    for name in var_names:
        G.add_node(name)
    
    # エッジを追加
    edge_labels = {}
    for j in range(N): # Effect
        for i in range(N): # Cause
            # 最大ラグまでのリンクを確認
            for tau in range(val_matrix.shape[2]):
                p_val = p_matrix[i, j, tau]
                strength = val_matrix[i, j, tau]
                
                # 有意かつ自己ループ(ラグ0)以外を表示
                if p_val < alpha and not (i == j and tau == 0):
                    # ラグ表記
                    lag_suffix = f"(t-{tau})" if tau > 0 else ""
                    # NetworkXは「ノード間の単一エッジ」が基本なので、
                    # 簡易化のため「最も強いラグ」だけ描画するか、ラベルにラグ情報を書く
                    
                    G.add_edge(var_names[i], var_names[j])
                    edge_labels[(var_names[i], var_names[j])] = f"{strength:.2f}\n{lag_suffix}"

    # 描画設定
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5) # レイアウト計算
    
    # ノード描画
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # エッジ描画
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    
    # エッジラベル（強度とラグ）描画
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Discovered Causal Graph (Custom Plot)")
    plt.axis('off')
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Graph image saved to {output_path}")
    plt.close()

def main():
    print("=== ResilientFlow Causal Discovery Test ===")
    
    # 1. データロード (テスト用ダミーデータ生成)
    df = generate_synthetic_data()
    print(f"[*] Generated synthetic data: {df.shape}")

    # 2. 因果探索の実行
    engine = TigramiteEngine(significance_level=0.05)
    results = engine.run_discovery(df, max_lag=2)
    
    # 3. テキストで結果表示
    print("\n--- Discovered Causal Links ---")
    val_matrix = results['val_matrix']
    p_matrix = results['p_matrix']
    var_names = results['var_names']
    
    found_links = False
    N = len(var_names)
    for j in range(N):
        for i in range(N):
            for tau in range(val_matrix.shape[2]):
                if p_matrix[i, j, tau] < 0.05:
                    if i == j and tau == 0: continue
                    cause = var_names[i]
                    effect = var_names[j]
                    lag = f"(t-{tau})" if tau > 0 else "(t)"
                    strength = val_matrix[i, j, tau]
                    print(f"{cause} {lag} --> {effect} : {strength:.3f}")
                    found_links = True

    # 4. 可視化 (Custom NetworkX)
    output_dir = Path("data/05_reporting")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if found_links:
        plot_causal_graph_custom(
            val_matrix, 
            p_matrix, 
            var_names, 
            output_dir / "causal_graph_custom.png"
        )
    else:
        print("[!] No links found to plot.")

if __name__ == "__main__":
    main()