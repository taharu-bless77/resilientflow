import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.causal.spatial.st_models import SpatioTemporalModel
from resilientflow.causal.spatial.gnn_loader import GNNDataLoader

def main():
    print("=== ResilientFlow Spatial Modeling Test ===")
    
    # 1. ダミーの多国間時系列データを作成
    # 3カ国 (A, B, C) x 2変数 (Rain, Traffic) x 20日分
    dates = pd.date_range("2024-01-01", periods=20)
    data = np.random.randn(20, 6) # 20日 x 6カラム
    cols = ["A_Rain", "A_Traffic", "B_Rain", "B_Traffic", "C_Rain", "C_Traffic"]
    df = pd.DataFrame(data, columns=cols, index=dates)
    
    print(f"[*] Input DataFrame: {df.shape}")
    
    # 2. データローダーでグラフ信号に変換
    loader = GNNDataLoader(df)
    dataset = loader.get_dataset()
    
    print("[*] Converted to StaticGraphTemporalSignal")
    # データの最初のスナップショットを取得してみる
    snapshot = next(iter(dataset))
    print(f"    Node Features shape (X): {snapshot.x.shape}  (Num_Nodes, Num_Features)")
    print(f"    Edge Index shape: {snapshot.edge_index.shape}  (2, Num_Edges)")
    
    # 3. モデルの初期化
    # 入力特徴量=2 (Rain, Traffic), 隠れ層=16, 出力=2 (次の時刻のRain, Traffic)
    model = SpatioTemporalModel(node_features=2, hidden_channels=16, out_channels=2)
    print("[*] Model initialized: SpatioTemporalModel (GConvGRU)")
    
    # 4. 順伝播 (Forward) テスト
    # データセット全体を回してエラーが出ないか確認
    print("[*] Running forward pass on snapshots...")
    
    model.train()
    cost = 0
    for time, snapshot in enumerate(dataset):
        # snapshot.x は (Nodes, Features) だが、GConvGRUは (Batch, Nodes, Features) を期待する場合がある
        # 今回のライブラリ仕様に合わせて調整
        x = torch.tensor(snapshot.x, dtype=torch.float)
        edge_index = torch.tensor(snapshot.edge_index, dtype=torch.long)
        y_target = torch.tensor(snapshot.y, dtype=torch.float)
        
        # モデル予測
        y_hat = model(x, edge_index)
        
        # 損失計算 (MSE)
        loss = torch.mean((y_hat - y_target)**2)
        cost += loss.item()
        
    cost = cost / (time + 1)
    print(f"[*] Test Run Complete. Mean MSE Loss: {cost:.4f}")
    print("=== Spatial Modeling Layer is Working! ===")

if __name__ == "__main__":
    main()