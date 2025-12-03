import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class SpatioTemporalModel(torch.nn.Module):
    """
    時空間グラフニューラルネットワーク (ST-GNN) モデル
    GConvGRU (Graph Convolutional GRU) を使用して、
    「時間の流れ」と「隣国への波及」を同時に学習する。
    """
    
    def __init__(self, node_features: int, hidden_channels: int, out_channels: int):
        """
        Args:
            node_features: 各ノード(国)が持つ特徴量の数 (例: 雨, 渋滞, 株価... = 3)
            hidden_channels: 隠れ層のニューロン数
            out_channels: 出力する未来の予測値の数 (例: 1 = 遅延予測)
        """
        super(SpatioTemporalModel, self).__init__()
        
        # GConvGRU: グラフ畳み込みを持つリカレント層
        # K=2: 2ホップ先(隣の隣の国)まで情報を集約する
        self.recurrent = GConvGRU(node_features, hidden_channels, K=2)
        
        # 最終的な予測値を出す線形層
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        """
        順伝播計算
        Args:
            x: ノード特徴量 (Batch, Node, Features)
            edge_index: グラフの接続関係 (2, Num_Edges)
            edge_weight: エッジの重み (Num_Edges)
        """
        # GConvGRUに入力
        h = self.recurrent(x, edge_index, edge_weight)
        
        # 活性化関数 (ReLU)
        h = F.relu(h)
        
        # 最終予測
        h = self.linear(h)
        return h