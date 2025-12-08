import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class GNNDataLoader:
    """
    時系列DataFrameをPyTorch Geometric Temporal用の信号データに変換するクラス
    """
    
    def __init__(self, df: pd.DataFrame, adj_matrix: pd.DataFrame = None):
        """
        Args:
            df: 日次時系列データ (カラム名は 'Country_Variable' 形式を想定)
            adj_matrix: 国間の隣接行列 (Noneの場合は完全グラフやランダムグラフを仮定)
        """
        self.df = df
        self.adj_matrix = adj_matrix
        
    def _parse_columns(self):
        """カラム名から国と変数を抽出する"""
        # カラム例: 'US_Traffic', 'JP_Traffic' -> countries=['US', 'JP'], features=['Traffic']
        countries = set()
        features = set()
        
        for col in self.df.columns:
            if "_" in col:
                parts = col.split("_", 1) # 最初の_で分割
                countries.add(parts[0])
                features.add(parts[1])
                
        return sorted(list(countries)), sorted(list(features))

    def get_dataset(self, lookback: int = 7) -> StaticGraphTemporalSignal:
        """
        StaticGraphTemporalSignal (グラフ構造が固定で、値が時間変化するデータ) を生成
        
        Args:
            lookback: 過去何日分を入力とするか (今回は簡易化のため1時点入力->1時点予測とするが、拡張可能)
        """
        countries, features = self._parse_columns()
        num_nodes = len(countries)
        num_features = len(features)
        
        # 1. ノード特徴量行列の構築 (Time, Nodes, Features)
        # DataFrameを (Time, Nodes * Features) から 3次元テンソルへ変換
        data_matrix = np.zeros((len(self.df), num_nodes, num_features))
        
        for t in range(len(self.df)):
            for i, country in enumerate(countries):
                for j, feature in enumerate(features):
                    col_name = f"{country}_{feature}"
                    if col_name in self.df.columns:
                        data_matrix[t, i, j] = self.df.iloc[t][col_name]

        # PyTorch Tensorに変換
        X = torch.FloatTensor(data_matrix)
        
        # 2. エッジインデックス (グラフ構造) の構築
        # adj_matrixがない場合は、テスト用に「リング状（隣の国とつながっている）」または「ランダム」にする
        # ここでは簡易的に「国リストの順にリング状につながっている」とする
        
        edge_source = []
        edge_target = []
        
        if self.adj_matrix is not None:
            # TODO: 実データに基づく隣接行列の実装
            pass
        else:
            # ダミーの接続: US -> UK -> JP -> CH -> US ...
            for i in range(num_nodes):
                source = i
                target = (i + 1) % num_nodes
                edge_source.append(source)
                edge_target.append(target)
                # 双方向にしたい場合
                edge_source.append(target)
                edge_target.append(source)
                
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        
        # 3. ターゲット (Y) の作成
        # ここでは「次の時刻の自分自身」を予測するタスクとする
        # X[t] -> Y[t+1]
        features_input = X[:-1] # t=0 to T-1
        targets = X[1:]         # t=1 to T
        
        # StaticGraphTemporalSignal オブジェクトを返す
        return StaticGraphTemporalSignal(
            edge_index=edge_index,
            edge_weight=None,
            features=features_input.numpy(), # ライブラリの仕様上 numpy を渡すことが多い
            targets=targets.numpy()
        )