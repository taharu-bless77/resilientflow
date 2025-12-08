# ファイル: src/resilientflow/causal/discovery/tigramite_engine.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC

class TigramiteEngine:
    """
    時系列データから因果グラフ構造(スケルトンと方向)を発見するエンジンのラッパー
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.results = None

    def run_discovery(self, df: pd.DataFrame, max_lag: int = 3) -> Dict[str, Any]:
        """
        PCMCI+アルゴリズムを実行し、因果グラフを発見する
        
        Args:
            df (pd.DataFrame): 時系列データ (インデックスは日付であること)
            max_lag (int): 探索する最大ラグ（何日前の影響まで見るか）
            
        Returns:
            Dict: 発見されたグラフ情報（隣接行列、p値行列など）
        """
        # 1. Tigramite用データフレーム形式に変換
        # 変数名のリストを保持
        var_names = df.columns.tolist()
        values = df.values

        # Tigramiteのdataframeオブジェクト作成
        dataframe = pp.DataFrame(values, var_names=var_names)

        # 2. 条件付き独立性検定の選択
        # ParCorr (Partial Correlation) は高速で線形関係に強い
        # 非線形を捉えたい場合は GPDC (Gaussian Process) を使うが計算が重い
        cond_ind_test = ParCorr(significance='analytic')

        # 3. PCMCIの初期化と実行
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        
        print(f"[*] Running PCMCI with max_lag={max_lag} on {len(var_names)} variables...")
        self.results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=self.alpha)
        
        # 4. 結果の整理
        # q_matrix: 多重検定補正後のp値行列
        # graph: 因果グラフ (link_matrix)
        # val_matrix: 係数の強さ (MCI: Momentary Conditional Independence)
        
        output = {
            "graph_matrix": self.results['graph'],      # 接続関係 (文字列)
            "val_matrix": self.results['val_matrix'],   # 係数の強さ
            "p_matrix": self.results['p_matrix'],       # p値
            "var_names": var_names
        }
        
        # ヒートマップ表示用に有意なリンクだけ抽出した行列を作成
        sig_links = (self.results['p_matrix'] < self.alpha).astype(int)
        output["significant_links"] = sig_links
        
        return output

    def save_results(self, output_dir: Path):
        """結果を保存する"""
        if self.results is None:
            raise ValueError("Run discovery first.")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 行列をNPY形式で保存
        np.save(output_dir / "val_matrix.npy", self.results['val_matrix'])
        np.save(output_dir / "graph_matrix.npy", self.results['graph'])
        
        # 変数名リストを保存
        with open(output_dir / "var_names.txt", "w") as f:
            for name in self.results['var_names']: # type: ignore
                f.write(f"{name}\n")
                
        print(f"[*] Causal graph artifacts saved to {output_dir}")