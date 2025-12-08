# ファイル: src/resilientflow/causal/inference/interventions.py

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional
import dowhy.gcm as gcm

class CausalInferenceEngine:
    """
    因果グラフとデータに基づいて、介入シミュレーションを行うエンジン
    (DoWhy GCM: Graphical Causal Model 機能を利用)
    """

    def __init__(self, graph: nx.DiGraph, data: pd.DataFrame):
        """
        Args:
            graph: NetworkXの有向グラフ（因果構造）
            data: 学習用データフレーム
        """
        self.graph = graph
        self.data = data
        self.causal_model = None
        
        # モデルの構築と学習
        self._fit_model()

    def _fit_model(self):
        """
        因果グラフの各ノードに対して、データから関係性を学習する
        （構造的因果モデル SCM の構築）
        """
        # DoWhyの構造的因果モデルを作成
        self.causal_model = gcm.StructuralCausalModel(self.graph) # type: ignore
        
        # データが少なすぎる場合の簡易チェック
        if len(self.data) < 10:
            print("[!] Warning: Data is too small for reliable inference.")

        # 各エッジの関係性を自動推定（線形・非線形を自動選択）
        # auto_assignmentは、データに合わせて最適なモデル（Ridge回帰やランダムフォレスト等）を選ぶ
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        
        print("[*] Fitting Causal Mechanisms to data...")
        # 学習実行
        gcm.fit(self.causal_model, self.data)
        print("[*] Model fitting complete.")

    def estimate_intervention(
        self,
        target_variable: str = None,
        intervention_variable: str = None,
        intervention_value: float = None,
        *args,
        return_dict: bool = False,
        **kwargs
    ):
        """
        「もし介入変数を特定の値に固定したら、ターゲット変数はどうなるか？」を推定する
        例: 「もし渋滞(intervention)を0にしたら、遅延(target)は平均いくつになるか？」
        
        Returns:
            介入後のターゲット変数の期待値（平均）
        """
        # 柔軟に呼び出し名を受け付ける（API側からの別名対応）
        # 例: target, intervention_var, value を受け取れるようにする
        if target_variable is None and "target" in kwargs:
            target_variable = kwargs.get("target")
        if intervention_variable is None and "intervention_var" in kwargs:
            intervention_variable = kwargs.get("intervention_var")
        if intervention_variable is None and "intervention" in kwargs:
            intervention_variable = kwargs.get("intervention")
        if intervention_value is None and "value" in kwargs:
            intervention_value = kwargs.get("value")

        # 引数チェック
        if target_variable is None or intervention_variable is None or intervention_value is None:
            raise ValueError("estimate_intervention requires target_variable, intervention_variable, and intervention_value")

        # 介入を実行（do-operator）
        # 特定のノードの値を固定する辞書を作成
        intervention_dict = {intervention_variable: lambda x: intervention_value}
        
        # サンプリングによるシミュレーション (1000回試行)
        samples = gcm.interventional_samples(
            self.causal_model,  # type: ignore
            intervention_dict, # type: ignore
            num_samples_to_draw=1000
        )

        # ターゲットのサンプル列
        target_samples = samples[target_variable].values

        mean_val = float(target_samples.mean())
        std_val = float(target_samples.std(ddof=1)) if len(target_samples) > 1 else 0.0
        # 95% 信頼区間（正規近似）
        import math

        z = 1.96
        se = std_val / math.sqrt(len(target_samples)) if len(target_samples) > 0 else 0.0
        ci_lower = mean_val - z * se
        ci_upper = mean_val + z * se

        if return_dict:
            return {
                "mean": mean_val,
                "std": std_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                # omit raw samples to keep payload small
            }

        return float(mean_val)

    def compare_scenarios(self, target: str, intervention: str, values: List[float]):
        """
        複数の介入シナリオを比較してコンソールに表示するヘルパー関数
        """
        baseline = self.data[target].mean()
        print(f"\n--- Scenario Analysis: Effect of '{intervention}' on '{target}' ---")
        print(f"Baseline (Current Reality): {baseline:.4f}")
        
        for val in values:
            outcome = self.estimate_intervention(target, intervention, val)
            diff = outcome - baseline
            print(f"  Scenario [do({intervention}={val})]: {target} -> {outcome:.4f} (Change: {diff:+.4f})")