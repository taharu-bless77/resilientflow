# ファイル: run_temporal_test.py

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.data.processing.temporal import aggregate_events_daily

def main():
    print("=== ResilientFlow Temporal Aggregation Test ===")
    
    # 1. 加工済み中間データの読み込み
    input_path = Path("data/02_intermediate/categorized_history.parquet")
    if not input_path.exists():
        print("[!] Previous data not found. Run 'run_history_test.py' first.")
        return
        
    df = pd.read_parquet(input_path)
    print(f"[*] Loaded {len(df)} categorized events.")
    
    # 2. 時系列集計の実行
    # テストとして、US(アメリカ), UK(イギリス), CH(中国) などを対象にしてみる
    # データに含まれている国コードを確認して適宜変更してください
    target_countries = ['US', 'UK', 'CH', 'JA', 'RS'] # JA:Japan, RS:Russia
    
    print(f"[*] Aggregating time series for: {target_countries}...")
    ts_df = aggregate_events_daily(df, target_countries=target_countries)
    
    if ts_df.empty:
        print("[!] Resulting time series is empty. (Maybe no events for target countries?)")
        # 国フィルタなしで再トライ
        print("[*] Retrying with ALL countries...")
        ts_df = aggregate_events_daily(df)
    
    # 3. 結果確認
    print("\n--- Time Series Data Structure ---")
    print(ts_df.info())
    
    print("\n--- Head of Time Series (First 5 Days) ---")
    print(ts_df.head())
    
    # 4. 簡易可視化 (Matplotlib)
    # 'INFRASTRUCTURE_DAMAGE' を含むカラムを探してプロット
    damage_cols = [c for c in ts_df.columns if "INFRASTRUCTURE_DAMAGE_count" in c]
    
    if damage_cols:
        print(f"\n[*] Plotting damage events for: {damage_cols}")
        plt.figure(figsize=(12, 6))
        ts_df[damage_cols].plot(kind='bar', figsize=(12, 6))
        plt.title("Infrastructure Damage Events per Country")
        plt.ylabel("Event Count")
        plt.tight_layout()
        
        # 画像として保存
        img_path = Path("data/05_reporting/test_timeseries.png")
        img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(img_path)
        print(f"[*] Plot saved to {img_path}")
    
    # 5. 分析用データとして保存
    output_path = Path("data/03_processed/daily_timeseries.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_parquet(output_path)
    print(f"[*] Saved processed time-series to {output_path}")

    # 6. データベースへも保存（存在する場合）
    try:
        from resilientflow.data.storage.repository import get_repository

        repo = get_repository()
        repo.save_timeseries(ts_df, table_name="daily_timeseries")
        print("[*] Saved processed time-series to database table 'daily_timeseries'.")
    except Exception as e:
        print(f"[!] Failed to save to database: {e}")

if __name__ == "__main__":
    main()