# ファイル: run_history_test.py

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.data.ingestion.gdelt_client import GDELTClient
from resilientflow.data.processing.cameo_mapping import enrich_with_categories

def main():
    print("=== ResilientFlow History Fetch Test ===")
    
    client = GDELTClient()
    
    # 昨日を指定
    yesterday = datetime.now() - timedelta(days=1)
    
    # 昨日のデータを取得 (コード内で最大5ファイルに制限しているので数分で終わるはず)
    df_raw = client.fetch_history(yesterday)
    
    if df_raw.empty:
        print("[!] No data found.")
        return
        
    print(f"[*] Total raw events fetched: {len(df_raw)}")
    
    # 加工処理
    print("[*] Applying CAMEO mapping...")
    df_processed = enrich_with_categories(df_raw)
    
    # 集計結果
    print("\n--- Event Category Distribution ---")
    print(df_processed["EventCategory"].value_counts())
    
    # 重要なイベントを抽出
    relevant = df_processed[df_processed["EventCategory"] != "OTHER"]
    
    if not relevant.empty:
        print(f"\n--- Found {len(relevant)} Relevant Events! Sample: ---")
        cols = ["Day", "EventCategory", "Actor1Name", "Actor2Name", "SOURCEURL"]
        print(relevant[cols].head())
        
        # 保存
        output_path = "categorized_history.parquet"
        relevant.to_parquet(Path("data/02_intermediate") / output_path)
        print(f"[*] Saved to data/02_intermediate/{output_path}")
    else:
        print("[!] Still no events found. Try another day or check mapping logic.")

if __name__ == "__main__":
    main()