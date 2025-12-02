# ファイル: run_processing_test.py

import sys
import pandas as pd
from pathlib import Path

# srcディレクトリをパスに通す
sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.data.processing.cameo_mapping import enrich_with_categories

def main():
    print("=== ResilientFlow Data Processing Test ===")
    
    # 1. 保存済みの生データを読み込む
    input_path = Path("data/01_raw/latest_events.parquet")
    if not input_path.exists():
        print(f"[!] File not found: {input_path}")
        print("Please run 'run_ingestion_test.py' first.")
        return

    print(f"[*] Loading raw data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 2. カテゴリ付与を実行
    print("[*] Applying CAMEO mapping...")
    df_processed = enrich_with_categories(df)
    
    # 3. 結果の集計
    category_counts = df_processed["EventCategory"].value_counts()
    
    print("\n--- Event Category Distribution ---")
    print(category_counts)
    
    # 4. 「その他(OTHER)」以外の重要なイベントを表示
    relevant_events = df_processed[df_processed["EventCategory"] != "OTHER"]
    
    if not relevant_events.empty:
        print(f"\n--- Found {len(relevant_events)} Relevant Events! ---")
        cols = ["Day", "EventCategory", "Actor1Name", "Actor2Name", "SOURCEURL"]
        print(relevant_events[cols].head(10))
        
        # 加工済みデータとして保存
        output_path = Path("data/02_intermediate/categorized_events.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        relevant_events.to_parquet(output_path, index=False)
        print(f"\n[*] Saved relevant events to {output_path}")
    else:
        print("\n[!] No infrastructure/disaster events found in this batch.")

if __name__ == "__main__":
    main()