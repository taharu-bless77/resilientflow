import sys
from pathlib import Path

# srcディレクトリをパスに通してインポートできるようにする
sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.data.ingestion.gdelt_client import GDELTClient

def main():
    print("=== ResilientFlow Data Ingestion Test ===")
    
    # クライアントの初期化
    client = GDELTClient()
    
    # 最新データの取得
    df = client.fetch_latest_events()
    
    if not df.empty:
        # データの確認
        print("\n--- Data Preview (Top 5 Rows) ---")
        # 修正箇所: "SourceURL" -> "SOURCEURL"
        print(df[["Day", "EventCode", "Actor1Name", "Actor2Name", "SOURCEURL"]].head())
        
        print("\n--- Summary ---")
        print(f"Total Events: {len(df)}")
        print(f"Unique Countries (Action): {df['ActionGeo_CountryCode'].nunique()}")
        
        # データの保存テスト
        client.save_raw_data(df, "latest_events.parquet")
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()