#!/usr/bin/env python3
"""
Test script for AutoML feature selection functionality.

This demonstrates the "democratization of data science" concept:
Users no longer need to choose variables manually - the AI does it for them.
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from resilientflow.causal.automl.feature_selection import (
    AutoFeatureSelector,
    run_auto_feature_selection_demo
)


def test_with_real_data():
    """Test with actual GDELT time-series data if available."""
    print("\n" + "="*70)
    print("Testing with Real GDELT Data")
    print("="*70 + "\n")
    
    # Check if processed data exists
    data_path = Path("data/03_processed/daily_timeseries.parquet")
    if not data_path.exists():
        print("[!] No processed data found. Run 'python run_temporal_test.py' first.")
        print("[*] Running synthetic demo instead...\n")
        run_auto_feature_selection_demo()
        return
    
    # Load real data
    df = pd.read_parquet(data_path)
    print(f"[*] Loaded time-series data: {df.shape}")
    print(f"[*] Columns: {len(df.columns)} features\n")
    
    # Select a target variable (example: infrastructure damage count for US)
    target_candidates = [col for col in df.columns if "INFRASTRUCTURE_DAMAGE" in col and "count" in col]
    
    if not target_candidates:
        print("[!] No suitable target variable found. Using synthetic demo.")
        run_auto_feature_selection_demo()
        return
    
    target_column = target_candidates[0]
    print(f"[*] Target KPI: {target_column}\n")
    
    # Prepare data (drop rows with all NaN, reset index)
    df = df.fillna(0).reset_index(drop=True)
    
    # Run automatic feature selection
    selector = AutoFeatureSelector(top_k=5)
    result = selector.select_features(
        data=df,
        target_column=target_column,
        method="ensemble",
        exclude_columns=["date"] if "date" in df.columns else None
    )
    
    # Display results
    print(result.explanation)
    
    print("\n" + "="*70)
    print("TOP 10 Feature Ranking:")
    print("="*70)
    for imp in result.importance_details[:10]:
        bar_length = int(imp.importance_score * 50)
        bar = "█" * bar_length
        print(f"{imp.rank:2d}. {imp.feature_name:40s} {bar} {imp.importance_score:.4f}")
    
    # Save results
    output_dir = Path("data/04_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame([
        {
            "rank": imp.rank,
            "feature": imp.feature_name,
            "importance": imp.importance_score,
            "method": imp.method
        }
        for imp in result.importance_details
    ])
    
    output_path = output_dir / "feature_importance.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n[*] Results saved to: {output_path}")
    
    # Recommendation for next steps
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Use these TOP 5 features for causal discovery:")
    print("   python run_discovery_test.py --features", " ".join(result.selected_features[:3]), "...")
    print("\n2. Build a causal graph focusing on these variables")
    print("\n3. Create an intervention slider UI for the most influential feature:")
    print(f"   '{result.selected_features[0]}'")
    

def main():
    print("="*70)
    print(" ResilientFlow: AutoML Feature Selection Test")
    print("="*70)
    print("\nThis test demonstrates 'Data Science Democratization':")
    print("- No manual variable selection needed")
    print("- AI automatically identifies TOP 5 most important features")
    print("- Results are ready for causal analysis\n")
    
    # Run synthetic demo first
    run_auto_feature_selection_demo()
    
    # Then try with real data
    test_with_real_data()


if __name__ == "__main__":
    main()
