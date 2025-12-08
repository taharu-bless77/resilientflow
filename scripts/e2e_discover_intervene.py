"""E2E sample: Run causal discovery then run an intervention simulation.

This script uses FastAPI's TestClient against the app to demonstrate
the typical flow the frontend would call:
  1. POST /api/v1/causal/discover
  2. POST /api/v1/causal/intervene

The script is resilient: if discovery fails due to heavy dependencies,
it falls back to creating a minimal causal_graph in `state` so the
intervention step can still be demonstrated.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

# ensure local package is importable
sys.path.insert(0, str(Path := "src"))

os.environ["TESTING"] = "1"

from resilientflow.api import main as main_mod


def prepare_data():
    # Small synthetic time-series-like dataframe
    n = 20
    t = np.arange(n)
    x = np.sin(t / 3.0) + np.random.normal(0, 0.1, n)
    y = 0.5 * x + np.random.normal(0, 0.1, n)
    df = pd.DataFrame({"X": x, "Y": y})
    main_mod.state.data = df
    print(f"[+] Prepared synthetic data: {df.shape}")


def run_discover_then_intervene():
    client = TestClient(main_mod.app)

    # Attempt discovery
    discover_payload = {"selected_features": ["X", "Y"], "tau_max": 1, "pc_alpha": 0.05}
    print("[+] Calling /api/v1/causal/discover ...")
    resp = client.post("/api/v1/causal/discover", json=discover_payload)

    if resp.status_code == 200:
        print("[+] Discovery succeeded")
        print(json.dumps(resp.json(), indent=2))
    else:
        print(f"[!] Discovery failed (status {resp.status_code}), falling back to minimal causal_graph")
        # Create a minimal causal_graph that mimic discovery output
        main_mod.state.causal_graph = {
            "var_names": ["X", "Y"],
            # shape: (n_vars, n_vars, lags+1)
            "val_matrix": np.zeros((2, 2, 2)).tolist(),
            "p_matrix": np.ones((2, 2, 2)).tolist()
        }

    # Run intervention: set X -> 1.0 and observe Y
    intervene_payload = {"intervention_variable": "X", "intervention_value": 1.0, "target_variable": "Y"}
    print("[+] Calling /api/v1/causal/intervene ...")
    resp2 = client.post("/api/v1/causal/intervene", json=intervene_payload)
    if resp2.status_code == 200:
        print("[+] Intervention succeeded")
        print(json.dumps(resp2.json(), indent=2))
    else:
        print(f"[!] Intervention failed: {resp2.status_code} - {resp2.text}")


if __name__ == "__main__":
    prepare_data()
    run_discover_then_intervene()
