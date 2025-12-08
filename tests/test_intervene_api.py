from fastapi.testclient import TestClient
import pandas as pd
import sys
import types

from resilientflow.api import main as main_mod


client = None


def setup_module(module):
    # Prepare a small synthetic dataset and ensure no causal graph exists
    df = pd.DataFrame({
        "A": [0.0, 1.0, 2.0, 1.5, 0.5],
        "B": [1.0, 2.0, 3.0, 2.5, 1.5]
    })
    # Inject a lightweight fake interventions module to avoid heavy dowhy imports
    fake_mod = types.ModuleType("resilientflow.causal.inference.interventions")

    class CausalInferenceEngine:
        def __init__(self, graph, data):
            self.graph = graph
            self.data = data

        def estimate_intervention(self, target_variable=None, intervention_variable=None, intervention_value=None, *args, **kwargs):
            # Accept alternate keyword names to mirror the production wrapper
            if target_variable is None and 'target' in kwargs:
                target_variable = kwargs.get('target')
            if intervention_variable is None and 'intervention_var' in kwargs:
                intervention_variable = kwargs.get('intervention_var')
            if intervention_variable is None and 'intervention' in kwargs:
                intervention_variable = kwargs.get('intervention')
            if intervention_value is None and 'value' in kwargs:
                intervention_value = kwargs.get('value')

            # Simple deterministic proxy: baseline + intervention_value
            if target_variable is None or intervention_variable is None or intervention_value is None:
                raise ValueError("Missing args")
            baseline = float(self.data[target_variable].mean())
            mean = float(baseline + intervention_value)
            # If caller requests dict with stats, provide a small deterministic std and CI
            if kwargs.get('return_dict'):
                std = 0.1
                import math
                se = std / math.sqrt(len(self.data))
                z = 1.96
                return {
                    "mean": mean,
                    "std": std,
                    "ci_lower": mean - z * se,
                    "ci_upper": mean + z * se,
                }
            return float(mean)

    fake_mod.CausalInferenceEngine = CausalInferenceEngine
    sys.modules["resilientflow.causal.inference.interventions"] = fake_mod

    main_mod.state.data = df
    main_mod.state.causal_graph = None
    main_mod.state.inference_engine = None

    # Create TestClient after monkeypatching heavy imports
    global client
    client = TestClient(main_mod.app)


def teardown_module(module):
    # Clean up state
    main_mod.state.data = None
    main_mod.state.causal_graph = None
    main_mod.state.inference_engine = None


def test_intervene_fallback_success():
    payload = {
        "intervention_variable": "A",
        "intervention_value": 5.0,
        "target_variable": "B"
    }

    resp = client.post("/api/v1/causal/intervene", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["intervention_variable"] == "A"
    assert data["target_variable"] == "B"
    assert "predicted_effect" in data
    assert "baseline_value" in data
    assert "change_percentage" in data


def test_intervene_missing_variable():
    payload = {
        "intervention_variable": "NON_EXISTENT",
        "intervention_value": 1.0,
        "target_variable": "B"
    }
    resp = client.post("/api/v1/causal/intervene", json=payload)
    assert resp.status_code == 400


def test_intervene_multiple_calls_keyword_compat():
    # Call multiple times with different values to exercise engine reuse
    for val in [0.0, 1.0, 3.5]:
        payload = {
            "intervention_variable": "A",
            "intervention_value": val,
            "target_variable": "B"
        }
        resp = client.post("/api/v1/causal/intervene", json=payload)
        assert resp.status_code == 200
        d = resp.json()
        assert isinstance(d.get("predicted_effect"), float)
