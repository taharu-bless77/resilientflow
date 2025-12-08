"""
ResilientFlow FastAPI Backend
==============================

RESTful API for causal analysis, intervention simulation, and feature selection.

This API serves as the backend for:
1. React SPA (web version)
2. Electron Desktop App
3. External integrations (Tableau, Power BI via REST)

Architecture:
- FastAPI for async performance
- Pydantic for type-safe schemas
- Background tasks for long-running computations
- OpenAPI documentation (Swagger UI at /docs)
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

from resilientflow.causal.automl.feature_selection import (
    AutoFeatureSelector,
    AutoFeatureSelectionResult
)
from resilientflow.causal.discovery.tigramite_engine import TigramiteEngine
# Note: import `CausalInferenceEngine` lazily inside endpoints to avoid
# heavy imports (dowhy/numba/llvmlite) at module import time which slows
# down test collection and CLI tasks.

# Initialize FastAPI app
app = FastAPI(
    title="ResilientFlow API",
    description="Causal Digital Twin System for Infrastructure Resilience",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Schemas (Type-Safe API Contracts)
# ============================================================================

class FeatureSelectionRequest(BaseModel):
    """Request schema for automatic feature selection."""
    target_column: str = Field(..., description="Name of target KPI column")
    method: str = Field(
        default="ensemble",
        description="Selection method: lasso, random_forest, pca, ensemble"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of features to select")
    exclude_columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to exclude from analysis"
    )


class FeatureImportanceResponse(BaseModel):
    """Response schema for feature importance."""
    feature_name: str
    importance_score: float
    rank: int


class FeatureSelectionResponse(BaseModel):
    """Response schema for feature selection."""
    selected_features: List[str]
    importance_details: List[FeatureImportanceResponse]
    explanation: str


class CausalDiscoveryRequest(BaseModel):
    """Request schema for causal discovery."""
    selected_features: List[str] = Field(..., description="Variables for causal analysis")
    tau_max: int = Field(default=3, ge=1, le=10, description="Maximum time lag")
    pc_alpha: float = Field(default=0.05, ge=0.01, le=0.2, description="Significance level")


class CausalLink(BaseModel):
    """Schema for a single causal link."""
    source: str
    target: str
    lag: int
    coefficient: float
    p_value: float


class CausalDiscoveryResponse(BaseModel):
    """Response schema for causal discovery."""
    links: List[CausalLink]
    variables: List[str]
    summary: str


class InterventionRequest(BaseModel):
    """Request schema for intervention simulation."""
    intervention_variable: str = Field(..., description="Variable to intervene on")
    intervention_value: float = Field(..., description="Value to set for intervention")
    target_variable: str = Field(..., description="Variable to predict")


class InterventionResponse(BaseModel):
    """Response schema for intervention simulation."""
    intervention_variable: str
    intervention_value: float
    target_variable: str
    predicted_effect: float
    baseline_value: float
    change_percentage: float
    predicted_std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    explanation: str


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    available_data: Dict[str, bool]



class SaveDataResponse(BaseModel):
    """Response for manual data save endpoint."""
    status: str
    rows: int
    columns: int
    message: Optional[str] = None


# ============================================================================
# Global State (In production: use Redis/Database)
# ============================================================================

class AppState:
    """Application state for caching data and models."""
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.causal_graph: Optional[Any] = None
        self.inference_engine: Optional[Any] = None
        
    def load_data(self):
        """Load processed time-series data."""
        data_path = Path("data/03_processed/daily_timeseries.parquet")
        if data_path.exists():
            self.data = pd.read_parquet(data_path).fillna(0)
            return True

        # Fallback: try loading from database
        try:
            from resilientflow.data.storage.repository import get_repository

            repo = get_repository()
            if repo.table_exists("daily_timeseries"):
                df = repo.load_timeseries("daily_timeseries")
                if df is not None:
                    self.data = df.fillna(0)
                    return True
        except Exception:
            # If DB is not configured or load fails, ignore and return False
            pass

        return False


state = AppState()


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup."""
    # In test environments we may want to skip heavy startup operations
    if os.environ.get("TESTING") == "1":
        print("🔬 Skipping startup data load (TESTING=1)")
        return

    print("🚀 ResilientFlow API starting up...")
    if state.load_data():
        print(f"✅ Loaded data: {state.data.shape}")
    else:
        print("⚠️ No processed data found. Run preprocessing first.")


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        System status and available resources
    """
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        available_data={
            "timeseries": state.data is not None,
            "causal_graph": state.causal_graph is not None,
            "inference_engine": state.inference_engine is not None
        }
    )


@app.post("/api/v1/features/select", response_model=FeatureSelectionResponse)
async def select_features(request: FeatureSelectionRequest):
    """
    Automatic feature selection endpoint.
    
    This is the "democratization of data science" API - users specify
    a target KPI, and AI automatically identifies the most important variables.
    
    Example:
        POST /api/v1/features/select
        {
            "target_column": "supply_chain_delay",
            "method": "ensemble",
            "top_k": 5
        }
    """
    if state.data is None:
        raise HTTPException(
            status_code=503,
            detail="No data loaded. Please run data preprocessing first."
        )
    
    if request.target_column not in state.data.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{request.target_column}' not found in data. "
                   f"Available columns: {state.data.columns.tolist()}"
        )
    
    try:
        selector = AutoFeatureSelector(top_k=request.top_k)
        result = selector.select_features(
            data=state.data,
            target_column=request.target_column,
            method=request.method,
            exclude_columns=request.exclude_columns
        )
        
        return FeatureSelectionResponse(
            selected_features=result.selected_features,
            importance_details=[
                FeatureImportanceResponse(
                    feature_name=imp.feature_name,
                    importance_score=imp.importance_score,
                    rank=imp.rank
                )
                for imp in result.importance_details
            ],
            explanation=result.explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature selection failed: {str(e)}")


@app.post("/api/v1/causal/discover", response_model=CausalDiscoveryResponse)
async def discover_causal_graph(
    request: CausalDiscoveryRequest,
    background_tasks: BackgroundTasks
):
    """
    Causal discovery endpoint using PCMCI+.
    
    This endpoint identifies causal relationships between selected variables
    using time-series causal discovery algorithms.
    
    Example:
        POST /api/v1/causal/discover
        {
            "selected_features": ["var1", "var2", "var3"],
            "tau_max": 3,
            "pc_alpha": 0.05
        }
    """
    if state.data is None:
        raise HTTPException(status_code=503, detail="No data loaded")
    
    # Validate selected features
    missing_features = set(request.selected_features) - set(state.data.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Features not found: {missing_features}"
        )
    
    try:
        # Prepare data for causal discovery
        selected_df = state.data[request.selected_features]
        
        # Run causal discovery
        engine = TigramiteEngine(significance_level=request.pc_alpha)
        result = engine.run_discovery(df=selected_df, max_lag=request.tau_max)
        
        # Extract significant links
        links = []
        val_matrix = result["val_matrix"]
        p_matrix = result["p_matrix"]
        
        for i, target_var in enumerate(request.selected_features):
            for j, source_var in enumerate(request.selected_features):
                for lag in range(request.tau_max + 1):
                    if p_matrix[i, j, lag] < request.pc_alpha:
                        links.append(CausalLink(
                            source=source_var,
                            target=target_var,
                            lag=lag,
                            coefficient=float(val_matrix[i, j, lag]),
                            p_value=float(p_matrix[i, j, lag])
                        ))
        
        # Store for later use
        state.causal_graph = result
        
        return CausalDiscoveryResponse(
            links=links,
            variables=request.selected_features,
            summary=f"Discovered {len(links)} significant causal links "
                    f"among {len(request.selected_features)} variables."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Causal discovery failed: {str(e)}")


@app.post("/api/v1/causal/intervene", response_model=InterventionResponse)
async def simulate_intervention(request: InterventionRequest):
    """
    Intervention simulation endpoint.
    
    This is the core "What-If" analysis API - users can simulate
    interventions and see predicted effects on target variables.
    
    Example:
        POST /api/v1/causal/intervene
        {
            "intervention_variable": "port_strike_events",
            "intervention_value": 10,
            "target_variable": "supply_chain_delay"
        }
    """
    if state.data is None:
        raise HTTPException(status_code=503, detail="No data loaded")
    
    if state.causal_graph is None:
        # Allow intervention simulation even if a full causal graph
        # hasn't been discovered yet. We'll fall back to a simple
        # two-variable graph constructed from the requested
        # intervention and target variables.
        print("⚠️ No causal graph available; using two-variable fallback graph for intervention simulation.")
    
    # Validate variables exist in data
    if request.intervention_variable not in state.data.columns:
        raise HTTPException(status_code=400, detail=f"Intervention variable '{request.intervention_variable}' not found in data")
    if request.target_variable not in state.data.columns:
        raise HTTPException(status_code=400, detail=f"Target variable '{request.target_variable}' not found in data")

    try:
        # Build or update causal inference engine using relevant variables
        relevant_vars = [request.intervention_variable, request.target_variable]

        if state.inference_engine is None:
            # build simple graph using discovery results if available
            import networkx as nx

            G = nx.DiGraph()
            # If causal_graph has explicit edges, try to use them; otherwise assume direct edge
            try:
                # result format: var_names, val_matrix, p_matrix
                var_names = state.causal_graph.get("var_names", []) if isinstance(state.causal_graph, dict) else []
                if var_names and request.intervention_variable in var_names and request.target_variable in var_names:
                    # add edge from discovery
                    G.add_edge(request.intervention_variable, request.target_variable)
                else:
                    G.add_edge(request.intervention_variable, request.target_variable)
            except Exception:
                G.add_edge(request.intervention_variable, request.target_variable)

            # Build inference engine on the two-variable subset
            # Import lazily so tests can monkeypatch the interventions module
            from resilientflow.causal.inference.interventions import CausalInferenceEngine

            df_sub = state.data[relevant_vars].dropna()
            state.inference_engine = CausalInferenceEngine(graph=G, data=df_sub)
        else:
            # Optionally, update inference engine's data if needed
            try:
                state.inference_engine.causal_model = state.inference_engine.causal_model
            except Exception:
                pass

        # Simulate intervention
        baseline_value = float(state.data[request.target_variable].mean())
        estimate = state.inference_engine.estimate_intervention(
            target=request.target_variable,
            intervention_var=request.intervention_variable,
            value=request.intervention_value,
            return_dict=True
        )

        # estimate may be a dict (with stats) or a float for backwards compatibility
        if isinstance(estimate, dict):
            predicted_value = float(estimate.get("mean"))
            predicted_std = float(estimate.get("std", 0.0))
            ci_lower = float(estimate.get("ci_lower", float("nan")))
            ci_upper = float(estimate.get("ci_upper", float("nan")))
        else:
            predicted_value = float(estimate)
            predicted_std = None
            ci_lower = None
            ci_upper = None

        # Safely compute percentage change
        if baseline_value == 0:
            change_pct = float('nan')
        else:
            change_pct = ((predicted_value - baseline_value) / baseline_value) * 100

        explanation = (
            f"Intervening on '{request.intervention_variable}' by setting it to {request.intervention_value} "
            f"is predicted to change '{request.target_variable}' from {baseline_value:.2f} (baseline) to {predicted_value:.2f}."
        )

        return InterventionResponse(
            intervention_variable=request.intervention_variable,
            intervention_value=request.intervention_value,
            target_variable=request.target_variable,
            predicted_effect=predicted_value,
            baseline_value=baseline_value,
            change_percentage=change_pct,
            predicted_std=predicted_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intervention simulation failed: {str(e)}")


@app.post("/api/v1/data/save", response_model=SaveDataResponse)
async def save_data_endpoint():
    """Manually trigger saving the currently loaded timeseries to DB and parquet."""
    if state.data is None:
        raise HTTPException(status_code=503, detail="No data loaded to save")

    try:
        # Save parquet
        output_path = Path("data/03_processed/daily_timeseries.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state.data.to_parquet(output_path)

        # Save to DB
        from resilientflow.data.storage.repository import get_repository
        repo = get_repository()
        repo.save_timeseries(state.data, table_name="daily_timeseries")

        return SaveDataResponse(status="ok", rows=state.data.shape[0], columns=state.data.shape[1], message="Saved to parquet and database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")


@app.get("/api/v1/data/columns")
async def get_available_columns():
    """
    Get list of available columns in loaded data.
    
    Useful for frontend to populate dropdown menus.
    """
    if state.data is None:
        raise HTTPException(status_code=503, detail="No data loaded")
    
    return {
        "columns": state.data.columns.tolist(),
        "count": len(state.data.columns),
        "sample_data": state.data.head(3).to_dict(orient="records")
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print(" ResilientFlow FastAPI Backend")
    print("="*70)
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("📊 Alternative Docs: http://localhost:8000/redoc\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Hot reload for development
    )
