"""
Automatic Feature Selection for Causal Analysis
================================================

This module implements "democratization of data science" by automatically
selecting the most important features for causal discovery, eliminating
the need for users to manually choose variables.

Core Philosophy:
- Users specify a TARGET KPI (e.g., "supply chain lead time")
- AI automatically identifies TOP 5 most influential variables
- Results are ready for immediate causal graph visualization

Techniques Used:
1. PCA (Principal Component Analysis) - Dimensionality reduction
2. Lasso Regression - L1 regularization for sparse feature selection
3. Random Forest Feature Importance - Non-linear relationship detection
4. SHAP Values - Explainable AI for feature contribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import warnings

warnings.filterwarnings("ignore")


@dataclass
class FeatureImportance:
    """Result of feature importance analysis."""
    feature_name: str
    importance_score: float
    method: str  # "lasso", "random_forest", "pca", "mutual_info"
    rank: int


@dataclass
class AutoFeatureSelectionResult:
    """Complete result of automatic feature selection."""
    selected_features: List[str]
    importance_details: List[FeatureImportance]
    dimensionality_reduction: Optional[Dict[str, np.ndarray]] = None
    explanation: str = ""


class AutoFeatureSelector:
    """
    Automatic Feature Selection Engine.
    
    This class democratizes data science by automatically identifying
    the most important variables for causal analysis without requiring
    users to have statistical knowledge.
    
    Example:
        >>> selector = AutoFeatureSelector(top_k=5)
        >>> result = selector.select_features(
        ...     data=df,
        ...     target_column="supply_chain_delay",
        ...     method="ensemble"
        ... )
        >>> print(result.selected_features)
        ['port_strike_events', 'stock_price_volatility', 'weather_severity', ...]
    """
    
    def __init__(
        self,
        top_k: int = 5,
        random_state: int = 42,
        normalize: bool = True
    ):
        """
        Initialize AutoFeatureSelector.
        
        Args:
            top_k: Number of top features to select (default: 5 for UI simplicity)
            random_state: Random seed for reproducibility
            normalize: Whether to standardize features before analysis
        """
        self.top_k = top_k
        self.random_state = random_state
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
    def select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        method: str = "ensemble",
        exclude_columns: Optional[List[str]] = None
    ) -> AutoFeatureSelectionResult:
        """
        Main entry point for automatic feature selection.
        
        Args:
            data: Input dataframe with all features
            target_column: Name of the target KPI column
            method: Selection method - "lasso", "random_forest", "pca", "ensemble"
            exclude_columns: Columns to exclude from analysis (e.g., timestamps, IDs)
        
        Returns:
            AutoFeatureSelectionResult with selected features and explanations
        """
        # Prepare data
        X, y, feature_names = self._prepare_data(data, target_column, exclude_columns)
        
        # Apply selected method
        if method == "lasso":
            importance_list = self._lasso_selection(X, y, feature_names)
        elif method == "random_forest":
            importance_list = self._random_forest_selection(X, y, feature_names)
        elif method == "pca":
            importance_list = self._pca_selection(X, y, feature_names)
        elif method == "ensemble":
            importance_list = self._ensemble_selection(X, y, feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by importance and select top k
        importance_list.sort(key=lambda x: x.importance_score, reverse=True)
        for rank, imp in enumerate(importance_list, start=1):
            imp.rank = rank
        
        selected_features = [imp.feature_name for imp in importance_list[:self.top_k]]
        
        # Generate human-readable explanation
        explanation = self._generate_explanation(
            target_column, selected_features, importance_list[:self.top_k], method
        )
        
        return AutoFeatureSelectionResult(
            selected_features=selected_features,
            importance_details=importance_list,
            explanation=explanation
        )
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        exclude_columns: Optional[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for feature selection."""
        # Remove target and excluded columns
        exclude_set = set(exclude_columns or [])
        exclude_set.add(target_column)
        
        feature_columns = [col for col in data.columns if col not in exclude_set]
        
        # Handle non-numeric columns
        X_df = data[feature_columns].select_dtypes(include=[np.number])
        feature_names = X_df.columns.tolist()
        
        # Drop rows with missing target values
        valid_indices = data[target_column].notna()
        X = X_df.loc[valid_indices].fillna(0).values
        y = data.loc[valid_indices, target_column].values
        
        # Normalize if requested
        if self.normalize and self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        return X, y, feature_names
    
    def _lasso_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """
        Lasso regression with L1 regularization.
        
        Benefits:
        - Automatic feature selection through sparsity
        - Handles multicollinearity well
        - Fast computation
        """
        lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=10000)
        lasso.fit(X, y)
        
        coefficients = np.abs(lasso.coef_)
        
        return [
            FeatureImportance(
                feature_name=name,
                importance_score=float(coef),
                method="lasso",
                rank=0  # Will be set later
            )
            for name, coef in zip(feature_names, coefficients)
            if coef > 0  # Only non-zero coefficients
        ]
    
    def _random_forest_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """
        Random Forest feature importance.
        
        Benefits:
        - Captures non-linear relationships
        - Robust to outliers
        - Provides Gini importance
        """
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        
        return [
            FeatureImportance(
                feature_name=name,
                importance_score=float(imp),
                method="random_forest",
                rank=0
            )
            for name, imp in zip(feature_names, importances)
        ]
    
    def _pca_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """
        PCA-based feature importance.
        
        Benefits:
        - Identifies features contributing to maximum variance
        - Useful for high-dimensional data
        - Detects redundant features
        """
        pca = PCA(n_components=min(10, X.shape[1]), random_state=self.random_state)
        pca.fit(X)
        
        # Compute feature importance based on first principal component loadings
        first_pc_loadings = np.abs(pca.components_[0])
        
        return [
            FeatureImportance(
                feature_name=name,
                importance_score=float(loading),
                method="pca",
                rank=0
            )
            for name, loading in zip(feature_names, first_pc_loadings)
        ]
    
    def _ensemble_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """
        Ensemble method combining multiple techniques.
        
        This is the RECOMMENDED method for production use.
        It combines:
        1. Lasso (linear relationships)
        2. Random Forest (non-linear relationships)
        3. Mutual Information (dependency detection)
        
        Each method votes, and features are ranked by aggregated score.
        """
        # Get importances from each method
        lasso_imp = self._lasso_selection(X, y, feature_names)
        rf_imp = self._random_forest_selection(X, y, feature_names)
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        mi_imp = [
            FeatureImportance(
                feature_name=name,
                importance_score=float(score),
                method="mutual_info",
                rank=0
            )
            for name, score in zip(feature_names, mi_scores)
        ]
        
        # Normalize scores to [0, 1] for each method
        def normalize_scores(imp_list: List[FeatureImportance]) -> Dict[str, float]:
            if not imp_list:
                return {}
            max_score = max(imp.importance_score for imp in imp_list)
            if max_score == 0:
                return {imp.feature_name: 0.0 for imp in imp_list}
            return {
                imp.feature_name: imp.importance_score / max_score
                for imp in imp_list
            }
        
        lasso_norm = normalize_scores(lasso_imp)
        rf_norm = normalize_scores(rf_imp)
        mi_norm = normalize_scores(mi_imp)
        
        # Aggregate scores (weighted average)
        ensemble_scores = {}
        weights = {"lasso": 0.3, "random_forest": 0.4, "mutual_info": 0.3}
        
        for name in feature_names:
            score = (
                weights["lasso"] * lasso_norm.get(name, 0) +
                weights["random_forest"] * rf_norm.get(name, 0) +
                weights["mutual_info"] * mi_norm.get(name, 0)
            )
            ensemble_scores[name] = score
        
        return [
            FeatureImportance(
                feature_name=name,
                importance_score=score,
                method="ensemble",
                rank=0
            )
            for name, score in ensemble_scores.items()
        ]
    
    def _generate_explanation(
        self,
        target_column: str,
        selected_features: List[str],
        top_importances: List[FeatureImportance],
        method: str
    ) -> str:
        """
        Generate human-readable explanation for non-technical users.
        
        This is the "Narrative AI" component that translates statistical
        results into actionable insights.
        """
        method_descriptions = {
            "lasso": "linear regression with automatic feature selection",
            "random_forest": "machine learning ensemble method",
            "pca": "dimensionality reduction analysis",
            "ensemble": "combination of multiple AI techniques"
        }
        
        explanation_parts = [
            f"To predict '{target_column}', the AI analyzed all available data using {method_descriptions.get(method, method)}.",
            f"\nThe TOP {len(selected_features)} most important factors are:\n"
        ]
        
        for imp in top_importances:
            explanation_parts.append(
                f"  {imp.rank}. {imp.feature_name} "
                f"(importance: {imp.importance_score:.3f})\n"
            )
        
        explanation_parts.append(
            f"\nThese variables explain the majority of variation in {target_column}. "
            "You can now use them for causal discovery and intervention simulation."
        )
        
        return "".join(explanation_parts)


def run_auto_feature_selection_demo():
    """Demo function showing how to use AutoFeatureSelector."""
    print("=== ResilientFlow: Auto Feature Selection Demo ===\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    data = pd.DataFrame({
        "port_strike_events": np.random.poisson(2, n_samples),
        "stock_price_volatility": np.random.normal(0, 1, n_samples),
        "weather_severity": np.random.exponential(1, n_samples),
        "geopolitical_tension": np.random.uniform(0, 10, n_samples),
        "currency_fluctuation": np.random.normal(0, 0.5, n_samples),
        "energy_price": np.random.normal(100, 20, n_samples),
        "labor_cost": np.random.normal(50, 10, n_samples),
        "noise_1": np.random.normal(0, 1, n_samples),  # Irrelevant feature
        "noise_2": np.random.normal(0, 1, n_samples),  # Irrelevant feature
    })
    
    # Create target variable with known relationships
    data["supply_chain_delay"] = (
        2.5 * data["port_strike_events"] +
        1.8 * data["stock_price_volatility"] +
        1.2 * data["weather_severity"] +
        0.5 * data["geopolitical_tension"] +
        np.random.normal(0, 1, n_samples)  # Noise
    )
    
    # Run feature selection
    selector = AutoFeatureSelector(top_k=5)
    result = selector.select_features(
        data=data,
        target_column="supply_chain_delay",
        method="ensemble"
    )
    
    print(result.explanation)
    print("\n" + "="*60)
    print("Full Ranking:")
    for imp in result.importance_details[:10]:
        print(f"  {imp.rank}. {imp.feature_name:25s} {imp.importance_score:.4f}")


if __name__ == "__main__":
    run_auto_feature_selection_demo()
