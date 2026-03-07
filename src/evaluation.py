"""
Evaluation Module
Computes metrics, feature importance, and performance analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


class Evaluator:
    """Evaluation and metrics computation"""
    
    def __init__(self):
        pass
    
    def compute_metrics(self, y_true, y_pred):
        """Compute regression metrics"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }
    
    def print_metrics(self, y_true, y_pred, label="Validation"):
        """Print formatted metrics"""
        metrics = self.compute_metrics(y_true, y_pred)
        print(f"\n{label} Metrics:")
        print(f"  R² Score:  {metrics['R2']:.6f}")
        print(f"  RMSE:      {metrics['RMSE']:.4f}")
        print(f"  MAE:       {metrics['MAE']:.4f}")
        return metrics
    
    def per_theater_metrics(self, df, y_true_col, y_pred_col):
        """Compute metrics per theater"""
        df = df.copy()
        df["error"] = df[y_true_col] - df[y_pred_col]
        df["squared_error"] = df["error"] ** 2
        df["abs_error"] = df["error"].abs()
        
        theater_metrics = df.groupby("book_theater_id").agg({
            y_true_col: "mean",
            y_pred_col: "mean",
            "squared_error": "mean",
            "abs_error": "mean"
        }).reset_index()
        
        theater_metrics["RMSE"] = np.sqrt(theater_metrics["squared_error"])
        theater_metrics["MAE"] = theater_metrics["abs_error"]
        theater_metrics["R2"] = 1 - (
            theater_metrics["squared_error"] / 
            (theater_metrics[y_true_col] - theater_metrics[y_true_col].mean()) ** 2
        )
        
        return theater_metrics[["book_theater_id", "R2", "RMSE", "MAE"]]
    
    def temporal_metrics(self, df, y_true_col, y_pred_col):
        """Compute metrics by temporal features"""
        df = df.copy()
        df["error"] = (df[y_true_col] - df[y_pred_col]) ** 2
        
        # By day of week
        dow_metrics = df.groupby("day_of_week").agg({
            y_true_col: "mean",
            y_pred_col: "mean",
            "error": "mean"
        })
        dow_metrics["RMSE"] = np.sqrt(dow_metrics["error"])
        
        # By month
        month_metrics = df.groupby("month").agg({
            y_true_col: "mean",
            y_pred_col: "mean",
            "error": "mean"
        })
        month_metrics["RMSE"] = np.sqrt(month_metrics["error"])
        
        return {
            "day_of_week": dow_metrics,
            "month": month_metrics
        }
    
    def feature_importance_analysis(self, feature_importance_dict):
        """Analyze and display feature importance from multiple models"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        for model_name, importance_df in feature_importance_dict.items():
            if importance_df is not None and len(importance_df) > 0:
                print(f"\n{model_name} - Top 15 Features:")
                print(importance_df.head(15).to_string(index=False))
        
        # Aggregate importance across models
        if len(feature_importance_dict) > 1:
            all_features = set()
            for imp_df in feature_importance_dict.values():
                if imp_df is not None:
                    all_features.update(imp_df["feature"].values)
            
            aggregated = pd.DataFrame({"feature": list(all_features)})
            for model_name, imp_df in feature_importance_dict.items():
                if imp_df is not None:
                    aggregated = aggregated.merge(
                        imp_df[["feature", "importance"]],
                        on="feature",
                        how="left",
                        suffixes=("", f"_{model_name}")
                    )
            
            # Average importance
            importance_cols = [col for col in aggregated.columns if "importance" in col]
            aggregated["avg_importance"] = aggregated[importance_cols].mean(axis=1)
            aggregated = aggregated.sort_values("avg_importance", ascending=False)
            
            print("\n\nAggregated Feature Importance (Top 20):")
            print(aggregated[["feature", "avg_importance"]].head(20).to_string(index=False))
            
            return aggregated
        
        return None
    
    def cross_validate(self, X, y, model, n_splits=5):
        """Perform time-series cross-validation"""
        print(f"\nPerforming {n_splits}-fold time-series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model (simplified - actual implementation depends on model type)
            # This is a placeholder - actual CV should be done in model classes
            
            print(f"  Fold {fold+1}/{n_splits}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        return cv_scores

