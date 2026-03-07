"""
Ensemble Module
Implements stacking and weighted blending of multiple models
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


class Ensemble:
    """Ensemble methods for combining model predictions"""
    
    def __init__(self, method='stacking'):
        self.method = method  # 'stacking' or 'weighted'
        self.meta_model = None
        self.optimal_weights = None
        
    def optimize_weights(self, predictions_list, y_true):
        """Optimize blending weights using scipy.optimize"""
        print("Optimizing ensemble weights...")
        
        def objective(weights):
            blended = np.average(predictions_list, axis=0, weights=weights)
            return np.sqrt(mean_squared_error(y_true, blended))
        
        # Initial weights (equal)
        n_models = len(predictions_list)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.optimal_weights = result.x
            print(f"  Optimal weights: {dict(zip(range(n_models), self.optimal_weights))}")
            print(f"  Validation RMSE: {result.fun:.4f}")
        else:
            print("  Warning: Weight optimization failed, using equal weights")
            self.optimal_weights = initial_weights
        
        return self.optimal_weights
    
    def weighted_blend(self, predictions_list, weights=None):
        """Weighted average blending"""
        if weights is None:
            weights = self.optimal_weights if self.optimal_weights is not None else np.ones(len(predictions_list)) / len(predictions_list)
        
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        blended = np.average(predictions_list, axis=0, weights=weights)
        return blended
    
    def stacking(self, oof_predictions_dict, y_train, X_val=None, y_val=None, val_predictions_dict=None):
        """Stacking ensemble with meta-learner"""
        print("Training stacking ensemble...")
        
        # Prepare meta-features from out-of-fold predictions
        meta_features_train = pd.DataFrame(oof_predictions_dict)
        
        # Train meta-learner
        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(meta_features_train, y_train)
        
        print(f"  Meta-learner coefficients: {dict(zip(meta_features_train.columns, self.meta_model.coef_))}")
        
        # Predict on validation if provided
        if val_predictions_dict is not None and X_val is not None:
            meta_features_val = pd.DataFrame(val_predictions_dict)
            val_predictions = self.meta_model.predict(meta_features_val)
            
            # Calculate validation score
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            print(f"  Validation RMSE: {val_rmse:.4f}")
            
            return val_predictions
        
        return None
    
    def predict_stacking(self, test_predictions_dict):
        """Predict using stacking ensemble"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call stacking() first.")
        
        meta_features_test = pd.DataFrame(test_predictions_dict)
        predictions = self.meta_model.predict(meta_features_test)
        return predictions
    
    def blend(self, predictions_dict, y_val=None, method=None):
        """Main blending method"""
        if method is None:
            method = self.method
        
        predictions_list = list(predictions_dict.values())
        model_names = list(predictions_dict.keys())
        
        if method == 'weighted':
            # Optimize weights on validation set
            if y_val is not None:
                self.optimize_weights(predictions_list, y_val)
            
            blended = self.weighted_blend(predictions_list)
            
        elif method == 'stacking':
            # For stacking, we need OOF predictions which should be handled separately
            # This is a simple weighted blend as fallback
            if y_val is not None:
                self.optimize_weights(predictions_list, y_val)
            blended = self.weighted_blend(predictions_list)
            
        else:
            # Simple average
            blended = np.mean(predictions_list, axis=0)
        
        return blended
    
    def get_feature_importance(self):
        """Get feature importance from meta-learner (for stacking)"""
        if self.meta_model is not None and hasattr(self.meta_model, 'coef_'):
            return self.meta_model.coef_
        return None

