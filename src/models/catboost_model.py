"""
CatBoost Model Implementation
"""

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


class CatBoostModel:
    """CatBoost model wrapper with cross-validation"""
    
    def __init__(self, params=None):
        self.params = params or {
            'iterations': 2000,
            'learning_rate': 0.01,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'verbose': False
        }
        self.models = []
        self.feature_names = None
        
    def train_cv(self, X_train, y_train, X_val=None, y_val=None,
                  categorical_features=None, n_splits=5):
        """Train with time-series cross-validation"""
        print("Training CatBoost with cross-validation...")
        
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        
        # Prepare categorical features
        if categorical_features is None:
            categorical_features = []
        cat_indices = [i for i, col in enumerate(X_train.columns) 
                      if col in categorical_features] if isinstance(X_train, pd.DataFrame) else []
        
        def _encode_cats(df):
            """Encode category dtypes to integer codes for numpy conversion"""
            out = df.copy()
            for col in out.select_dtypes(include=['category']).columns:
                out[col] = out[col].cat.codes
            return out

        # Convert to numpy — encode category cols to int codes first
        if isinstance(X_train, pd.DataFrame):
            X_train = _encode_cats(X_train).values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = _encode_cats(X_val).values

        oof_predictions = np.zeros(len(X_train))
        test_predictions = []

        if X_val is not None:
            # Use provided validation set
            print(f"  Training on {len(X_train)} samples, validating on {len(X_val)} samples")

            model = CatBoostRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                cat_features=cat_indices if cat_indices else None,
                verbose=False
            )
            
            self.models.append(model)
            oof_predictions = model.predict(X_train)
            test_predictions = [model.predict(X_val)]
            
        else:
            # Use time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                print(f"  Fold {fold+1}/{n_splits}: Train={len(train_idx)}, Val={len(val_idx)}")
                
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = CatBoostRegressor(**self.params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=(X_val_fold, y_val_fold),
                    early_stopping_rounds=100,
                    cat_features=cat_indices if cat_indices else None,
                    verbose=False
                )
                
                self.models.append(model)
                oof_predictions[val_idx] = model.predict(X_val_fold)
        
        return oof_predictions, test_predictions
    
    def predict(self, X_test):
        """Predict using all trained models"""
        if isinstance(X_test, pd.DataFrame):
            out = X_test.copy()
            for col in out.select_dtypes(include=['category']).columns:
                out[col] = out[col].cat.codes
            X_test = out.values
        
        predictions = []
        for model in self.models:
            preds = model.predict(X_test)
            predictions.append(preds)
        
        # Average predictions from all models
        return np.mean(predictions, axis=0)
    
    def get_feature_importance(self):
        """Get feature importance from the last model"""
        if len(self.models) > 0:
            importance = self.models[-1].feature_importances_
            if self.feature_names:
                return pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
        return None

