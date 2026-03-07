"""
Random Forest Model Implementation
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


class RandomForestModel:
    """Random Forest model wrapper with cross-validation"""
    
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        self.models = []
        self.feature_names = None
        
    def train_cv(self, X_train, y_train, X_val=None, y_val=None, n_splits=5):
        """Train with time-series cross-validation"""
        print("Training Random Forest with cross-validation...")
        
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        
        # Convert categoricals to integer codes for RF
        if isinstance(X_train, pd.DataFrame):
            X_train_rf = X_train.copy()
            X_val_rf = X_val.copy() if X_val is not None else None
            
            for col in X_train_rf.columns:
                if X_train_rf[col].dtype == 'category':
                    X_train_rf[col] = X_train_rf[col].cat.codes
                    if X_val_rf is not None:
                        X_val_rf[col] = X_val_rf[col].cat.codes
            
            X_train = X_train_rf.values
            if X_val_rf is not None:
                X_val = X_val_rf.values
        else:
            X_train = X_train.values
            if X_val is not None:
                X_val = X_val.values
        
        oof_predictions = np.zeros(len(X_train))
        test_predictions = []
        
        if X_val is not None:
            # Use provided validation set
            print(f"  Training on {len(X_train)} samples, validating on {len(X_val)} samples")
            
            model = RandomForestRegressor(**self.params)
            model.fit(X_train, y_train)
            
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
                
                model = RandomForestRegressor(**self.params)
                model.fit(X_train_fold, y_train_fold)
                
                self.models.append(model)
                oof_predictions[val_idx] = model.predict(X_val_fold)
        
        return oof_predictions, test_predictions
    
    def predict(self, X_test):
        """Predict using all trained models"""
        # Convert categoricals if needed
        if isinstance(X_test, pd.DataFrame):
            X_test_rf = X_test.copy()
            for col in X_test_rf.columns:
                if X_test_rf[col].dtype == 'category':
                    X_test_rf[col] = X_test_rf[col].cat.codes
            X_test = X_test_rf.values
        else:
            X_test = X_test.values
        
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

