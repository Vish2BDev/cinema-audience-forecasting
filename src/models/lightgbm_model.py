"""
LightGBM Model Implementation
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")


class LightGBMModel:
    """LightGBM model wrapper with cross-validation"""
    
    def __init__(self, params=None):
        self.params = params or {
            'n_estimators': 2000,
            'max_depth': 12,
            'learning_rate': 0.02,
            'num_leaves': 127,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.models = []
        self.feature_names = None
        
    def train_cv(self, X_train, y_train, X_val=None, y_val=None, 
                 categorical_features=None, n_splits=5):
        """Train with time-series cross-validation"""
        print("Training LightGBM with cross-validation...")
        
        self.feature_names = X_train.columns.tolist()
        
        # Prepare categorical features
        if categorical_features is None:
            categorical_features = []
        cat_indices = [i for i, col in enumerate(X_train.columns) 
                      if col in categorical_features]
        
        def _encode_cats(df, ref_df=None):
            """Encode category dtypes to integer codes for numpy conversion"""
            out = df.copy()
            for col in out.select_dtypes(include=['category']).columns:
                if ref_df is not None and col in ref_df.columns:
                    out[col] = pd.Categorical(out[col], categories=ref_df[col].cat.categories).codes
                else:
                    out[col] = out[col].cat.codes
            return out

        # Convert to numpy — encode category cols to int codes first
        if isinstance(X_train, pd.DataFrame):
            X_train_enc = _encode_cats(X_train)
            X_train = X_train_enc.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = _encode_cats(X_val).values

        oof_predictions = np.zeros(len(X_train))
        test_predictions = []

        if X_val is not None:
            # Use provided validation set
            print(f"  Training on {len(X_train)} samples, validating on {len(X_val)} samples")

            model = lgb.LGBMRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=0)
                ],
                categorical_feature=cat_indices if cat_indices else 'auto'
            )
            
            self.models.append(model)
            oof_predictions = model.predict(X_train)
            test_predictions = [model.predict(X_val)]
            
        else:
            # Use time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                print(f"  Fold {fold+1}/{n_splits}: Train={len(train_idx)}, Val={len(val_idx)}")

                X_train_fold = X_train[train_idx].copy()
                X_val_fold   = X_train[val_idx].copy()
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = lgb.LGBMRegressor(**self.params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=100, verbose=False),
                        lgb.log_evaluation(period=0)
                    ],
                    categorical_feature=cat_indices if cat_indices else 'auto'
                )
                
                self.models.append(model)
                oof_predictions[val_idx] = model.predict(X_val_fold)
        
        return oof_predictions, test_predictions
    
    def predict(self, X_test):
        """Predict using all trained models"""
        if isinstance(X_test, pd.DataFrame):
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

