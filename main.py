"""
Main Pipeline
End-to-end pipeline: preprocessing → feature engineering → model training → ensemble → post-processing → submission
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models.lightgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.random_forest_model import RandomForestModel
try:
    from models.prophet_model import ProphetModel
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available, skipping Prophet model")
from ensemble import Ensemble
from post_processing import PostProcessor
from evaluation import Evaluator
import warnings
warnings.filterwarnings("ignore")


def select_features(df):
    """Select features for modeling"""
    # Exclude non-feature columns
    exclude_cols = [
        "ID", "book_theater_id", "show_date", "audience_count",
        "theater_dow_interaction", "area_month_interaction"  # String interaction features
    ]
    
    # Get all feature columns (exclude string columns)
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            # Exclude string columns (keep only numeric and category)
            if df[col].dtype in ['int64', 'float64', 'category'] or col in ["day_of_week", "month", "theater_area", "theater_type", "geo_cluster"]:
                feature_cols.append(col)
    
    # Categorical features (for LightGBM/CatBoost)
    categorical_features = [
        "theater_area", "theater_type", "geo_cluster", "day_of_week", "month"
    ]
    categorical_features = [c for c in categorical_features if c in df.columns and c in feature_cols]
    
    return feature_cols, categorical_features


def main():
    print("="*80)
    print("CINEMA AUDIENCE FORECASTING - ELITE KAGGLE SOLUTION")
    print("="*80)
    
    # ========================================================================
    # PHASE 1: DATA PREPROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor(data_dir=".", val_start="2024-01-01", val_end="2024-02-28")
    data_dict = preprocessor.process()
    
    train_df = data_dict['train']
    val_df = data_dict['val']
    test_df = data_dict['test']
    
    # ========================================================================
    # PHASE 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*80)
    
    engineer = FeatureEngineer(
        lag_days=[1, 2, 3, 7, 14, 21, 28, 30, 60, 90],
        rolling_windows=[7, 14, 30, 60, 90]
    )
    
    # Engineer features for all data
    full_df = engineer.process(
        data_dict['full'],
        data_dict['bookings_bn'],
        data_dict['bookings_pos']
    )
    
    # Split back into train/val/test
    train_df = full_df[(full_df["audience_count"].notna()) & 
                       (full_df["show_date"] < pd.to_datetime("2024-01-01"))].copy()
    val_df = full_df[(full_df["audience_count"].notna()) & 
                     (full_df["show_date"] >= pd.to_datetime("2024-01-01")) &
                     (full_df["show_date"] <= pd.to_datetime("2024-02-28"))].copy()
    test_df = full_df[full_df["audience_count"].isna()].copy()
    
    # Ensure test_df has ID column
    if "ID" not in test_df.columns:
        test_df["ID"] = test_df["book_theater_id"].astype(str) + "_" + test_df["show_date"].dt.strftime("%Y-%m-%d")
    
    # Sort by date for time-series models
    train_df = train_df.sort_values("show_date")
    val_df = val_df.sort_values("show_date")
    test_df = test_df.sort_values("show_date")
    
    # ========================================================================
    # PHASE 3: PREPARE FEATURES
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: PREPARING FEATURES")
    print("="*80)
    
    feature_cols, categorical_features = select_features(full_df)
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df["audience_count"].copy()
    X_val = val_df[feature_cols].copy()
    y_val = val_df["audience_count"].copy()
    X_test = test_df[feature_cols].copy()
    
    # Convert categorical columns to category type
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    # Fill any remaining NaN values in numerical columns only
    for col in X_train.columns:
        if col not in categorical_features and X_train[col].dtype in ['float64', 'int64']:
            X_train[col] = X_train[col].fillna(0)
            X_val[col] = X_val[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
    
    print(f"Training features: {len(feature_cols)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Train set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # ========================================================================
    # PHASE 4: MODEL TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: MODEL TRAINING")
    print("="*80)
    
    # Train all models
    models = {}
    oof_predictions = {}
    val_predictions = {}
    test_predictions = {}
    
    # LightGBM
    print("\n--- LightGBM ---")
    lgb_model = LightGBMModel()
    oof_lgb, val_preds_lgb = lgb_model.train_cv(
        X_train, y_train, X_val, y_val,
        categorical_features=categorical_features
    )
    oof_predictions['lgb'] = oof_lgb
    val_predictions['lgb'] = val_preds_lgb[0] if val_preds_lgb else None
    test_predictions['lgb'] = lgb_model.predict(X_test)
    models['lgb'] = lgb_model
    
    # XGBoost
    print("\n--- XGBoost ---")
    xgb_model = XGBoostModel()
    oof_xgb, val_preds_xgb = xgb_model.train_cv(X_train, y_train, X_val, y_val)
    oof_predictions['xgb'] = oof_xgb
    val_predictions['xgb'] = val_preds_xgb[0] if val_preds_xgb else None
    test_predictions['xgb'] = xgb_model.predict(X_test)
    models['xgb'] = xgb_model
    
    # CatBoost
    print("\n--- CatBoost ---")
    cat_model = CatBoostModel()
    oof_cat, val_preds_cat = cat_model.train_cv(
        X_train, y_train, X_val, y_val,
        categorical_features=categorical_features
    )
    oof_predictions['cat'] = oof_cat
    val_predictions['cat'] = val_preds_cat[0] if val_preds_cat else None
    test_predictions['cat'] = cat_model.predict(X_test)
    models['cat'] = cat_model
    
    # Random Forest
    print("\n--- Random Forest ---")
    rf_model = RandomForestModel()
    oof_rf, val_preds_rf = rf_model.train_cv(X_train, y_train, X_val, y_val)
    oof_predictions['rf'] = oof_rf
    val_predictions['rf'] = val_preds_rf[0] if val_preds_rf else None
    test_predictions['rf'] = rf_model.predict(X_test)
    models['rf'] = rf_model
    
    # Prophet (optional - can be slow)
    if PROPHET_AVAILABLE:
        print("\n--- Prophet ---")
        prophet_model = ProphetModel()
        try:
            oof_prophet, val_pred_prophet = prophet_model.get_oof_predictions(train_df, val_df)
            oof_predictions['prophet'] = oof_prophet
            val_predictions['prophet'] = val_pred_prophet
            test_predictions['prophet'] = prophet_model.predict(test_df)
            models['prophet'] = prophet_model
        except Exception as e:
            print(f"  Warning: Prophet model failed: {e}")
            print("  Continuing without Prophet...")
    else:
        print("\n--- Prophet ---")
        print("  Skipping Prophet (not installed)")
    
    # ========================================================================
    # PHASE 5: ENSEMBLE
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: ENSEMBLE")
    print("="*80)
    
    # Remove None predictions
    val_predictions = {k: v for k, v in val_predictions.items() if v is not None}
    test_predictions = {k: v for k, v in test_predictions.items() if v is not None}
    
    # Stacking ensemble
    ensemble = Ensemble(method='stacking')
    
    # Train stacking meta-learner
    ensemble.stacking(oof_predictions, y_train, X_val, y_val, val_predictions)
    
    # Generate final predictions
    final_test_predictions = ensemble.predict_stacking(test_predictions)
    
    # Also try weighted blending as backup
    weighted_predictions = ensemble.blend(test_predictions, y_val, method='weighted')
    
    # Use stacking if available, otherwise weighted
    if final_test_predictions is not None:
        final_predictions = final_test_predictions
    else:
        final_predictions = weighted_predictions
    
    # ========================================================================
    # PHASE 6: POST-PROCESSING
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 6: POST-PROCESSING")
    print("="*80)
    
    postprocessor = PostProcessor(train_df=train_df)
    final_predictions = postprocessor.process(
        final_predictions,
        test_df,
        apply_smoothing_flag=True,
        apply_dow_consistency=True,
        round_predictions_flag=False
    )
    
    # ========================================================================
    # PHASE 7: EVALUATION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 7: EVALUATION")
    print("="*80)
    
    evaluator = Evaluator()
    
    # Evaluate individual models on validation set
    print("\nIndividual Model Performance (Validation):")
    for model_name, preds in val_predictions.items():
        if preds is not None:
            evaluator.print_metrics(y_val, preds, label=model_name)
    
    # Evaluate ensemble on validation set
    if final_test_predictions is not None:
        val_ensemble_preds = ensemble.predict_stacking(val_predictions)
        if val_ensemble_preds is not None:
            evaluator.print_metrics(y_val, val_ensemble_preds, label="Ensemble (Stacking)")
    
    # Feature importance
    feature_importance_dict = {}
    for model_name, model in models.items():
        if hasattr(model, 'get_feature_importance'):
            feature_importance_dict[model_name] = model.get_feature_importance()
    
    evaluator.feature_importance_analysis(feature_importance_dict)
    
    # ========================================================================
    # PHASE 8: GENERATE SUBMISSION
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 8: GENERATING SUBMISSION")
    print("="*80)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        "ID": test_df["ID"].values,
        "audience_count": final_predictions
    })
    
    # Ensure all IDs from sample submission are present
    sample_sub = pd.read_csv("sample_submission.csv")
    submission_df = sample_sub[["ID"]].merge(submission_df, on="ID", how="left")
    submission_df["audience_count"] = submission_df["audience_count"].fillna(0)
    
    # Save submission
    submission_df.to_csv("submission.csv", index=False)
    print(f"\n[OK] Submission saved to submission.csv")
    print(f"   Total predictions: {len(submission_df):,}")
    print(f"   Prediction range: [{submission_df['audience_count'].min():.2f}, {submission_df['audience_count'].max():.2f}]")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

