# Cinema Audience Forecasting - Improvement Plan
## Current Score: 0.354 → Target: 0.95+ R²

## Critical Gaps in Current Code

### 1. **Missing Target Encoding** (Expected: +15-20% improvement)
**Current**: Only using `avg_theater_audience` (global mean)
**Missing**: 
- Theater-specific day-of-week averages (Sunday avg per theater)
- Theater-month averages (seasonal patterns per theater)
- Area-day-of-week interactions (spatial-temporal patterns)
- Time-decayed target encoding (recent data weighted more)

**Impact**: This is the BIGGEST gap. Target encoding captures theater-specific patterns that global features miss.

### 2. **Insufficient Lag Features** (Expected: +5-10% improvement)
**Current**: Lags [1, 7, 14, 21, 28]
**Missing**:
- 30, 60, 90 day lags (monthly/quarterly cycles)
- Lag ratios (lag_7/lag_14, lag_30/lag_60) - captures trends
- Lag differences (lag_7 - lag_14) - acceleration/deceleration
- Same-day-of-week lags (last Sunday, last Monday) - weekly patterns

### 3. **Weak Booking Features** (Expected: +8-12% improvement)
**Current**: Only `total_tickets` (sum of bookings)
**Missing**:
- Advance booking patterns (mean/std hours before show)
- Booking velocity (rolling mean of bookings over 7/14/30 days)
- Peak booking hour (temporal preference indicator)
- Booking frequency (count of unique booking times per show)
- Last 7 days booking sum (recent demand signal)
- Booking trend (increasing/decreasing booking patterns)

### 4. **No Holiday/Event Features** (Expected: +3-5% improvement)
**Missing**: Indian holidays (Holi, Diwali, Eid, Christmas, New Year)
- `is_holiday` flag
- `days_to_holiday`, `days_since_holiday`
- `is_holiday_week` (week before/after holidays)

### 5. **Missing Date Handling** (Expected: +2-4% improvement)
**Current**: Not handling theaters that don't operate every day
**Missing**:
- `days_since_last_show` (theater closure detection)
- `is_operating_day` (binary - does theater operate on this day-of-week?)
- `theater_operating_frequency` (days per week)
- `consecutive_missing_days` (closure periods)

### 6. **Limited Model Diversity** (Expected: +10-15% improvement)
**Current**: LightGBM + RandomForest (simple 70/30 blend)
**Missing**:
- XGBoost (different regularization, captures different patterns)
- CatBoost (excellent categorical handling)
- Prophet (time-series expert, per-theater seasonality)
- Stacking ensemble (meta-learner on OOF predictions)

### 7. **No Post-Processing** (Expected: +2-5% improvement)
**Missing**:
- Floor at 0 (negative predictions)
- Cap at theater historical max (capacity constraint)
- Temporal smoothing (3-day rolling median)
- Missing date handling (predict 0 if theater never operated on that day)

### 8. **No Interaction Features** (Expected: +3-5% improvement)
**Missing**:
- `theater_id × day_of_week` (some theaters are weekend-focused)
- `area × month` (seasonal patterns vary by geography)
- `lag_7 × is_weekend` (previous weekend predicts current weekend)
- `roll_mean_7 × theater_type` (type-specific trends)

### 9. **Weak Statistical Features** (Expected: +2-3% improvement)
**Current**: Only `avg_theater_audience`
**Missing**:
- Theater median, std, min, max, q25, q75
- Coefficient of variation (std/mean)
- Skewness, kurtosis (distribution shape)

### 10. **No Hyperparameter Tuning** (Expected: +3-5% improvement)
**Current**: Fixed hyperparameters
**Missing**: Optuna/Bayesian optimization for each model

---

## Implementation Plan

### Phase 1: Advanced Feature Engineering (Target: +25-30% → 0.60-0.65 R²)

#### 1.1 Target Encoding (HIGHEST PRIORITY)
```python
# Theater-day-of-week target encoding (with time-based CV)
def create_target_encoding(df, group_cols, target_col, suffix):
    df = df.sort_values('show_date')
    df[f'te_{suffix}'] = 0
    
    # Use expanding window (only past data)
    for idx in range(len(df)):
        mask = (df.index < idx) & (df[target_col].notna())
        for group in group_cols:
            group_mask = mask & (df[group] == df.loc[idx, group])
            if group_mask.sum() > 0:
                df.loc[idx, f'te_{suffix}'] = df.loc[group_mask, target_col].mean()
    
    return df

# Create encodings:
# - Theater × Day-of-week
# - Theater × Month  
# - Area × Day-of-week
# - Area × Month
```

#### 1.2 Extended Lag Features
```python
# Add to existing lag loop:
for lag in [30, 60, 90]:
    full_df[f"aud_lag_{lag}"] = full_df.groupby("book_theater_id")["audience_count"].shift(lag)

# Lag ratios (trend indicators)
full_df["lag_ratio_7_14"] = full_df["aud_lag_7"] / (full_df["aud_lag_14"] + 1e-6)
full_df["lag_ratio_30_60"] = full_df["aud_lag_30"] / (full_df["aud_lag_60"] + 1e-6)

# Same day-of-week lags
full_df["same_dow_lag_7"] = full_df.groupby(["book_theater_id", "day_of_week"])["audience_count"].shift(1)
full_df["same_dow_lag_14"] = full_df.groupby(["book_theater_id", "day_of_week"])["audience_count"].shift(2)
```

#### 1.3 Advanced Booking Features
```python
# From bookings_bn and bookings_pos
bookings_bn["advance_hours"] = (
    pd.to_datetime(bookings_bn["show_datetime"]) - 
    pd.to_datetime(bookings_bn["booking_datetime"])
).dt.total_seconds() / 3600

booking_features = bookings_bn.groupby(["book_theater_id", "show_date"]).agg({
    "tickets_booked": ["sum", "mean", "count"],
    "advance_hours": ["mean", "std", "min", "max"]
}).reset_index()

# Booking velocity (rolling)
full_df["booking_velocity_7"] = full_df.groupby("book_theater_id")["total_tickets"].shift(1).rolling(7).mean()
full_df["booking_velocity_14"] = full_df.groupby("book_theater_id")["total_tickets"].shift(1).rolling(14).mean()
```

#### 1.4 Holiday Features
```python
# Indian holidays 2023-2024
holidays = {
    "2023-01-26": "Republic Day",
    "2023-03-08": "Holi",
    "2023-04-14": "Ambedkar Jayanti",
    "2023-08-15": "Independence Day",
    "2023-10-24": "Dussehra",
    "2023-11-12": "Diwali",
    "2023-12-25": "Christmas",
    "2024-01-01": "New Year",
    "2024-03-25": "Holi",
    "2024-04-14": "Ambedkar Jayanti",
    # Add more...
}

full_df["is_holiday"] = full_df["show_date"].isin([pd.to_datetime(d) for d in holidays.keys()]).astype(int)
```

#### 1.5 Missing Date Handling
```python
# Days since last show
full_df["days_since_last_show"] = (
    full_df.groupby("book_theater_id")["show_date"]
    .diff().dt.days.fillna(999)
)

# Operating day detection
theater_dow_pattern = (
    full_df[full_df["audience_count"].notna()]
    .groupby(["book_theater_id", "day_of_week"])["audience_count"]
    .count() > 0
).reset_index()
theater_dow_pattern.columns = ["book_theater_id", "day_of_week", "is_operating_day"]
full_df = full_df.merge(theater_dow_pattern, on=["book_theater_id", "day_of_week"], how="left")
```

### Phase 2: Additional Models (Target: +10-15% → 0.75-0.80 R²)

#### 2.1 XGBoost Model
```python
import xgboost as xgb

xgb_params = {
    'n_estimators': 2000,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}

model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              early_stopping_rounds=100,
              verbose=False)
```

#### 2.2 CatBoost Model
```python
from catboost import CatBoostRegressor

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_state': 42,
    'verbose': False
}

model_cat = CatBoostRegressor(**cat_params)
model_cat.fit(X_train_fold, y_train_fold,
              eval_set=(X_val_fold, y_val_fold),
              early_stopping_rounds=100,
              cat_features=categorical_features)
```

#### 2.3 Prophet Model (Per-Theater)
```python
from prophet import Prophet

# For top 50 theaters by volume, train individual Prophet models
# For others, cluster by area/type and train cluster-level models

def train_prophet_per_theater(theater_id, df_theater):
    df_prophet = df_theater[['show_date', 'audience_count']].rename(
        columns={'show_date': 'ds', 'audience_count': 'y'}
    )
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df  # Indian holidays
    )
    model.fit(df_prophet)
    return model
```

### Phase 3: Advanced Ensemble (Target: +5-10% → 0.85-0.90 R²)

#### 3.1 Stacking Ensemble
```python
# Collect out-of-fold predictions from all models
oof_predictions = {
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat,
    'rf': oof_rf
}

# Meta-features for stacking
meta_features = pd.DataFrame(oof_predictions)

# Train meta-learner (Ridge or LightGBM)
from sklearn.linear_model import Ridge

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features, y_train)

# Predict on test
test_meta_features = pd.DataFrame({
    'lgb': test_preds_lgb,
    'xgb': test_preds_xgb,
    'cat': test_preds_cat,
    'rf': test_preds_rf
})
final_preds = meta_model.predict(test_meta_features)
```

#### 3.2 Optimized Weight Blending
```python
from scipy.optimize import minimize

def objective(weights, preds_list, y_true):
    blended = np.average(preds_list, axis=0, weights=weights)
    return np.sqrt(np.mean((y_true - blended) ** 2))

# Optimize weights on validation set
initial_weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights
result = minimize(
    objective,
    initial_weights,
    args=(val_predictions_list, y_val),
    method='SLSQP',
    bounds=[(0, 1)] * len(initial_weights),
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
)

optimal_weights = result.x
final_preds = np.average([pred_lgb, pred_xgb, pred_cat, pred_rf], 
                         axis=0, weights=optimal_weights)
```

### Phase 4: Post-Processing (Target: +2-5% → 0.90-0.95 R²)

#### 4.1 Constraints
```python
# Floor at 0
final_preds = np.maximum(final_preds, 0)

# Cap at theater historical max
theater_max = train_df.groupby("book_theater_id")["audience_count"].max()
final_preds = np.minimum(final_preds, theater_max[test_df["book_theater_id"]].values)

# Handle missing operating days
final_preds[test_df["is_operating_day"] == 0] = 0
```

#### 4.2 Temporal Smoothing
```python
# 3-day rolling median
test_df["pred"] = final_preds
test_df = test_df.sort_values(["book_theater_id", "show_date"])
test_df["pred_smooth"] = test_df.groupby("book_theater_id")["pred"].transform(
    lambda x: x.rolling(3, center=True, min_periods=1).median()
)
final_preds = test_df["pred_smooth"].values
```

### Phase 5: Hyperparameter Tuning (Target: +3-5% → 0.95+ R²)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
    }
    
    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    preds = model.predict(X_val_fold)
    return np.sqrt(np.mean((y_val_fold - preds) ** 2))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

---

## Expected Performance Progression

| Phase | Improvement | Cumulative Score |
|-------|-------------|------------------|
| Baseline | - | 0.354 |
| Phase 1: Advanced Features | +25-30% | 0.60-0.65 |
| Phase 2: Additional Models | +10-15% | 0.75-0.80 |
| Phase 3: Advanced Ensemble | +5-10% | 0.85-0.90 |
| Phase 4: Post-Processing | +2-5% | 0.90-0.95 |
| Phase 5: Hyperparameter Tuning | +3-5% | **0.95+** |

---

## Implementation Priority

1. **IMMEDIATE (Biggest Impact)**:
   - Target encoding (theater-day-of-week, theater-month)
   - Extended lag features (30, 60, 90 days)
   - Advanced booking features

2. **HIGH PRIORITY**:
   - Add XGBoost and CatBoost models
   - Implement stacking ensemble
   - Holiday features

3. **MEDIUM PRIORITY**:
   - Missing date handling
   - Post-processing constraints
   - Interaction features

4. **FINE-TUNING**:
   - Hyperparameter tuning
   - Feature selection
   - Temporal smoothing

---

## Key Files to Create/Modify

1. `improved_model.py` - Main script with all improvements
2. `feature_engineering_advanced.py` - All new features
3. `target_encoding.py` - Time-safe target encoding
4. `ensemble_stacking.py` - Stacking implementation
5. `post_processing.py` - Constraints and smoothing
6. `hyperparameter_tuning.py` - Optuna optimization

---

## Critical Success Factors

1. **Target Encoding**: Must use time-based CV to prevent leakage
2. **Model Diversity**: Each model should capture different patterns
3. **Validation Strategy**: Use 2024-01-01 to 2024-02-28 as holdout (matches test period)
4. **Missing Dates**: Properly handle theaters that don't operate every day
5. **Ensemble Weights**: Optimize on validation set, not training set

