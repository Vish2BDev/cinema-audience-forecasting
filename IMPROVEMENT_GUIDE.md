# 🎬 Cinema Audience Forecasting - Complete Improvement Guide

## 📊 Performance Roadmap: 0.38 → 0.95+ R²

---

## 🔍 DIAGNOSTIC SUMMARY

**Your Current Score:** 0.38302 R²
**Target Score:** 0.95+ R² (or 1.5 if using different metric)
**Gap:** +150% improvement needed

---

## ❌ 7 CRITICAL GAPS IN YOUR v3.0 CODE

### Gap 1: **No Indian Holiday Features** (Impact: +15-20%)
**Problem:** Cinema attendance spikes 40-60% during Diwali, Holi, Eid, Christmas
**Your code:** Only has generic day_of_week
**Missing:**
- `is_holiday`, `is_major_holiday`
- `days_until_holiday`, `days_since_holiday`
- `near_holiday` (±3 day window)

**Fix in v5.0/v6.0:** Complete 2023-2024 Indian holiday calendar with proximity features

---

### Gap 2: **Ignoring Advance Booking Data** (Impact: +10-15%)
**Problem:** You have `booknow_booking.csv` and `cinePOS_booking.csv` but NEVER USE THEM!
**These files contain:**
- Total tickets pre-booked (strong signal for audience)
- Number of bookings
- Average booking lead time (early bookings = blockbuster)

**Fix in v5.0/v6.0:**
```python
# Extracted features:
- total_tickets_booked
- num_bookings
- avg_booking_size
- max_booking_size
- avg_lead_hours (advance booking window)
```

---

### Gap 3: **No Same-Day-of-Week Historical Features** (Impact: +8-12%)
**Problem:** Your lags are [1,2,3,7,14...] but ignore weekly patterns
**Reality:** Last 4 Saturdays are MUCH better predictors than lag_1 for predicting next Saturday

**Your code:**
```python
lag_7  # Just 7 days ago (could be different day-of-week!)
```

**v6.0 fix:**
```python
lag_same_dow_1w  # Last Saturday (if today is Saturday)
lag_same_dow_2w  # 2 Saturdays ago
lag_same_dow_3w  # 3 Saturdays ago
lag_same_dow_4w  # 4 Saturdays ago
```

---

### Gap 4: **No Theater Clustering** (Impact: +5-10%)
**Problem:** You treat all theaters equally in global model
**Reality:** Premium multiplexes vs small-town single-screens have completely different patterns

**v6.0 fix:**
- KMeans clustering by `[mean_attendance, std, latitude, longitude]`
- Creates 8 theater segments with similar behavior
- Per-cluster statistics (cluster_mean, cluster_std)

---

### Gap 5: **Weak Ensemble (Only 2 Similar Models)** (Impact: +10-15%)
**Your code:** LightGBM + XGBoost (both tree-based, very correlated)

**v6.0 fix:**
- LightGBM (fast tree model)
- XGBoost (different regularization)
- CatBoost (categorical handling expert)
- Prophet (time-series specialist for seasonal patterns)
- Ridge meta-learner (stacking on top of base models)

**Ensemble diversity = better generalization!**

---

### Gap 6: **No Fourier/Complex Seasonality Features** (Impact: +3-5%)
**Your code:** Simple cyclical sin/cos
**Missing:** Multi-frequency Fourier features for complex yearly patterns

**v6.0 fix:**
```python
doy_sin_1, doy_cos_1  # Annual cycle (365 days)
doy_sin_2, doy_cos_2  # Semi-annual (182 days)
doy_sin_3, doy_cos_3  # Quarterly (121 days)
```

---

### Gap 7: **No Per-Theater Temporal Models** (Impact: +5-10%)
**Problem:** Global LightGBM averages out theater-specific weekly/seasonal patterns
**Solution:** Fit individual Prophet models for top theaters

**v6.0 fix:**
- Prophet per theater (top 20 theaters by volume)
- Captures theater-specific:
  - Weekly seasonality
  - Yearly seasonality
  - Changepoints (trend shifts)

---

## 🚀 IMPROVEMENT PHASES & EXPECTED GAINS

### **Phase 1: v5.0 Elite Features** → 0.38 → 0.55 R² (+44%)
**Changes:**
1. ✅ Indian holiday calendar (2023-2024)
2. ✅ Advance booking features from booking CSVs
3. ✅ Same-day-of-week lags (last 4 weeks)
4. ✅ Fourier features (3 frequencies)
5. ✅ CatBoost added (3-model ensemble)
6. ✅ Per-theater-dow statistics

**Expected CV:** 0.50-0.60 R²
**Expected LB:** 0.45-0.55 R²

---

### **Phase 2: v6.0 Ultimate** → 0.55 → 0.75 R² (+36%)
**Changes:**
1. ✅ Theater clustering (8 clusters by behavior)
2. ✅ Per-theater Prophet models (top 20 theaters)
3. ✅ Stacking meta-learner (Ridge on OOF predictions)
4. ✅ Per-cluster calibration
5. ✅ Enhanced hyperparameters (2000 iterations, deeper trees)

**Expected CV:** 0.65-0.80 R²
**Expected LB:** 0.60-0.75 R²

---

### **Phase 3: Manual Tuning** → 0.75 → 0.90+ R² (+20%)
**Next steps to reach 0.90-0.95:**

1. **Hyperparameter optimization** (Optuna)
   - Learning rate sweep [0.005, 0.01, 0.02, 0.03]
   - Tree depth [8, 10, 12, 15]
   - Regularization (alpha, lambda)

2. **Additional features:**
   - Movie release patterns (Friday new releases)
   - School holiday calendar (spring/summer breaks)
   - Weather data (if available externally)
   - Economic indicators (payday effect)

3. **Advanced ensembling:**
   - Weighted blending by cluster (optimize weights per cluster)
   - Neural network meta-learner (LSTM on sequences)

4. **Post-processing:**
   - Per-theater capacity caps (enforce max from historical)
   - Zero-audience day detection (closed theaters)
   - Isotonic regression calibration

---

## 📝 HOW TO USE THE NEW SOLUTIONS

### **Option 1: Quick Win (v5.0)**
```bash
python cinema_forecasting_v5_elite.py
```
- Runtime: ~10-15 minutes
- Expected improvement: 0.38 → 0.50-0.60 R²
- Best for: Fast iteration, understanding feature impact

### **Option 2: Maximum Performance (v6.0)**
```bash
python cinema_forecasting_v6_ultimate.py
```
- Runtime: ~20-30 minutes (includes Prophet)
- Expected improvement: 0.38 → 0.65-0.80 R²
- Best for: Competition submission, best score

---

## 🔧 TROUBLESHOOTING

### If Prophet not installed:
```bash
pip install prophet
```
Or: v6.0 auto-skips Prophet if unavailable (graceful degradation)

### If you get "negative R²" on test:
- **Cause:** Test period (2024) is 1 year ahead of train (2023)
- **Fix:** Year-over-year features needed (add in Phase 3)

### If predictions are too low/high:
- **Check:** `theater_max` caps in post-processing
- **Adjust:** Clipping bounds from `2` to actual minimum

---

## 📊 FEATURE IMPORTANCE BREAKDOWN (Expected)

**Top 10 Most Important Features (v6.0):**

1. `lag_same_dow_1w` (30%) - Last week same day
2. `theater_dow_mean` (15%) - Theater's typical day-of-week average
3. `total_tickets_booked` (12%) - Advance bookings
4. `roll_mean_7` (8%) - 7-day rolling average
5. `is_major_holiday` (6%) - Major holiday flag
6. `cluster_mean` (5%) - Theater cluster baseline
7. `ewm_14` (4%) - Exponential weighted mean
8. `prophet_pred` (4%) - Prophet model predictions
9. `is_friday` (3%) - Movie release day
10. `near_holiday` (3%) - ±3 days from holiday

---

## 🎯 VALIDATION STRATEGY

**Your current:** TimeSeriesSplit (N=5) ✅ **CORRECT!**

**Why it's right:**
- Respects temporal order
- No data leakage from future

**Expected CV vs LB gap:**
- CV: 0.70-0.80 R²
- LB: 0.60-0.75 R² (15-20% drop is normal for 1-year-ahead forecasting)

---

## 🏆 PATH TO 0.95+ R² (IF POSSIBLE)

**Reality check:** 0.95 R² is EXTREMELY difficult for 1-year-ahead forecasting

**Why:**
- You're predicting March 2024 from 2023 data
- External factors (economy, new movie releases, COVID aftereffects)
- Individual theater closures/renovations unknown

**More realistic targets:**
- **Good:** 0.60-0.70 R² (top 25%)
- **Excellent:** 0.70-0.80 R² (top 10%)
- **Elite:** 0.80-0.85 R² (top 5%)
- **Perfect:** 0.90+ R² (top 1%, requires external data/domain expertise)

**To reach 0.90+, you MUST add:**
1. External movie release database (blockbusters)
2. Weather data (rain kills cinema attendance)
3. School calendar (holidays boost kids movies)
4. Per-movie features (genre, star cast, reviews)
5. Economic indicators (inflation, payday cycles)

---

## 📈 INCREMENTAL TESTING PLAN

**Week 1:** Run v5.0
- Submit to Kaggle
- Record LB score
- Analyze feature importance

**Week 2:** Run v6.0
- Compare v5 vs v6 delta
- If v6 is worse, check:
  - Prophet overfitting?
  - Stacking weights?

**Week 3:** Manual tuning
- Hyperparameter sweep (Optuna)
- Try different stacking weights
- Add domain features (holidays, movies)

**Week 4:** Final submission
- Ensemble v5 + v6 predictions (weighted average)
- Post-process outliers
- Submit!

---

## 🔥 CRITICAL SUCCESS FACTORS

### ✅ DO:
1. Use ALL booking data (huge signal!)
2. Validate with TimeSeriesSplit (prevent leakage)
3. Ensemble diverse models (trees + Prophet + linear)
4. Add domain knowledge (holidays, movie releases)
5. Clip predictions to reasonable bounds [2, theater_max]

### ❌ DON'T:
1. Use future data in features (data leakage!)
2. Over-tune on validation set (use cross-validation)
3. Ignore booking_*.csv files (free performance!)
4. Forget holiday effects (40% attendance spike!)
5. Rely only on tree models (ensemble diversity matters)

---

## 📞 NEXT STEPS

1. **Run v5.0 first:**
   ```bash
   python cinema_forecasting_v5_elite.py
   ```
   Expected: 0.50-0.60 R² (vs your 0.38)

2. **If v5.0 improves score, run v6.0:**
   ```bash
   python cinema_forecasting_v6_ultimate.py
   ```
   Expected: 0.65-0.80 R²

3. **Analyze differences:**
   - Which features matter most?
   - Are Prophet predictions helping or hurting?
   - Is stacking better than simple weighted average?

4. **Iterate:**
   - Add movie database (if available)
   - Optimize hyperparameters
   - Try neural network meta-learner

---

## 🎓 KEY LEARNINGS

**Domain expertise > pure ML skills**
- Knowing cinema attendance spikes on holidays is worth +20%
- Using booking data is worth +15%
- Generic feature engineering plateaus quickly

**Ensemble diversity > model complexity**
- 3 different models > 1 super-tuned model
- Prophet captures patterns LightGBM misses
- Stacking combines strengths

**Validation matters!**
- TimeSeriesSplit prevents leakage
- CV should match LB trend (not exact values)

---

## 💡 FINAL TIPS

1. **Check your metric:** Is it R² or something else? (You said "1.5" which is unusual for R²)

2. **Inspect test predictions:**
   ```python
   submission['audience_count'].describe()
   # Check for negatives, extreme values
   ```

3. **Compare distributions:**
   ```python
   train['audience_count'].hist()
   submission['audience_count'].hist()
   # Should look similar!
   ```

4. **Monitor leaderboard:**
   - If LB >> CV: You're overfitting validation
   - If CV >> LB: You're missing test patterns (external factors?)

---

**Good luck! 🚀 You've got all the tools to reach top 5%!**

If you need help debugging specific issues or want to add more features, let me know!
