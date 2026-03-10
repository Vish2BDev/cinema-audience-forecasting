# Cinema Audience Forecasting

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-22D3EE?style=flat-square)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-F97316?style=flat-square)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-6366F1?style=flat-square)](https://catboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

Predicting daily cinema attendance across 337 Indian theaters — 174,535 training rows, 109 engineered features, and a stacking ensemble of five models that pushed validation R² from **0.354 to 0.79**.

🔗 **[Live Project Report](https://portfolio-1y8m79lua-vishal-bhandaris-projects.vercel.app)**

---

## The Problem

A mid-sized Indian multiplex chain needs to forecast daily audience counts per screen, 53 days into the future. Operations teams use these forecasts to schedule staff, order concessions, and allocate screens — decisions that must be locked in 72 hours before showtime.

The data comes from two disconnected POS systems (BookNow and CinePOS), each using its own theater ID namespace. Bridging them is step zero. From there, the forecasting challenge is genuinely hard: 337 individual time series, a strong weekly seasonal pattern, Bollywood release-day spikes, and an Indian holiday calendar that breaks most generic models.

The key insight driving most of the R² gain: **advance booking velocity is a leading indicator**. When a blockbuster opens pre-sales 10+ days out, that signal is available before the date arrives — standard lag features cannot capture it.

---

## Results

| Model | Validation R² |
|---|---|
| LightGBM (standalone) | 0.71 |
| XGBoost (standalone) | 0.68 |
| CatBoost (standalone) | 0.72 |
| Random Forest (standalone) | 0.58 |
| Prophet per-theater (standalone) | 0.55 |
| **Stacking Ensemble (Ridge meta-learner)** | **0.79** |

Baseline (median-per-theater) started at **0.354**. The final ensemble is a **+124% improvement**.

The biggest single jump came from per-theater, per-day-of-week **target encoding** (+18 R² points). Indian holiday proximity features were the second-largest contributor — Diwali weekend attendance runs 40–60% above the weekly baseline for most screens.

---

## How It Works

### Data

Eight CSV files spanning two POS systems:

| File | Description | Size |
|---|---|---|
| `booknow_visits.csv` | Daily audience counts per theater — the prediction target | 174,535 rows |
| `booknow_booking.csv` | Advance booking transactions with timestamps | Transaction-level |
| `cinePOS_booking.csv` | In-theater POS transactions (separate ID namespace) | Transaction-level |
| `booknow_theaters.csv` | Theater metadata: lat/lon, type, area | 337 theaters |
| `cinePOS_theaters.csv` | CinePOS theater metadata | Subset |
| `movie_theater_id_relation.csv` | ID bridge: cinePOS → BookNow | Many-to-one map |
| `date_info.csv` | Calendar flags for the full date range | ~480 dates |
| `sample_submission.csv` | Test set IDs defining the 53-day forecast horizon | 38,062 rows |

The merge pipeline starts with `movie_theater_id_relation.csv` to unify both ID namespaces into a single theater-date view. All subsequent aggregation and feature engineering operates at the `(theater_id, show_date)` grain.

### Feature Engineering

109 features across 8 categories. The three that matter most:

**1. Same-day-of-week lags**
Predicting Saturday attendance from last Saturday (lag-7 aligned to the same day of week) consistently outperforms a raw lag-7 by a meaningful margin. Momentum ratios like `lag_7 / lag_14` add directional context. Windows extend to 90 days to cover monthly and quarterly cycles.

**2. Target encoding (time-safe)**
`te_theater_dow` encodes each theater's historical mean attendance on each day of the week, computed with an expanding window so no future data bleeds into training features. This single feature class contributed more R² than any other category.

**3. Advance booking signals**
`avg_lead_hours`, `booking_velocity_7`, and `num_bookings` are aggregated from raw BookNow transaction data. High advance lead-time signals an anticipated blockbuster before opening day — giving the model forward-looking information that pure lag features lack.

Other categories: rolling statistics (5 windows × 5 stats = 25 features), Indian holiday calendar (35 holidays, 2023–2024, with ±3 day proximity windows), geographic clustering (KMeans k=10 on lat/lon coordinates), cyclical time encoding (sin/cos for DOW and month), and theater-level statistical profiles.

### Modeling

Five base models, each trained independently with their own hyperparameters and tuned for their respective inductive biases:

- **LightGBM** — leaf-wise growth, 2,000 estimators, 127 leaves. Fast and aggressive on non-linearities.
- **XGBoost** — level-wise growth with L1/L2 regularization. Smoother predictions than LightGBM; provides useful variance in the ensemble.
- **CatBoost** — handles `theater_area`, `theater_type`, and `geo_cluster` via ordered target statistics, avoiding the target leakage endemic to naive one-hot encoding of high-cardinality categoricals.
- **Random Forest** — bagged averaging acts as a conservative anchor, pulling the ensemble away from extreme predictions at distribution tails.
- **Prophet** — individual per-theater models trained on the top-N theaters by data volume. Captures theater-specific weekly and annual seasonality that global models smooth over; out-of-fold predictions feed into the meta-learner as features.

### Ensemble

A Ridge meta-learner is trained on out-of-fold predictions from all five base models. The learned stacking weights: LightGBM 0.31 · CatBoost 0.28 · XGBoost 0.24 · Prophet 0.11 · RF 0.06.

Post-processing applies three deterministic constraints: floor predictions at zero (no negative attendance), cap at each theater's observed historical maximum, and apply 3-day rolling median smoothing to enforce day-of-week coherence in the final submission.

---

## Project Structure

```
cinema-audience-forecasting/
├── src/
│   ├── data_preprocessing.py      # Data loading, ID unification, train/val split
│   ├── feature_engineering.py     # All 109 features across 8 categories
│   ├── ensemble.py                # Stacking and weighted blending
│   ├── post_processing.py         # Floor, cap, and smoothing constraints
│   ├── evaluation.py              # Metrics and diagnostics
│   └── models/
│       ├── lightgbm_model.py
│       ├── xgboost_model.py
│       ├── catboost_model.py
│       ├── random_forest_model.py
│       └── prophet_model.py
├── config/
│   └── config.yaml                # All hyperparameters and pipeline settings
├── main.py                        # End-to-end pipeline entry point
├── submit.py                      # Formats final predictions to submission.csv
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

Place all eight data CSVs in the project root, then:

```bash
python main.py
```

The pipeline preprocesses, engineers features, trains all models sequentially, runs stacking, applies post-processing, and writes `submission.csv`. Full training takes roughly 15–25 minutes on a modern laptop. Prophet is the bottleneck — it can be disabled in `config.yaml` for faster iteration.

---

## Configuration

All hyperparameters, feature settings, and validation options live in `config/config.yaml`:

```yaml
prophet:
  enabled: true          # Set false to skip Prophet (~10x faster training)
  top_n_theaters: 50     # How many theaters get individual Prophet models

validation:
  split_date: "2024-01-01"   # Start of held-out validation window
  end_date:   "2024-02-28"   # Matches the test period (Mar–Apr 2024)

ensemble:
  method: stacking       # Options: "stacking" | "weighted_blend"
```

---

## Technical Notes

A few things worth knowing before you run this:

- **Target encoding leakage is subtle.** The expanding-window implementation is intentional — using a simple historical mean on the full training set inflates validation R² by roughly 0.04, which doesn't show up until test time.
- **Prophet is optional, not decorative.** Its 0.11 stacking weight looks small, but it covers specific theaters with strong yearly seasonality that gradient boosting models average away.
- **The CinePOS ID bridge is not optional.** About 18% of test theater IDs appear only in the CinePOS namespace. Without the unification step, those rows receive no booking features and predictions deteriorate significantly for those screens.
- **Day-of-week coherence matters operationally.** A forecast that ranks Friday > Thursday > Monday correctly is more useful for staffing decisions than one with better absolute RMSE but inverted day-of-week ordering. The post-processing smoothing step was added for this reason.

---

## License

[MIT](LICENSE)
