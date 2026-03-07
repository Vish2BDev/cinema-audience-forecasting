"""
Prophet Model Implementation
Per-theater or clustered Prophet models for time-series forecasting
"""

from prophet import Prophet
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ProphetModel:
    """Prophet model wrapper for per-theater forecasting"""
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.models = {}  # Store models per theater
        self.holidays_df = None
        
    def create_holidays(self):
        """Create Indian holidays dataframe for Prophet"""
        holidays = [
            {"holiday": "Republic Day", "ds": "2023-01-26", "lower_window": -1, "upper_window": 1},
            {"holiday": "Holi", "ds": "2023-03-08", "lower_window": -2, "upper_window": 2},
            {"holiday": "Ambedkar Jayanti", "ds": "2023-04-14", "lower_window": 0, "upper_window": 0},
            {"holiday": "Independence Day", "ds": "2023-08-15", "lower_window": -1, "upper_window": 1},
            {"holiday": "Dussehra", "ds": "2023-10-24", "lower_window": -2, "upper_window": 2},
            {"holiday": "Diwali", "ds": "2023-11-12", "lower_window": -3, "upper_window": 3},
            {"holiday": "Christmas", "ds": "2023-12-25", "lower_window": -1, "upper_window": 1},
            {"holiday": "New Year", "ds": "2024-01-01", "lower_window": -1, "upper_window": 1},
            {"holiday": "Republic Day", "ds": "2024-01-26", "lower_window": -1, "upper_window": 1},
            {"holiday": "Holi", "ds": "2024-03-25", "lower_window": -2, "upper_window": 2},
            {"holiday": "Ambedkar Jayanti", "ds": "2024-04-14", "lower_window": 0, "upper_window": 0},
            {"holiday": "Independence Day", "ds": "2024-08-15", "lower_window": -1, "upper_window": 1},
            {"holiday": "Dussehra", "ds": "2024-10-12", "lower_window": -2, "upper_window": 2},
            {"holiday": "Diwali", "ds": "2024-11-01", "lower_window": -3, "upper_window": 3},
            {"holiday": "Christmas", "ds": "2024-12-25", "lower_window": -1, "upper_window": 1},
        ]
        
        self.holidays_df = pd.DataFrame(holidays)
        self.holidays_df["ds"] = pd.to_datetime(self.holidays_df["ds"])
        return self.holidays_df
    
    def train_per_theater(self, df, min_records=50):
        """Train Prophet model for each theater with sufficient data"""
        print("Training Prophet models per theater...")
        
        if self.holidays_df is None:
            self.create_holidays()
        
        theaters = df["book_theater_id"].unique()
        trained_count = 0
        
        for theater_id in theaters:
            theater_data = df[df["book_theater_id"] == theater_id].copy()
            theater_data = theater_data[theater_data["audience_count"].notna()]
            
            if len(theater_data) < min_records:
                continue
            
            # Prepare data for Prophet
            prophet_df = theater_data[["show_date", "audience_count"]].rename(
                columns={"show_date": "ds", "audience_count": "y"}
            )
            prophet_df = prophet_df.sort_values("ds")
            
            try:
                model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=False,
                    holidays=self.holidays_df,
                    seasonality_mode='multiplicative'
                )
                model.fit(prophet_df)
                self.models[theater_id] = model
                trained_count += 1
            except Exception as e:
                print(f"  Warning: Failed to train Prophet for {theater_id}: {e}")
                continue
        
        print(f"  Trained Prophet models for {trained_count}/{len(theaters)} theaters")
        return self.models
    
    def predict(self, df):
        """Predict using trained Prophet models"""
        print("Generating Prophet predictions...")
        
        predictions = np.full(len(df), np.nan)
        
        for theater_id in df["book_theater_id"].unique():
            if theater_id not in self.models:
                continue
            
            theater_mask = df["book_theater_id"] == theater_id
            theater_data = df[theater_mask].copy()
            
            # Prepare future dataframe
            future_df = theater_data[["show_date"]].rename(columns={"show_date": "ds"})
            future_df = future_df.sort_values("ds")
            
            try:
                model = self.models[theater_id]
                forecast = model.predict(future_df)
                
                # Map predictions back
                theater_indices = df[theater_mask].index
                predictions[theater_indices] = forecast["yhat"].values
            except Exception as e:
                print(f"  Warning: Failed to predict for {theater_id}: {e}")
                continue
        
        # Fill missing predictions with median of available predictions
        if np.isnan(predictions).any():
            median_pred = np.nanmedian(predictions)
            predictions = np.where(np.isnan(predictions), median_pred, predictions)
        
        return predictions
    
    def get_oof_predictions(self, train_df, val_df):
        """Get out-of-fold predictions for training and validation"""
        # Train on training data
        self.train_per_theater(train_df)
        
        # Predict on validation
        val_predictions = self.predict(val_df)
        
        # For training, use historical average as proxy (Prophet doesn't do OOF well)
        train_predictions = train_df.groupby("book_theater_id")["audience_count"].transform("mean")
        
        return train_predictions.values, val_predictions

