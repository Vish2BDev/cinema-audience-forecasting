"""
Post-Processing Module
Applies constraints, smoothing, and calibration to predictions
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class PostProcessor:
    """Post-processing for predictions"""
    
    def __init__(self, train_df=None):
        self.train_df = train_df
        self.theater_max = None
        self.theater_operating_patterns = None
        
        if train_df is not None:
            self._compute_theater_stats()
    
    def _compute_theater_stats(self):
        """Compute theater-level statistics from training data"""
        print("Computing theater statistics for post-processing...")
        
        # Theater maximums (capacity proxy)
        self.theater_max = (
            self.train_df.groupby("book_theater_id")["audience_count"].max()
        )
        
        # Operating day patterns
        self.theater_operating_patterns = (
            self.train_df.groupby(["book_theater_id", "day_of_week"])["audience_count"]
            .count() > 0
        ).reset_index()
        self.theater_operating_patterns.columns = [
            "book_theater_id", "day_of_week", "is_operating_day"
        ]
        
        print(f"  Computed stats for {len(self.theater_max)} theaters")
    
    def apply_constraints(self, predictions, test_df):
        """Apply constraints to predictions"""
        print("Applying post-processing constraints...")
        
        predictions = predictions.copy()
        
        # 1. Floor at 0
        predictions = np.maximum(predictions, 0)
        print(f"  Floored {np.sum(predictions < 0)} negative predictions to 0")
        
        # 2. Cap at theater historical maximum
        if self.theater_max is not None:
            test_df = test_df.copy()
            test_df["pred"] = predictions
            test_df["theater_max"] = test_df["book_theater_id"].map(self.theater_max)
            test_df["theater_max"] = test_df["theater_max"].fillna(predictions.max())
            
            capped_mask = test_df["pred"] > test_df["theater_max"]
            test_df.loc[capped_mask, "pred"] = test_df.loc[capped_mask, "theater_max"]
            predictions = test_df["pred"].values
            
            print(f"  Capped {capped_mask.sum()} predictions at theater maximum")
        
        # 3. Handle missing operating days
        if self.theater_operating_patterns is not None:
            test_df = test_df.copy()
            test_df["pred"] = predictions
            
            # Merge operating patterns
            test_df = test_df.merge(
                self.theater_operating_patterns,
                on=["book_theater_id", "day_of_week"],
                how="left"
            )
            test_df["is_operating_day"] = test_df["is_operating_day"].fillna(0).astype(int)
            
            # Set to 0 for non-operating days
            non_operating_mask = test_df["is_operating_day"] == 0
            test_df.loc[non_operating_mask, "pred"] = 0
            predictions = test_df["pred"].values
            
            print(f"  Set {non_operating_mask.sum()} non-operating day predictions to 0")
        
        return predictions
    
    def apply_smoothing(self, predictions, test_df, window=3):
        """Apply temporal smoothing"""
        print(f"Applying temporal smoothing (window={window})...")
        
        test_df = test_df.copy()
        test_df["pred"] = predictions
        test_df = test_df.sort_values(["book_theater_id", "show_date"])
        
        # Rolling median per theater
        test_df["pred_smooth"] = (
            test_df.groupby("book_theater_id")["pred"]
            .transform(lambda x: x.rolling(window, center=True, min_periods=1).median())
        )
        
        smoothed_count = np.sum(test_df["pred_smooth"] != test_df["pred"])
        print(f"  Smoothed {smoothed_count} predictions")
        
        return test_df["pred_smooth"].values
    
    def apply_day_of_week_consistency(self, predictions, test_df, train_df=None):
        """Ensure predictions respect day-of-week patterns"""
        if train_df is None:
            return predictions
        
        print("Applying day-of-week consistency...")
        
        # Compute day-of-week multipliers from training data
        dow_multipliers = (
            train_df.groupby("day_of_week")["audience_count"].mean() /
            train_df["audience_count"].mean()
        )
        
        test_df = test_df.copy()
        test_df["pred"] = predictions
        test_df["dow_multiplier"] = test_df["day_of_week"].map(dow_multipliers)
        test_df["dow_multiplier"] = test_df["dow_multiplier"].fillna(1.0)
        
        # Adjust predictions by day-of-week multiplier
        test_df["pred"] = test_df["pred"] * test_df["dow_multiplier"]
        
        return test_df["pred"].values
    
    def round_predictions(self, predictions):
        """Round predictions to integers if needed"""
        print("Rounding predictions to integers...")
        return np.round(predictions).astype(int)
    
    def process(self, predictions, test_df, apply_smoothing_flag=True, 
                apply_dow_consistency=True, round_predictions_flag=False):
        """Main post-processing pipeline"""
        print("\n" + "="*60)
        print("POST-PROCESSING PIPELINE")
        print("="*60)
        
        # Apply constraints
        predictions = self.apply_constraints(predictions, test_df)
        
        # Apply smoothing
        if apply_smoothing_flag:
            predictions = self.apply_smoothing(predictions, test_df)
        
        # Apply day-of-week consistency
        if apply_dow_consistency and self.train_df is not None:
            predictions = self.apply_day_of_week_consistency(
                predictions, test_df, self.train_df
            )
        
        # Round to integers
        if round_predictions_flag:
            predictions = self.round_predictions(predictions)
        
        print("\n[OK] Post-processing complete!")
        return predictions

