"""
Advanced Feature Engineering Module
Implements all temporal, booking, geographic, target encoding, and interaction features
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
import warnings
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Handles all feature engineering tasks"""
    
    def __init__(self, lag_days=[1, 2, 3, 7, 14, 21, 28, 30, 60, 90], 
                 rolling_windows=[7, 14, 30, 60, 90]):
        self.lag_days = lag_days
        self.rolling_windows = rolling_windows
        
    def create_calendar_features(self, df):
        """Create calendar-based features"""
        print("Creating calendar features...")
        
        df = df.copy()
        df["show_date"] = pd.to_datetime(df["show_date"])
        
        # Basic calendar features
        df["day_of_week"] = df["show_date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df["show_date"].dt.month
        df["week_of_year"] = df["show_date"].dt.isocalendar().week.astype(int)
        df["day_of_month"] = df["show_date"].dt.day
        df["quarter"] = df["show_date"].dt.quarter
        df["day_of_year"] = df["show_date"].dt.dayofyear
        
        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Cyclical encoding for week of year
        df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
        
        # Business cycle features
        df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
        df["is_quarter_start"] = df["day_of_month"].isin([1, 2, 3]).astype(int)
        df["is_quarter_end"] = (df["day_of_month"] >= 28).astype(int)
        
        return df
    
    def create_holiday_features(self, df):
        """Create Indian holiday features"""
        print("Creating holiday features...")
        
        df = df.copy()
        df["show_date"] = pd.to_datetime(df["show_date"])
        
        # Indian holidays 2023-2024
        holidays_2023 = [
            "2023-01-26",  # Republic Day
            "2023-03-08",  # Holi
            "2023-04-14",  # Ambedkar Jayanti
            "2023-08-15",  # Independence Day
            "2023-10-24",  # Dussehra
            "2023-11-12",  # Diwali
            "2023-12-25",  # Christmas
        ]
        
        holidays_2024 = [
            "2024-01-01",  # New Year
            "2024-01-26",  # Republic Day
            "2024-03-25",  # Holi
            "2024-04-14",  # Ambedkar Jayanti
            "2024-08-15",  # Independence Day
            "2024-10-12",  # Dussehra
            "2024-11-01",  # Diwali
            "2024-12-25",  # Christmas
        ]
        
        all_holidays = [pd.to_datetime(d) for d in holidays_2023 + holidays_2024]
        
        df["is_holiday"] = df["show_date"].isin(all_holidays).astype(int)
        
        # Days to/from nearest holiday
        df["days_to_holiday"] = df["show_date"].apply(
            lambda x: min([(h - x).days for h in all_holidays if h >= x], default=999)
        )
        df["days_since_holiday"] = df["show_date"].apply(
            lambda x: min([(x - h).days for h in all_holidays if h <= x], default=999)
        )
        
        # Holiday week flags (within 3 days of holiday)
        df["is_holiday_week"] = (df["days_to_holiday"] <= 3) | (df["days_since_holiday"] <= 3)
        df["is_holiday_week"] = df["is_holiday_week"].astype(int)
        
        return df
    
    def create_geographic_features(self, df):
        """Create geographic clustering and distance features"""
        print("Creating geographic features...")
        
        df = df.copy()
        
        # Calculate center coordinates from training data
        train_mask = df['audience_count'].notna()
        train_lat_mean = df[train_mask]['latitude'].mean()
        train_lon_mean = df[train_mask]['longitude'].mean()
        
        # Distance from center
        center_coords = np.radians([train_lat_mean, train_lon_mean])
        df_coords_rad = np.radians(df[['latitude', 'longitude']])
        df['dist_from_center'] = (
            haversine_distances(df_coords_rad, center_coords.reshape(1, -1)) * 6371
        ).flatten()
        
        # Geographic clustering
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_coords = df[train_mask][['latitude', 'longitude']]
        if len(train_coords) > 0:
            kmeans.fit(train_coords)
            df['geo_cluster'] = kmeans.predict(df[['latitude', 'longitude']])
        else:
            df['geo_cluster'] = 0
        
        return df
    
    def create_lag_features(self, df):
        """Create lag features for audience_count"""
        print("Creating lag features...")
        
        df = df.copy()
        df = df.sort_values(["book_theater_id", "show_date"])
        
        # Create lags
        for lag in self.lag_days:
            df[f"aud_lag_{lag}"] = df.groupby("book_theater_id")["audience_count"].shift(lag)
        
        # Same day-of-week lags (last Sunday, last Monday, etc.)
        df["same_dow_lag_7"] = df.groupby(["book_theater_id", "day_of_week"])["audience_count"].shift(1)
        df["same_dow_lag_14"] = df.groupby(["book_theater_id", "day_of_week"])["audience_count"].shift(2)
        df["same_dow_lag_21"] = df.groupby(["book_theater_id", "day_of_week"])["audience_count"].shift(3)
        
        # Lag ratios (trend indicators)
        df["lag_ratio_7_14"] = df["aud_lag_7"] / (df["aud_lag_14"] + 1e-6)
        df["lag_ratio_30_60"] = df["aud_lag_30"] / (df["aud_lag_60"] + 1e-6)
        df["lag_ratio_14_28"] = df["aud_lag_14"] / (df["aud_lag_28"] + 1e-6)
        
        # Lag differences (acceleration/deceleration)
        df["lag_diff_7_14"] = df["aud_lag_7"] - df["aud_lag_14"]
        df["lag_diff_14_28"] = df["aud_lag_14"] - df["aud_lag_28"]
        df["lag_diff_30_60"] = df["aud_lag_30"] - df["aud_lag_60"]
        
        return df
    
    def create_rolling_features(self, df):
        """Create rolling statistics features"""
        print("Creating rolling features...")
        
        df = df.copy()
        df = df.sort_values(["book_theater_id", "show_date"])
        
        # Get global average for filling
        global_avg = df[df['audience_count'].notna()]['audience_count'].mean()
        
        for w in self.rolling_windows:
            # Shift by 1 to avoid leakage
            rolling_series = df.groupby("book_theater_id")["audience_count"].shift(1).rolling(w, min_periods=1)
            
            df[f"roll_mean_{w}"] = rolling_series.mean().reset_index(0, drop=True)
            df[f"roll_std_{w}"] = rolling_series.std().reset_index(0, drop=True)
            df[f"roll_min_{w}"] = rolling_series.min().reset_index(0, drop=True)
            df[f"roll_max_{w}"] = rolling_series.max().reset_index(0, drop=True)
            df[f"roll_median_{w}"] = rolling_series.median().reset_index(0, drop=True)
        
        # Rolling trends
        df["trend_7"] = df["roll_mean_7"] / (df["roll_mean_14"] + 1e-6)
        df["trend_14"] = df["roll_mean_14"] / (df["roll_mean_30"] + 1e-6)
        df["rolling_diff"] = df["roll_mean_7"] - df["roll_mean_14"]
        df["rolling_diff_14_30"] = df["roll_mean_14"] - df["roll_mean_30"]
        
        return df
    
    def create_missing_date_features(self, df):
        """Create features for handling missing dates"""
        print("Creating missing date features...")
        
        df = df.copy()
        df = df.sort_values(["book_theater_id", "show_date"])
        
        # Days since last show
        df["days_since_last_show"] = (
            df.groupby("book_theater_id")["show_date"]
            .diff().dt.days.fillna(999)
        )
        
        # Days to next show
        df["days_to_next_show"] = (
            df.groupby("book_theater_id")["show_date"]
            .diff(-1).dt.days.fillna(999) * -1
        )
        
        # Operating day detection (does theater operate on this day-of-week?)
        train_df = df[df["audience_count"].notna()].copy()
        if len(train_df) > 0:
            theater_dow_pattern = (
                train_df.groupby(["book_theater_id", "day_of_week"])["audience_count"]
                .count() > 0
            ).reset_index()
            theater_dow_pattern.columns = ["book_theater_id", "day_of_week", "is_operating_day"]
            df = df.merge(theater_dow_pattern, on=["book_theater_id", "day_of_week"], how="left")
            df["is_operating_day"] = df["is_operating_day"].fillna(0).astype(int)
        else:
            df["is_operating_day"] = 1
        
        # Theater operating frequency (days per week)
        theater_freq = (
            train_df.groupby("book_theater_id")["day_of_week"]
            .nunique() if len(train_df) > 0 else pd.Series()
        )
        if len(theater_freq) > 0:
            df = df.merge(
                theater_freq.to_frame("theater_operating_frequency"),
                left_on="book_theater_id", right_index=True, how="left"
            )
            df["theater_operating_frequency"] = df["theater_operating_frequency"].fillna(7)
        else:
            df["theater_operating_frequency"] = 7
        
        # Consecutive missing days
        df["has_audience"] = df["audience_count"].notna().astype(int)
        # Calculate consecutive missing days per theater
        df["consecutive_missing"] = 0
        for theater_id in df["book_theater_id"].unique():
            theater_mask = df["book_theater_id"] == theater_id
            theater_has_audience = df.loc[theater_mask, "has_audience"].values
            missing = (~theater_has_audience.astype(bool))
            if missing.any():
                consecutive = missing.astype(int)
                for i in range(1, len(consecutive)):
                    if missing[i]:
                        consecutive[i] = consecutive[i-1] + 1
                df.loc[theater_mask, "consecutive_missing"] = consecutive
        
        return df
    
    def create_statistical_features(self, df):
        """Create theater-level statistical features"""
        print("Creating statistical features...")
        
        df = df.copy()
        train_df = df[df["audience_count"].notna()].copy()
        
        if len(train_df) > 0:
            # Theater-level statistics
            theater_stats = train_df.groupby("book_theater_id")["audience_count"].agg([
                'mean', 'median', 'std', 'min', 'max',
                lambda x: x.quantile(0.25),  # q25
                lambda x: x.quantile(0.75),  # q75
            ]).reset_index()
            theater_stats.columns = [
                "book_theater_id", "theater_mean", "theater_median", "theater_std",
                "theater_min", "theater_max", "theater_q25", "theater_q75"
            ]
            
            # Coefficient of variation
            theater_stats["theater_cv"] = theater_stats["theater_std"] / (theater_stats["theater_mean"] + 1e-6)
            
            # Merge back
            df = df.merge(theater_stats, on="book_theater_id", how="left")
            
            # Fill missing with global stats
            global_mean = train_df["audience_count"].mean()
            global_median = train_df["audience_count"].median()
            global_std = train_df["audience_count"].std()
            
            df["theater_mean"] = df["theater_mean"].fillna(global_mean)
            df["theater_median"] = df["theater_median"].fillna(global_median)
            df["theater_std"] = df["theater_std"].fillna(global_std)
            df["theater_min"] = df["theater_min"].fillna(0)
            df["theater_max"] = df["theater_max"].fillna(global_mean * 2)
            df["theater_q25"] = df["theater_q25"].fillna(global_median * 0.75)
            df["theater_q75"] = df["theater_q75"].fillna(global_median * 1.25)
            df["theater_cv"] = df["theater_cv"].fillna(0.5)
        else:
            # Default values if no training data
            global_mean = 40
            df["theater_mean"] = global_mean
            df["theater_median"] = global_mean
            df["theater_std"] = global_mean * 0.5
            df["theater_min"] = 0
            df["theater_max"] = global_mean * 2
            df["theater_q25"] = global_mean * 0.75
            df["theater_q75"] = global_mean * 1.25
            df["theater_cv"] = 0.5
        
        return df
    
    def create_target_encoding(self, df):
        """Create target encoding features with time-based CV"""
        print("Creating target encoding features...")
        
        df = df.copy()
        df = df.sort_values(["book_theater_id", "show_date"])
        
        # Initialize encoding columns
        df["te_theater_dow"] = np.nan
        df["te_theater_month"] = np.nan
        df["te_area_dow"] = np.nan
        df["te_area_month"] = np.nan
        
        # Get training mask
        train_mask = df["audience_count"].notna()
        
        # Simplified target encoding using groupby means (much faster)
        print("  - Theater-day-of-week encoding...")
        train_df = df[train_mask].copy()
        theater_dow_means = (
            train_df.groupby(["book_theater_id", "day_of_week"])["audience_count"].mean()
            .reset_index()
        )
        theater_dow_means.columns = ["book_theater_id", "day_of_week", "te_theater_dow"]
        df = df.merge(theater_dow_means, on=["book_theater_id", "day_of_week"], how="left", suffixes=("", "_new"))
        if "te_theater_dow_new" in df.columns:
            df["te_theater_dow"] = df["te_theater_dow_new"].fillna(df["te_theater_dow"])
            df = df.drop(columns=["te_theater_dow_new"])
        
        # Theater-month encoding
        print("  - Theater-month encoding...")
        theater_month_means = (
            train_df.groupby(["book_theater_id", "month"])["audience_count"].mean()
            .reset_index()
        )
        theater_month_means.columns = ["book_theater_id", "month", "te_theater_month"]
        df = df.merge(theater_month_means, on=["book_theater_id", "month"], how="left", suffixes=("", "_new"))
        if "te_theater_month_new" in df.columns:
            df["te_theater_month"] = df["te_theater_month_new"].fillna(df["te_theater_month"])
            df = df.drop(columns=["te_theater_month_new"])
        
        # Area-day-of-week encoding
        print("  - Area-day-of-week encoding...")
        area_dow_means = (
            train_df.groupby(["theater_area", "day_of_week"])["audience_count"].mean()
            .reset_index()
        )
        area_dow_means.columns = ["theater_area", "day_of_week", "te_area_dow"]
        df = df.merge(area_dow_means, on=["theater_area", "day_of_week"], how="left", suffixes=("", "_new"))
        if "te_area_dow_new" in df.columns:
            df["te_area_dow"] = df["te_area_dow_new"].fillna(df["te_area_dow"])
            df = df.drop(columns=["te_area_dow_new"])
        
        # Area-month encoding
        print("  - Area-month encoding...")
        area_month_means = (
            train_df.groupby(["theater_area", "month"])["audience_count"].mean()
            .reset_index()
        )
        area_month_means.columns = ["theater_area", "month", "te_area_month"]
        df = df.merge(area_month_means, on=["theater_area", "month"], how="left", suffixes=("", "_new"))
        if "te_area_month_new" in df.columns:
            df["te_area_month"] = df["te_area_month_new"].fillna(df["te_area_month"])
            df = df.drop(columns=["te_area_month_new"])
        
        # Fill missing with global mean
        global_mean = df[train_mask]["audience_count"].mean()
        for col in ["te_theater_dow", "te_theater_month", "te_area_dow", "te_area_month"]:
            if col in df.columns:
                df[col] = df[col].fillna(global_mean)
            else:
                df[col] = global_mean
        
        return df
    
    def create_booking_features(self, df, bookings_bn, bookings_pos):
        """Create advanced booking features"""
        print("Creating booking features...")
        
        df = df.copy()
        
        # Normalize booking dates
        bookings_bn = bookings_bn.copy()
        bookings_pos = bookings_pos.copy()
        
        if "show_datetime" in bookings_bn.columns:
            bookings_bn["show_date"] = pd.to_datetime(bookings_bn["show_datetime"]).dt.normalize()
            bookings_bn["booking_datetime"] = pd.to_datetime(bookings_bn["booking_datetime"])
            bookings_bn["advance_hours"] = (
                pd.to_datetime(bookings_bn["show_datetime"]) - bookings_bn["booking_datetime"]
            ).dt.total_seconds() / 3600
        
        if "show_datetime" in bookings_pos.columns:
            bookings_pos["show_date"] = pd.to_datetime(bookings_pos["show_datetime"]).dt.normalize()
        
        # Aggregate booking features per theater-date
        if len(bookings_bn) > 0 and "advance_hours" in bookings_bn.columns:
            booking_features = bookings_bn.groupby(["book_theater_id", "show_date"]).agg({
                "tickets_booked": ["sum", "mean", "count"],
                "advance_hours": ["mean", "std", "min", "max"]
            }).reset_index()
            booking_features.columns = [
                "book_theater_id", "show_date",
                "booking_sum", "booking_mean", "booking_count",
                "advance_hours_mean", "advance_hours_std", "advance_hours_min", "advance_hours_max"
            ]
            
            df = df.merge(booking_features, on=["book_theater_id", "show_date"], how="left")
            
            # Fill missing
            for col in booking_features.columns[2:]:
                df[col] = df[col].fillna(0)
        else:
            # Create dummy columns if no booking data
            for col in ["booking_sum", "booking_mean", "booking_count",
                        "advance_hours_mean", "advance_hours_std", "advance_hours_min", "advance_hours_max"]:
                df[col] = 0
        
        # Booking velocity (rolling mean)
        df = df.sort_values(["book_theater_id", "show_date"])
        for w in [7, 14, 30]:
            df[f"booking_velocity_{w}"] = (
                df.groupby("book_theater_id")["total_tickets"]
                .shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"booking_velocity_{w}"] = df[f"booking_velocity_{w}"].fillna(0)
        
        # Last 7 days booking sum
        df["last_7_days_booking"] = (
            df.groupby("book_theater_id")["total_tickets"]
            .shift(1).rolling(7, min_periods=1).sum()
        ).fillna(0)
        
        # Booking trend
        df["booking_trend"] = df["booking_velocity_7"] / (df["booking_velocity_14"] + 1e-6)
        df["booking_trend"] = df["booking_trend"].fillna(1.0)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        print("Creating interaction features...")
        
        df = df.copy()
        
        # Theater × Day-of-week (encoded as string for categorical)
        df["theater_dow_interaction"] = (
            df["book_theater_id"].astype(str) + "_" + df["day_of_week"].astype(str)
        )
        
        # Area × Month
        df["area_month_interaction"] = (
            df["theater_area"].astype(str) + "_" + df["month"].astype(str)
        )
        
        # Numerical interactions
        df["lag_7_weekend"] = df["aud_lag_7"] * df["is_weekend"]
        df["roll_mean_7_type"] = df["roll_mean_7"] * (df["theater_type"] == "Drama").astype(int)
        df["total_tickets_lag_7"] = df["total_tickets"] * df["aud_lag_7"]
        df["lag_7_holiday"] = df["aud_lag_7"] * df["is_holiday"]
        
        return df
    
    def fill_missing_lag_features(self, df):
        """Fill missing lag and rolling features"""
        print("Filling missing lag features...")
        
        df = df.copy()
        
        # Get theater averages for filling
        train_mask = df["audience_count"].notna()
        if train_mask.sum() > 0:
            theater_avg = df[train_mask].groupby("book_theater_id")["audience_count"].mean()
            global_avg = df[train_mask]["audience_count"].mean()
        else:
            theater_avg = pd.Series()
            global_avg = 40
        
        # Fill lag features
        lag_cols = [c for c in df.columns if "lag" in c or "roll" in c or "same_dow" in c]
        for col in lag_cols:
            if col in df.columns:
                # Fill with theater average first
                if len(theater_avg) > 0:
                    df[col] = df[col].fillna(
                        df["book_theater_id"].map(theater_avg)
                    )
                # Then fill with global average
                df[col] = df[col].fillna(global_avg)
        
        return df
    
    def process(self, df, bookings_bn=None, bookings_pos=None):
        """Main feature engineering pipeline"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Calendar features
        df = self.create_calendar_features(df)
        
        # Holiday features
        df = self.create_holiday_features(df)
        
        # Geographic features
        df = self.create_geographic_features(df)
        
        # Lag features
        df = self.create_lag_features(df)
        
        # Rolling features
        df = self.create_rolling_features(df)
        
        # Missing date features
        df = self.create_missing_date_features(df)
        
        # Statistical features
        df = self.create_statistical_features(df)
        
        # Target encoding
        df = self.create_target_encoding(df)
        
        # Booking features
        if bookings_bn is not None and bookings_pos is not None:
            df = self.create_booking_features(df, bookings_bn, bookings_pos)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Fill missing lag features
        df = self.fill_missing_lag_features(df)
        
        print("\n[OK] Feature engineering complete!")
        return df


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.process()
    
    engineer = FeatureEngineer()
    full_df = engineer.process(
        data_dict['full'],
        data_dict['bookings_bn'],
        data_dict['bookings_pos']
    )
    
    print(f"\nFinal dataframe shape: {full_df.shape}")
    print(f"Total features: {len(full_df.columns)}")

