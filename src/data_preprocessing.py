"""
Data Preprocessing Module
Handles data loading, merging, train/val split, and missing date handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self, data_dir=".", val_start="2024-01-01", val_end="2024-02-28"):
        self.data_dir = Path(data_dir)
        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)
        
    def load_data(self):
        """Load all CSV files"""
        print("Loading data files...")
        
        data = {}
        data['visits'] = pd.read_csv(self.data_dir / "booknow_visits.csv")
        data['bookings_bn'] = pd.read_csv(self.data_dir / "booknow_booking.csv")
        data['bookings_pos'] = pd.read_csv(self.data_dir / "cinePOS_booking.csv")
        data['theaters_bn'] = pd.read_csv(self.data_dir / "booknow_theaters.csv")
        data['theaters_pos'] = pd.read_csv(self.data_dir / "cinePOS_theaters.csv")
        data['map_ids'] = pd.read_csv(self.data_dir / "movie_theater_id_relation.csv")
        data['date_info'] = pd.read_csv(self.data_dir / "date_info.csv")
        data['sample_sub'] = pd.read_csv(self.data_dir / "sample_submission.csv")
        
        print("[OK] All files loaded successfully!")
        return data
    
    def normalize_dates(self, df, date_cols):
        """Normalize date columns to remove time component"""
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.normalize()
        return df
    
    def create_unified_theaters(self, theaters_bn, theaters_pos, map_ids):
        """Create unified theater metadata"""
        print("Creating unified theater metadata...")
        
        # Standardize column names
        if 'lat' in theaters_pos.columns:
            theaters_pos = theaters_pos.rename(columns={
                "lat": "latitude",
                "lon": "longitude",
                "area": "theater_area",
                "type": "theater_type"
            })
        
        # Map cinePOS theaters to booknow IDs
        theaters_pos_mapped = theaters_pos.merge(
            map_ids, on="cine_theater_id", how="left"
        )
        
        # Combine both theater datasets
        cols_to_keep = ["book_theater_id", "latitude", "longitude", 
                       "theater_area", "theater_type"]
        cols_bn = [col for col in cols_to_keep if col in theaters_bn.columns]
        cols_pos = [col for col in cols_to_keep if col in theaters_pos_mapped.columns]
        
        all_theaters = pd.concat([
            theaters_bn[cols_bn],
            theaters_pos_mapped[cols_pos]
        ])
        
        # Remove duplicates, keep first occurrence
        all_theaters = all_theaters.drop_duplicates(
            subset=["book_theater_id"], keep="first"
        )
        
        print(f"Total unique theaters with metadata: {len(all_theaters)}")
        return all_theaters
    
    def aggregate_bookings(self, bookings_bn, bookings_pos, map_ids):
        """Aggregate booking data to daily level"""
        print("Aggregating booking data...")
        
        # Normalize dates
        bookings_bn = self.normalize_dates(bookings_bn, ["show_datetime", "booking_datetime"])
        bookings_pos = self.normalize_dates(bookings_pos, ["show_datetime", "booking_datetime"])
        
        # Create show_date column
        bookings_bn["show_date"] = bookings_bn["show_datetime"]
        bookings_pos["show_date"] = bookings_pos["show_datetime"]
        
        # Map cinePOS to booknow IDs
        bookings_pos = bookings_pos.merge(map_ids, on="cine_theater_id", how="left")
        
        # Aggregate to daily level
        bn_daily = bookings_bn.groupby(["book_theater_id", "show_date"])["tickets_booked"].sum().reset_index()
        pos_daily = bookings_pos.groupby(["book_theater_id", "show_date"])["tickets_sold"].sum().reset_index()
        
        print("[OK] Booking aggregations complete")
        return bn_daily, pos_daily, bookings_bn, bookings_pos
    
    def create_full_dataframe(self, visits, test_df, bn_daily, pos_daily, 
                            all_theaters, date_info):
        """Create full dataframe with train and test combined"""
        print("Creating full dataframe...")
        
        # Normalize dates
        visits = self.normalize_dates(visits, ["show_date"])
        test_df = self.normalize_dates(test_df, ["show_date"])
        date_info = self.normalize_dates(date_info, ["show_date"])
        
        # Remove ID column from visits if present
        if "ID" in visits.columns:
            visits = visits.drop(columns=["ID"])
        
        # Combine train and test
        test_df["audience_count"] = np.nan
        full_df = pd.concat([visits, test_df], ignore_index=True, sort=False)
        
        # Merge booking data
        full_df = full_df.merge(bn_daily, on=["book_theater_id", "show_date"], how="left")
        full_df = full_df.merge(pos_daily, on=["book_theater_id", "show_date"], how="left")
        
        # Merge theater metadata
        full_df = full_df.merge(all_theaters, on="book_theater_id", how="left")
        
        # Merge date info
        full_df = full_df.merge(date_info, on="show_date", how="left")
        
        # Fill missing values
        full_df[["tickets_booked", "tickets_sold"]] = full_df[["tickets_booked", "tickets_sold"]].fillna(0)
        full_df["total_tickets"] = full_df["tickets_booked"] + full_df["tickets_sold"]
        full_df["theater_area"] = full_df["theater_area"].fillna("Unknown")
        full_df["theater_type"] = full_df["theater_type"].fillna("Unknown")
        
        # Fill missing coordinates with train mean
        train_mask = full_df['audience_count'].notna()
        train_lat_mean = full_df[train_mask]['latitude'].mean()
        train_lon_mean = full_df[train_mask]['longitude'].mean()
        full_df["latitude"] = full_df["latitude"].fillna(train_lat_mean)
        full_df["longitude"] = full_df["longitude"].fillna(train_lon_mean)
        
        print("[OK] Full dataframe created")
        return full_df
    
    def create_train_val_split(self, full_df):
        """Create train/validation split based on dates"""
        print("Creating train/validation split...")
        
        # Separate train and test
        train_df = full_df[full_df["audience_count"].notna()].copy()
        test_df = full_df[full_df["audience_count"].isna()].copy()
        
        # Split train into train and validation
        train_df["show_date"] = pd.to_datetime(train_df["show_date"])
        train_mask = train_df["show_date"] < self.val_start
        val_mask = (train_df["show_date"] >= self.val_start) & (train_df["show_date"] <= self.val_end)
        
        train_split = train_df[train_mask].copy()
        val_split = train_df[val_mask].copy()
        
        print(f"Train set: {len(train_split):,} records ({train_split['show_date'].min()} to {train_split['show_date'].max()})")
        print(f"Validation set: {len(val_split):,} records ({val_split['show_date'].min()} to {val_split['show_date'].max()})")
        print(f"Test set: {len(test_df):,} records ({test_df['show_date'].min()} to {test_df['show_date'].max()})")
        
        return train_split, val_split, test_df
    
    def process(self):
        """Main processing pipeline"""
        # Load data
        data = self.load_data()
        
        # Normalize dates in visits
        data['visits'] = self.normalize_dates(data['visits'], ["show_date"])
        data['date_info'] = self.normalize_dates(data['date_info'], ["show_date"])
        
        # Create unified theaters
        all_theaters = self.create_unified_theaters(
            data['theaters_bn'], data['theaters_pos'], data['map_ids']
        )
        
        # Aggregate bookings
        bn_daily, pos_daily, bookings_bn, bookings_pos = self.aggregate_bookings(
            data['bookings_bn'], data['bookings_pos'], data['map_ids']
        )
        
        # Prepare test dataframe
        test_df = data['sample_sub'].copy()
        test_df[["book_theater_id", "show_date"]] = test_df["ID"].str.rsplit("_", n=1, expand=True)
        test_df = self.normalize_dates(test_df, ["show_date"])
        
        # Create full dataframe
        full_df = self.create_full_dataframe(
            data['visits'], test_df, bn_daily, pos_daily,
            all_theaters, data['date_info']
        )
        
        # Create train/val split
        train_df, val_df, test_df = self.create_train_val_split(full_df)
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'full': full_df,
            'bookings_bn': bookings_bn,
            'bookings_pos': bookings_pos,
            'all_theaters': all_theaters
        }


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.process()
    print("\n[OK] Data preprocessing complete!")

