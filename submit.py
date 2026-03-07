"""
Submission Script
Formats predictions and validates submission format
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def validate_submission(submission_file, sample_submission_file="sample_submission.csv"):
    """Validate submission format"""
    print("Validating submission format...")
    
    # Load files
    submission = pd.read_csv(submission_file)
    sample = pd.read_csv(sample_submission_file)
    
    # Check columns
    if list(submission.columns) != list(sample.columns):
        print(f"  ERROR: Column mismatch!")
        print(f"  Expected: {list(sample.columns)}")
        print(f"  Got: {list(submission.columns)}")
        return False
    
    # Check ID format
    sample_ids = set(sample["ID"].values)
    submission_ids = set(submission["ID"].values)
    
    if sample_ids != submission_ids:
        missing = sample_ids - submission_ids
        extra = submission_ids - sample_ids
        if missing:
            print(f"  ERROR: Missing {len(missing)} IDs")
        if extra:
            print(f"  WARNING: {len(extra)} extra IDs")
        return False
    
    # Check for NaN values
    if submission["audience_count"].isna().any():
        print(f"  ERROR: {submission['audience_count'].isna().sum()} NaN values found")
        return False
    
    # Check value range
    if (submission["audience_count"] < 0).any():
        print(f"  WARNING: {submission['audience_count'].lt(0).sum()} negative values found")
    
    print("  [OK] Submission format is valid!")
    print(f"  Total rows: {len(submission):,}")
    print(f"  Value range: [{submission['audience_count'].min():.2f}, {submission['audience_count'].max():.2f}]")
    print(f"  Mean: {submission['audience_count'].mean():.2f}")
    print(f"  Median: {submission['audience_count'].median():.2f}")
    
    return True


def format_submission(predictions, test_df, output_file="submission.csv"):
    """Format predictions for submission"""
    print("Formatting submission...")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        "ID": test_df["ID"].values,
        "audience_count": predictions
    })
    
    # Load sample submission to ensure all IDs are present
    sample_sub = pd.read_csv("sample_submission.csv")
    submission_df = sample_sub[["ID"]].merge(submission_df, on="ID", how="left")
    
    # Fill missing with 0
    submission_df["audience_count"] = submission_df["audience_count"].fillna(0)
    
    # Ensure non-negative
    submission_df["audience_count"] = np.maximum(submission_df["audience_count"], 0)
    
    # Save
    submission_df.to_csv(output_file, index=False)
    print(f"  [OK] Submission saved to {output_file}")
    
    # Validate
    validate_submission(output_file)
    
    return submission_df


if __name__ == "__main__":
    # This script is typically called from main.py
    # But can be used standalone if needed
    print("Submission formatting script")
    print("Run main.py to generate predictions first")

