import pandas as pd
import numpy as np
from typing import List, Tuple

# --- Core Setup and Loading ---

def load_clean_data(path: str) -> pd.DataFrame:
    """Loads data, ensures a unique policy identifier, and cleans column names."""
    # Note: Assumes the transaction file is a single CSV
    df = pd.read_csv(path)
    
    # Ensure expected columns exist
    if 'PolicyID' not in df.columns:
        # Assuming UnderwrittenCoverID might serve as a policy/cover identifier
        if 'UnderwrittenCoverID' in df.columns:
            df['PolicyID'] = df['UnderwrittenCoverID'].astype(str)
        else:
            df['PolicyID'] = df.index.astype(str) # Fallback
            
    # Simple cleaning (e.g., removing leading/trailing spaces from column names)
    df.columns = df.columns.str.strip()
    
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Converts a list of columns to numeric, coercing errors to NaN."""
    for c in cols:
        if c in df.columns:
            # Setting errors='coerce' replaces non-numeric values with NaN
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

# --- Feature Engineering (Derived Metrics) ---

def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the key insurance risk and profit metrics required for A/B testing and EDA.
    
    Metrics: HasClaim, Claim_Amount_GT_0 (alias), Severity, Margin.
    """
    
    # Ensure financial columns are numeric first
    financial_cols = ['TotalPremium', 'TotalClaims']
    df = ensure_numeric(df, financial_cols)
    
    # 1. HasClaim (Risk Frequency indicator)
    # A claim is assumed to have occurred if TotalClaims > 0
    df['HasClaim'] = np.where(df['TotalClaims'] > 0, 1, 0)
    
    # Alias for clarity in some analyses
    df['Claim_Amount_GT_0'] = df['TotalClaims'].apply(lambda x: x if x > 0 else np.nan)
    
    # 2. Margin (Profit Metric)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    
    # 3. Severity (Average Claim Amount, GIVEN a Claim Occurred)
    # We use a policy-level severity which is NaN if no claim occurred, 
    # but the cohort_summary will calculate the true mean severity later.
    df['Severity'] = df['Claim_Amount_GT_0']
    
    print("Derived metrics (HasClaim, Margin, Severity) calculated.")
    return df

# --- Cohort Summary ---

def cohort_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return grouped counts and simple metrics (loss ratio, freq, severity, margin mean)"""
    
    # Check if required derived columns exist
    required_cols = ['PolicyID', 'HasClaim', 'TotalPremium', 'TotalClaims', 'Severity', 'Margin']
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"Required columns {required_cols} missing. Run calculate_derived_metrics first.")
        
    g = df.groupby(group_col).agg(
        policies=('PolicyID', 'nunique'),          # Total number of unique policies
        claims=('HasClaim', 'sum'),               # Total number of policies with a claim
        total_premium=('TotalPremium', 'sum'),    # Sum of premiums for the cohort
        total_claims=('TotalClaims', 'sum'),      # Sum of claims for the cohort
        
        # Mean severity is calculated only over non-NaN values (i.e., where claims occurred)
        mean_severity=('Severity', 'mean'),
        mean_margin=('Margin', 'mean')            # Average margin per policy in the cohort
    )
    
    # Calculate Risk Metrics (Portfolio Level)
    g['claim_freq'] = g['claims'] / g['policies']               # Claim Frequency (Proportion)
    g['loss_ratio'] = g['total_claims'] / g['total_premium']    # Loss Ratio (Claim Cost / Premium Earned)
    
    return g.reset_index()


# --- Example Usage (Conceptual) ---

if __name__ == '__main__':
    # REPLACE WITH YOUR ACTUAL DATA PATH
    data_path = 'path/to/your/insurance_data.csv' 
    
    # 1. Load and Clean
    # df_raw = load_clean_data(data_path) 
    
    # Create mock data for demonstration if actual data isn't available
    data = {
        'PolicyID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Gender': ['Women', 'Men', 'Women', 'Men', 'Women', 'Men', 'Women', 'Men'],
        'Province': ['Gauteng', 'Gauteng', 'WCape', 'WCape', 'Gauteng', 'WCape', 'Gauteng', 'WCape'],
        'TotalPremium': [1000, 1200, 800, 1100, 1050, 950, 850, 1150],
        'TotalClaims': [0, 500, 0, 200, 2000, 0, 0, 600]
    }
    df_raw = pd.DataFrame(data)
    
    print("--- 1. Raw Data Loaded ---")
    print(df_raw.head())
    
    # 2. Calculate Derived Metrics
    df_processed = calculate_derived_metrics(df_raw)
    
    print("\n--- 2. Processed Data with Derived Metrics ---")
    print(df_processed[['PolicyID', 'Gender', 'TotalPremium', 'TotalClaims', 'HasClaim', 'Margin', 'Severity']].head(8))
    
    # 3. Generate Cohort Summary for a group (e.g., Province)
    province_summary = cohort_summary(df_processed, group_col='Province')
    
    print("\n--- 3. Province Cohort Summary (EDA/A/B Test Input) ---")
    print(province_summary.set_index('Province'))
    
    # Example interpretation:
    # If loss_ratio for Gauteng > WCape, Gauteng is 'riskier' by this metric.
    # If claim_freq for Gauteng > WCape, Gauteng is 'riskier' by this metric.
    # If mean_severity for Gauteng > WCape, Gauteng claims are 'more expensive'.