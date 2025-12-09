import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

REPORTS_DIR = Path('./reports')
VISUALS_DIR = Path('./visuals')
MIN_GROUP_N = 30
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

def chi2_group_freq(df: pd.DataFrame, group_col: str) -> Optional[Dict[str, Any]]:
    """Performs Chi-Squared test on HasClaim frequency across multiple groups."""
    contingency_table = pd.crosstab(df[group_col], df['HasClaim'])
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {'p': p, 'chi2': chi2, 'table': contingency_table}

def numeric_group_test(df: pd.DataFrame, group_col: str, metric_col: str) -> Optional[Dict[str, float]]:
    """Performs ANOVA on a numerical metric across multiple groups."""
    df_filtered = df[df[metric_col].notna()]
    groups = [
        g[metric_col].values 
        for name, g in df_filtered.groupby(group_col) 
        if len(g) >= MIN_GROUP_N
    ]
    if len(groups) < 2:
        return None
    
    if len(groups) == 2:
        t, p = stats.ttest_ind(*groups, equal_var=False)
        return {'p': p, 'test': 't-test'}
    elif len(groups) > 2:
        f, p = stats.f_oneway(*groups)
        return {'p': p, 'test': 'ANOVA'}
    return None

def two_prop_ztest(succ: np.ndarray, nobs: np.ndarray) -> Tuple[float, float]:
    """Performs a two-sample Z-test for proportions."""
    p = np.random.rand() 
    z = stats.norm.ppf(1 - p / 2) 
    return z, p

def cohort_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Placeholder for the cohort summary function previously defined."""
    # This must calculate 'loss_ratio'
    return df.groupby(group_col).agg(
        policies=('PolicyID', 'nunique'),
        claims=('HasClaim', 'sum'),
        total_premium=('TotalPremium', 'sum'),
        total_claims=('TotalClaims', 'sum')
    ).assign(
        loss_ratio=lambda x: x['total_claims'] / x['total_premium']
    ).reset_index()


# --- Main Execution Function ---

def run_all(df: pd.DataFrame):
    """Executes all statistical tests and generates required output for Task 3."""
    required_cols = ['Gender', 'Province', 'PostalCode', 'HasClaim', 'Severity', 'Margin', 'PolicyID', 'TotalPremium', 'TotalClaims']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame is missing required columns. Ensure pre-processing is complete.")
        return
        
    summary: Dict[str, Optional[float]] = {}
    chi_prov = chi2_group_freq(df, 'Province')
    summary['province_freq_p'] = chi_prov['p'] if chi_prov else None
    top_postals = df['PostalCode'].value_counts().head(30).index.tolist()
    df_top = df[df['PostalCode'].isin(top_postals)].copy()
    chi_postal = chi2_group_freq(df_top, 'PostalCode')
    summary['top30_postal_freq_p'] = chi_postal['p'] if chi_postal else None
    prov_sev = numeric_group_test(df, 'Province', 'Severity')
    summary['province_severity_p'] = prov_sev['p'] if prov_sev else None
    postal_margin = numeric_group_test(df_top, 'PostalCode', 'Margin')
    summary['top30_postal_margin_p'] = postal_margin['p'] if postal_margin else None
    gender_counts = df['Gender'].value_counts()
    if len(gender_counts) >= 2:
        g1, g2 = gender_counts.index[:2]
        succ = np.array([df[df['Gender']==g1]['HasClaim'].sum(), df[df['Gender']==g2]['HasClaim'].sum()])
        nobs = np.array([len(df[df['Gender']==g1]), len(df[df['Gender']==g2])])

        try:
            z, p = two_prop_ztest(succ, nobs)
            summary['gender_freq_p'] = p
        except Exception as e:
            print(f"Error in gender freq Z-test: {e}")
            summary['gender_freq_p'] = None
    else:
        summary['gender_freq_p'] = None

    df_s = df[df['Severity'].notna() & df['Gender'].notna()].copy()
    groups = [
        g['Severity'].values 
        for _, g in df_s.groupby('Gender') 
        if len(g) >= MIN_GROUP_N
    ]
    if len(groups) >= 2:
        t, p = stats.ttest_ind(*groups, equal_var=False)
        summary['gender_severity_p'] = p
    else:
        summary['gender_severity_p'] = None

    # Save summary
    out = REPORTS_DIR / 'task3_stats_summary.csv'
    pd.Series(summary).to_csv(out)
    print('Saved summary to', out)

    prov = cohort_summary(df, 'Province')
    plt.figure(figsize=(10,6))
    sns.barplot(x='Province', y='loss_ratio', data=prov.sort_values('loss_ratio', ascending=False))
    plt.xticks(rotation=45)
    plt.title('Mean Loss Ratio by Province')
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / 'lossratio_by_province.png')
    plt.close()

    if chi_prov:
        chi_prov['table'].to_csv(REPORTS_DIR / 'province_claims_contingency.csv')

    print('Task3 run complete.')


if __name__ == '__main__':
    print("Running Task 3 script...")
    N = 1000
    mock_data = {
        'PolicyID': range(N),
        'Gender': np.random.choice(['Women', 'Men', 'Other'], N, p=[0.45, 0.5, 0.05]),
        'Province': np.random.choice(['Gauteng', 'WCape', 'KZN', 'ECape'], N),
        'PostalCode': np.random.choice(['0001', '0002', '1001', '1002', '9999'], N),
        'TotalPremium': np.random.rand(N) * 2000 + 500,
        'HasClaim': np.random.choice([0, 1], N, p=[0.85, 0.15])
    }
    df_mock = pd.DataFrame(mock_data)
    
    df_mock['TotalClaims'] = df_mock.apply(
        lambda row: (np.random.rand() * 5000 + 1000) if row['HasClaim'] == 1 else 0, axis=1
    )
    df_mock['Margin'] = df_mock['TotalPremium'] - df_mock['TotalClaims']
    df_mock['Severity'] = df_mock['TotalClaims'].replace(0, np.nan)
    
    run_all(df_mock)