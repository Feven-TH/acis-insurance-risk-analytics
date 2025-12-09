import pandas as pd
import numpy as np
from .config import DATA_CLEAN
from .utils import ensure_numeric




def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # ensure numeric
    df = ensure_numeric(df, ['TotalClaims','TotalPremium','RegistrationYear','CustomValueEstimate'])


    # HasClaim
    df['HasClaim'] = (df['TotalClaims'].fillna(0) > 0).astype(int)


    # Margin
    df['Margin'] = df['TotalPremium'].fillna(0) - df['TotalClaims'].fillna(0)


    # Severity: if NumberOfClaims available use it, else use TotalClaims for policies with claim
    if 'NumberOfClaims' in df.columns:
        df['NumberOfClaims'] = pd.to_numeric(df['NumberOfClaims'], errors='coerce')
        df.loc[df['NumberOfClaims']>0, 'Severity'] = df.loc[df['NumberOfClaims']>0, 'TotalClaims'] / df.loc[df['NumberOfClaims']>0, 'NumberOfClaims']
    else:
        df.loc[df['HasClaim']==1, 'Severity'] = df.loc[df['HasClaim']==1, 'TotalClaims']


    # LossRatio
    df['LossRatio'] = np.where(df['TotalPremium']>0, df['TotalClaims'] / df['TotalPremium'], np.nan)


    # Clean PostalCode: string and fillna
    if 'PostalCode' in df.columns:
        df['PostalCode'] = df['PostalCode'].astype(str).fillna('Unknown')


    # Normalize Gender
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.title().replace({'M':'Male','F':'Female'})

    return df




if __name__ == '__main__':
    df = pd.read_csv(DATA_CLEAN)
    df2 = build_features(df)
    out = DATA_CLEAN.parent / 'insurance_claims_task3_ready.csv'
    df2.to_csv(out, index=False)
print('Saved preprocessed file to', out)