import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to dataset."""
    new_df = df.copy()
    bins = [-1, 12, 24, 48, 72, 80]
    labels = ['0-1yr', '1-2yrs', '2-4yrs', '4-6yrs', '6yrs+']
    new_df['tenure_group'] = pd.cut(new_df['tenure'], bins=bins, labels=labels)

    monthly_mean = new_df['MonthlyCharges'].mean()
    new_df['HighSpender'] = (new_df['MonthlyCharges'] > monthly_mean).astype(int)

    if 'PaymentMethod' in new_df.columns:
        risky_methods = ['Electronic check']
        new_df['PaymentRisk'] = new_df['PaymentMethod'].apply(lambda x: 1 if x in risky_methods else 0)

    new_df['HighChurnRisk'] = new_df.apply(
        lambda x: 1 if (x['SeniorCitizen']==1 and x['MonthlyCharges'] > monthly_mean and x['tenure'] <= 12) else 0,
        axis=1
    )
    return new_df