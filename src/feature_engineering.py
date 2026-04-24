import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['emi_income_ratio'] = df['emi'] / df['income']
    df['loan_income_ratio'] = df['loan_amount'] / df['income']
    df['age_band'] = pd.cut(df['age'], bins=[18,25,35,50,100], labels=[0,1,2,3])
    return df

def select_features(df: pd.DataFrame):
    y = df['default']
    X = df.drop(columns=['default'])
    return X, y
