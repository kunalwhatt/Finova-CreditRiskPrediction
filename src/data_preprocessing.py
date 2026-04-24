import pandas as pd

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df['income'] = df['income'].fillna(df['income'].median())
    df['loan_amount'] = df['loan_amount'].fillna(df['loan_amount'].median())
    df = df[df['income'] > 1000]
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, drop_first=True)
