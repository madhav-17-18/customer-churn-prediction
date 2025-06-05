# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess():
    # Load data
    df = pd.read_csv("C:\\Users\\saima\\OneDrive\\Desktop\\internship\\Churn_Modelling (1).csv")

    # Step 1: Drop irrelevant columns
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # Step 2: Encode categorical features
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Female=0, Male=1
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)  # One-hot encoding

    # Step 3: Add a binary flag for zero balances
    df['ZeroBalance'] = (df['Balance'] == 0).astype(int)

    # Step 4: Split into features and label
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Step 5: Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 6: Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
