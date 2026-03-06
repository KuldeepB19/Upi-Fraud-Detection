import pandas as pd
import numpy as np
import os

def generate_transactions(n=10000, fraud_pct=0.10, seed=42):
    """
    Generate synthetic UPI transaction data.
    Returns a DataFrame and also saves to data/transactions.csv
    """
    np.random.seed(seed)

    locations         = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
                         'Kolkata', 'Pune', 'Ahmedabad', 'Unknown', 'Foreign']
    transaction_types = ['P2P', 'P2M', 'Bill Payment', 'Recharge', 'Online Shopping']
    banks             = ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'PNB', 'BOB']

    n_fraud = int(n * fraud_pct)
    n_legit = n - n_fraud

    # --- Legitimate ---
    legit = pd.DataFrame({
        'transaction_id'  : [f'TXN{str(i).zfill(7)}' for i in range(n_legit)],
        'amount'          : np.random.lognormal(7, 1.2, n_legit).clip(10, 50000).round(2),
        'hour_of_day'     : np.random.choice(range(6, 23), n_legit),
        'location'        : np.random.choice(locations[:8], n_legit),
        'transaction_type': np.random.choice(transaction_types, n_legit),
        'sender_bank'     : np.random.choice(banks, n_legit),
        'receiver_bank'   : np.random.choice(banks, n_legit),
        'is_new_device'   : np.random.choice([0, 1], n_legit, p=[0.95, 0.05]),
        'failed_attempts' : np.random.choice([0, 1, 2], n_legit, p=[0.90, 0.08, 0.02]),
        'is_fraud'        : 0,
    })

    # --- Fraudulent ---
    fraud = pd.DataFrame({
        'transaction_id'  : [f'TXN{str(i).zfill(7)}' for i in range(n_legit, n)],
        'amount'          : np.random.lognormal(9, 1.5, n_fraud).clip(500, 200000).round(2),
        'hour_of_day'     : np.random.choice(list(range(0, 5)) + list(range(22, 24)), n_fraud),
        'location'        : np.random.choice(
                                ['Unknown', 'Foreign'] + locations[:3], n_fraud,
                                p=[0.40, 0.30, 0.10, 0.10, 0.10]),
        'transaction_type': np.random.choice(transaction_types, n_fraud),
        'sender_bank'     : np.random.choice(banks, n_fraud),
        'receiver_bank'   : np.random.choice(banks, n_fraud),
        'is_new_device'   : np.random.choice([0, 1], n_fraud, p=[0.30, 0.70]),
        'failed_attempts' : np.random.choice([0, 1, 2, 3], n_fraud, p=[0.20, 0.30, 0.30, 0.20]),
        'is_fraud'        : 1,
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Derived columns
    df['time_of_day'] = df['hour_of_day'].apply(
        lambda h: 'Night' if h < 6 or h >= 22
        else ('Morning' if h < 12
        else ('Afternoon' if h < 17
        else 'Evening'))
    )

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions.csv', index=False)
    return df


LOCATIONS         = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
                     'Kolkata', 'Pune', 'Ahmedabad', 'Unknown', 'Foreign']
TRANSACTION_TYPES = ['P2P', 'P2M', 'Bill Payment', 'Recharge', 'Online Shopping']
BANKS             = ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'PNB', 'BOB']
