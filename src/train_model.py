"""
Train Random Forest + XGBoost models on UPI transaction data.
Run: python src/train_model.py
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed — only Random Forest will be trained.")
    print("    Run: pip install xgboost")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import generate_transactions
from src.utils import FEATURES


def train(csv_path='data/transactions.csv', models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)

    # ── Load / generate data ──────────────────────────────────────────────────
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} transactions from {csv_path}")
    else:
        print("⚙️  No CSV found — generating synthetic data...")
        df = generate_transactions()
        print(f"✅ Generated {len(df):,} transactions")

    print(f"   Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")

    # ── Encoders ──────────────────────────────────────────────────────────────
    le_location = LabelEncoder().fit(df['location'])
    le_type     = LabelEncoder().fit(df['transaction_type'])
    le_sbank    = LabelEncoder().fit(df['sender_bank'])
    le_rbank    = LabelEncoder().fit(df['receiver_bank'])
    encoders    = {'location': le_location, 'type': le_type,
                   'sbank': le_sbank, 'rbank': le_rbank}

    high_amount_threshold = float(df['amount'].quantile(0.90))

    # ── Feature engineering ───────────────────────────────────────────────────
    df['is_night']       = df['hour_of_day'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df['is_high_amount'] = (df['amount'] > high_amount_threshold).astype(int)
    df['is_unknown_loc'] = df['location'].apply(lambda l: 1 if l in ['Unknown', 'Foreign'] else 0)

    df['location_enc']         = le_location.transform(df['location'])
    df['transaction_type_enc'] = le_type.transform(df['transaction_type'])
    df['sender_bank_enc']      = le_sbank.transform(df['sender_bank'])
    df['receiver_bank_enc']    = le_rbank.transform(df['receiver_bank'])

    X = df[FEATURES]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n⏳ Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    print(f"✅ Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(classification_report(y_test, rf_preds, target_names=['Legit', 'Fraud']))
    joblib.dump(rf, f'{models_dir}/rf_model.pkl')

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if HAS_XGB:
        print("⏳ Training XGBoost...")
        scale = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale, use_label_encoder=False,
            eval_metric='logloss', random_state=42, verbosity=0
        )
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_acc   = accuracy_score(y_test, xgb_preds)
        print(f"✅ XGBoost Accuracy: {xgb_acc*100:.2f}%")
        print(classification_report(y_test, xgb_preds, target_names=['Legit', 'Fraud']))
        joblib.dump(xgb, f'{models_dir}/xgb_model.pkl')
    else:
        # Save a dummy copy of RF as XGB fallback so app doesn't break
        joblib.dump(rf, f'{models_dir}/xgb_model.pkl')
        print("⚠️  Saved RF as XGB fallback (install xgboost for real XGB model)")

    # ── Save encoders + threshold ─────────────────────────────────────────────
    joblib.dump(encoders, f'{models_dir}/encoders.pkl')
    joblib.dump(high_amount_threshold, f'{models_dir}/threshold.pkl')
    print(f"\n✅ All models saved to /{models_dir}")
    print(f"   High-amount threshold: ₹{high_amount_threshold:,.2f}")


if __name__ == '__main__':
    train()
