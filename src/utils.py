import pandas as pd
import numpy as np
import joblib
import os

FEATURES = [
    'amount', 'hour_of_day', 'is_new_device', 'failed_attempts',
    'is_night', 'is_high_amount', 'is_unknown_loc',
    'location_enc', 'transaction_type_enc', 'sender_bank_enc', 'receiver_bank_enc'
]

RISK_THRESHOLDS = {'low': 30, 'medium': 60}   # % below = low, above 60 = high


def engineer_features(df: pd.DataFrame, encoders: dict, high_amount_threshold: float) -> pd.DataFrame:
    """Add engineered features and encode categoricals. Returns feature matrix."""
    df = df.copy()
    df['is_night']       = df['hour_of_day'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df['is_high_amount'] = (df['amount'] > high_amount_threshold).astype(int)
    df['is_unknown_loc'] = df['location'].apply(lambda l: 1 if l in ['Unknown', 'Foreign'] else 0)

    def safe_enc(enc, col):
        return df[col].apply(lambda v: enc.transform([v])[0] if v in enc.classes_ else -1)

    df['location_enc']         = safe_enc(encoders['location'], 'location')
    df['transaction_type_enc'] = safe_enc(encoders['type'],     'transaction_type')
    df['sender_bank_enc']      = safe_enc(encoders['sbank'],    'sender_bank')
    df['receiver_bank_enc']    = safe_enc(encoders['rbank'],    'receiver_bank')

    return df[FEATURES]


def load_artifacts():
    """Load models + encoders + threshold. Returns dict or None if not trained yet."""
    paths = {
        'rf' : 'models/rf_model.pkl',
        'xgb': 'models/xgb_model.pkl',
        'enc': 'models/encoders.pkl',
        'thr': 'models/threshold.pkl',
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return {
        'rf'       : joblib.load(paths['rf']),
        'xgb'      : joblib.load(paths['xgb']),
        'encoders' : joblib.load(paths['enc']),
        'threshold': joblib.load(paths['thr']),
    }


def predict_single(artifacts: dict, amount, hour, location, txn_type,
                   sender_bank, receiver_bank, is_new_device, failed_attempts):
    """
    Run both models on a single transaction.
    Returns dict with rf_prob, xgb_prob, avg_prob, verdict, risk_level, explanation.
    """
    row = pd.DataFrame([{
        'amount'          : amount,
        'hour_of_day'     : hour,
        'location'        : location,
        'transaction_type': txn_type,
        'sender_bank'     : sender_bank,
        'receiver_bank'   : receiver_bank,
        'is_new_device'   : is_new_device,
        'failed_attempts' : failed_attempts,
    }])

    X = engineer_features(row, artifacts['encoders'], artifacts['threshold'])

    rf_prob  = artifacts['rf'].predict_proba(X)[0][1] * 100
    xgb_prob = artifacts['xgb'].predict_proba(X)[0][1] * 100
    avg_prob = (rf_prob + xgb_prob) / 2

    risk_level = (
        'HIGH'   if avg_prob >= RISK_THRESHOLDS['medium'] else
        'MEDIUM' if avg_prob >= RISK_THRESHOLDS['low']    else
        'LOW'
    )
    verdict = 'FRAUD' if avg_prob >= RISK_THRESHOLDS['medium'] else 'LEGITIMATE'

    explanation = _build_explanation(amount, hour, location, is_new_device,
                                     failed_attempts, artifacts['threshold'])

    return {
        'rf_prob'   : round(rf_prob, 1),
        'xgb_prob'  : round(xgb_prob, 1),
        'avg_prob'  : round(avg_prob, 1),
        'verdict'   : verdict,
        'risk_level': risk_level,
        'explanation': explanation,
    }


def predict_batch(artifacts: dict, df: pd.DataFrame):
    """Run prediction on a full dataframe. Returns df with new columns."""
    df = df.copy()
    X = engineer_features(df, artifacts['encoders'], artifacts['threshold'])
    df['rf_prob']      = (artifacts['rf'].predict_proba(X)[:, 1]  * 100).round(1)
    df['xgb_prob']     = (artifacts['xgb'].predict_proba(X)[:, 1] * 100).round(1)
    df['fraud_prob']   = ((df['rf_prob'] + df['xgb_prob']) / 2).round(1)
    df['risk_level']   = df['fraud_prob'].apply(
        lambda p: 'HIGH' if p >= RISK_THRESHOLDS['medium']
        else ('MEDIUM' if p >= RISK_THRESHOLDS['low'] else 'LOW')
    )
    df['verdict'] = df['fraud_prob'].apply(
        lambda p: 'FRAUD' if p >= RISK_THRESHOLDS['medium'] else 'LEGITIMATE'
    )
    return df


def _build_explanation(amount, hour, location, is_new_device, failed_attempts, threshold):
    """Return list of plain-English reasons why this looks fraudulent or safe."""
    flags = []
    clears = []

    if hour < 6 or hour >= 22:
        flags.append(f"🕐 Transaction at {hour}:00 — unusual late night / early morning hour")
    else:
        clears.append(f"🕐 Normal business hour ({hour}:00)")

    if location in ['Unknown', 'Foreign']:
        flags.append(f"📍 Location is '{location}' — high-risk origin")
    else:
        clears.append(f"📍 Known location ({location})")

    if amount > threshold:
        flags.append(f"💰 High amount ₹{amount:,.0f} — top 10% of transactions")
    else:
        clears.append(f"💰 Normal transaction amount (₹{amount:,.0f})")

    if is_new_device:
        flags.append("📱 New / unrecognised device used")
    else:
        clears.append("📱 Known device")

    if failed_attempts >= 2:
        flags.append(f"⚠️ {failed_attempts} failed PIN attempts before this transaction")
    elif failed_attempts == 1:
        flags.append("⚠️ 1 failed PIN attempt noted")
    else:
        clears.append("⚠️ No failed PIN attempts")

    return {'flags': flags, 'clears': clears}


def load_data():
    """Load transactions CSV, generate if missing."""
    path = 'data/transactions.csv'
    if not os.path.exists(path):
        from src.data_generator import generate_transactions
        return generate_transactions()
    return pd.read_csv(path)


def risk_color(level: str) -> str:
    return {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}.get(level, '#ffffff')
