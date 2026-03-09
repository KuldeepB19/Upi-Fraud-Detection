import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_artifacts, engineer_features, FEATURES
from src.train_model import train
from src.data_generator import generate_transactions
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.markdown("""
<style>
  .metric-card {
    background: #1C2333; border-radius: 12px; padding: 20px 24px;
    border-left: 4px solid; margin-bottom: 8px; text-align: center;
  }
  .metric-val   { font-size: 2rem; font-weight: 700; margin: 0; }
  .metric-label { font-size: 0.8rem; color: #8B9BB4; margin: 0;
                  text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 Model Performance")
st.caption("Evaluation metrics for Random Forest and XGBoost on test data")
st.divider()

# ── Load artifacts ────────────────────────────────────────────────────────────
artifacts = load_artifacts()
if artifacts is None:
    st.info("No trained model found. Training now...")
    with st.spinner("Training..."):
        train()
    artifacts = load_artifacts()

# ── Rebuild test set (same seed = same split as training) ─────────────────────
@st.cache_data
def get_test_data():
    path = 'data/transactions.csv'
    if not os.path.exists(path):
        df = generate_transactions()
    else:
        df = pd.read_csv(path)

    X = engineer_features(df, artifacts['encoders'], artifacts['threshold'])
    y = df['is_fraud']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

X_test, y_test = get_test_data()

# ── Get predictions ───────────────────────────────────────────────────────────
rf_preds  = artifacts['rf'].predict(X_test)
xgb_preds = artifacts['xgb'].predict(X_test)

# Ensemble: average probabilities then threshold at 0.5
rf_proba  = artifacts['rf'].predict_proba(X_test)[:, 1]
xgb_proba = artifacts['xgb'].predict_proba(X_test)[:, 1]
ens_proba = (rf_proba + xgb_proba) / 2
ens_preds = (ens_proba >= 0.5).astype(int)

def metrics(y_true, y_pred, name):
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_true, y_pred) * 100, 2),
        'Precision': round(precision_score(y_true, y_pred) * 100, 2),
        'Recall'   : round(recall_score(y_true, y_pred) * 100, 2),
        'F1 Score' : round(f1_score(y_true, y_pred) * 100, 2),
    }

results = [
    metrics(y_test, rf_preds,  'Random Forest'),
    metrics(y_test, xgb_preds, 'XGBoost'),
    metrics(y_test, ens_preds, 'Ensemble (Avg)'),
]
results_df = pd.DataFrame(results)

# ── Model selector ────────────────────────────────────────────────────────────
selected = st.radio("Select model to inspect:",
                    ['Random Forest', 'XGBoost', 'Ensemble (Avg)'],
                    horizontal=True)

preds_map = {
    'Random Forest'  : rf_preds,
    'XGBoost'        : xgb_preds,
    'Ensemble (Avg)' : ens_preds,
}
chosen_preds = preds_map[selected]
chosen_row   = results_df[results_df['Model'] == selected].iloc[0]

st.divider()

# ── 4 KPI cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

def card(col, label, value, color):
    col.markdown(f"""
    <div class="metric-card" style="border-color:{color}">
      <p class="metric-label">{label}</p>
      <p class="metric-val" style="color:{color}">{value}%</p>
    </div>""", unsafe_allow_html=True)

card(c1, "Accuracy",  chosen_row['Accuracy'],  "#2E86C1")
card(c2, "Precision", chosen_row['Precision'], "#8E44AD")
card(c3, "Recall",    chosen_row['Recall'],    "#E74C3C")
card(c4, "F1 Score",  chosen_row['F1 Score'],  "#27AE60")

st.write("")

# ── Side by side: confusion matrix + comparison bar chart ─────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, chosen_preds)
    # cm = [[TN, FP], [FN, TP]]
    labels = ['Legit (0)', 'Fraud (1)']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Legit', 'Predicted Fraud'],
        y=['Actual Legit', 'Actual Fraud'],
        colorscale='Blues',
        text=cm, texttemplate="%{text}",
        showscale=False
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#FAFAFA',
        height=320,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain English explanation below matrix
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    - **True Negatives (TN) = {tn}** — Legit transactions correctly identified as legit  
    - **False Positives (FP) = {fp}** — Legit transactions wrongly flagged as fraud  
    - **False Negatives (FN) = {fn}** — Fraud transactions missed (most dangerous!)  
    - **True Positives (TP) = {tp}** — Fraud transactions correctly caught
    """)

with col_right:
    st.markdown("#### All Models Comparison")
    fig2 = go.Figure()
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors_list  = ['#2E86C1', '#8E44AD', '#E74C3C', '#27AE60']

    for i, row in results_df.iterrows():
        fig2.add_trace(go.Bar(
            name=row['Model'],
            x=metrics_cols,
            y=[row[m] for m in metrics_cols],
            text=[f"{row[m]}%" for m in metrics_cols],
            textposition='outside',
        ))

    fig2.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#FAFAFA',
        yaxis=dict(range=[0, 115]),
        legend=dict(orientation='h', y=-0.2),
        height=320,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Full metrics table ────────────────────────────────────────────────────────
st.markdown("#### Full Metrics Table")
st.dataframe(results_df.set_index('Model'), use_container_width=True)

st.divider()

# ── Feature importance (RF only) ──────────────────────────────────────────────
st.markdown("#### Feature Importance (Random Forest)")
st.caption("Which features matter most when predicting fraud")

importances = artifacts['rf'].feature_importances_
feat_df = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': (importances * 100).round(2)
}).sort_values('Importance', ascending=True)

fig3 = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
              color='Importance', color_continuous_scale='Blues',
              labels={'Importance': 'Importance (%)'})
fig3.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='#FAFAFA',
    coloraxis_showscale=False,
    height=380,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig3, use_container_width=True)
