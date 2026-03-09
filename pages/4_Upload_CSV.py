import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_artifacts, predict_batch, risk_color
from src.train_model import train

st.set_page_config(page_title="Upload CSV | UPI Fraud", page_icon="📁", layout="wide")

REQUIRED_COLS = ['amount', 'hour_of_day', 'location', 'transaction_type',
                 'sender_bank', 'receiver_bank', 'is_new_device', 'failed_attempts']

st.markdown("## 📁 Upload & Analyse Your CSV")
st.caption("Upload any UPI transaction CSV — the model will predict fraud probability for every row")
st.divider()

# ── Auto-train if needed ──────────────────────────────────────────────────────
artifacts = load_artifacts()
if artifacts is None:
    st.info("🤖 Training models first — takes ~10 seconds...")
    with st.spinner("Training..."):
        train()
    artifacts = load_artifacts()

# ── Template download ────────────────────────────────────────────────────────
with st.expander("📋 Need a template? Download sample CSV format"):
    sample = pd.DataFrame([{
        'amount': 5000, 'hour_of_day': 14, 'location': 'Mumbai',
        'transaction_type': 'P2P', 'sender_bank': 'HDFC',
        'receiver_bank': 'SBI', 'is_new_device': 0, 'failed_attempts': 0
    }, {
        'amount': 87000, 'hour_of_day': 2, 'location': 'Unknown',
        'transaction_type': 'P2P', 'sender_bank': 'ICICI',
        'receiver_bank': 'Axis', 'is_new_device': 1, 'failed_attempts': 3
    }])
    st.dataframe(sample, use_container_width=True)
    st.download_button("⬇️ Download Template CSV",
                        sample.to_csv(index=False).encode(),
                        file_name="upi_template.csv", mime="text/csv")

st.write("")

# ── File uploader ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your CSV file", type=["csv"],
                              help=f"Required columns: {', '.join(REQUIRED_COLS)}")

if uploaded is None:
    st.markdown("""
    <div style="background:#1C2333;border-radius:12px;padding:40px;text-align:center;color:#8B9BB4;margin-top:20px">
      <div style="font-size:3rem">📂</div>
      <p>Drag and drop a CSV file above, or click to browse</p>
      <p style="font-size:.85rem">Required columns: amount · hour_of_day · location · transaction_type · sender_bank · receiver_bank · is_new_device · failed_attempts</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load & validate ───────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: **{', '.join(missing)}**")
    st.info(f"Your file has: {', '.join(df.columns.tolist())}")
    st.stop()

# Auto-coerce types silently
for num_col in ['amount', 'hour_of_day', 'is_new_device', 'failed_attempts']:
    df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0)

st.success(f"✅ File loaded — {len(df):,} rows, {len(df.columns)} columns")

# ── Preview ───────────────────────────────────────────────────────────────────
with st.expander("👀 Preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# ── Run prediction ────────────────────────────────────────────────────────────
if st.button("🚀 Run Fraud Detection on All Rows", type="primary", use_container_width=True):
    with st.spinner(f"Analysing {len(df):,} transactions..."):
        results = predict_batch(artifacts, df)

    st.divider()
    st.markdown("## 📊 Results")

    # KPIs
    fraud_count  = (results['verdict'] == 'FRAUD').sum()
    high_count   = (results['risk_level'] == 'HIGH').sum()
    med_count    = (results['risk_level'] == 'MEDIUM').sum()
    avg_prob     = results['fraud_prob'].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Analysed", f"{len(results):,}")
    k2.metric("🚨 Flagged as Fraud", f"{fraud_count:,}",
               delta=f"{fraud_count/len(results)*100:.1f}%", delta_color="inverse")
    k3.metric("🟡 Medium Risk", f"{med_count:,}")
    k4.metric("Avg Risk Score", f"{avg_prob:.1f}%")

    st.write("")

    # Charts
    c1, c2 = st.columns(2)

    with c1:
        risk_counts = results['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        color_map = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
        fig = px.pie(risk_counts, values='Count', names='Risk Level',
                     color='Risk Level', color_discrete_map=color_map, hole=0.4,
                     title='Risk Level Distribution')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA',
                          margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(results, x='fraud_prob', nbins=40,
                           color_discrete_sequence=['#E74C3C'],
                           title='Fraud Probability Distribution',
                           labels={'fraud_prob': 'Fraud Probability (%)'})
        fig.add_vline(x=30, line_dash='dash', line_color='#f39c12', annotation_text='Medium')
        fig.add_vline(x=60, line_dash='dash', line_color='#e74c3c', annotation_text='High')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#FAFAFA', margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown("### 🗂️ Full Results Table")
    display = results.copy()
    display['amount']     = display['amount'].apply(lambda x: f"₹{x:,.2f}")
    display['verdict']    = display['verdict'].apply(
        lambda v: '🚨 FRAUD' if v == 'FRAUD' else '✅ LEGIT')

    show_cols = ['amount', 'hour_of_day', 'location', 'transaction_type',
                 'is_new_device', 'failed_attempts', 'rf_prob', 'xgb_prob',
                 'fraud_prob', 'risk_level', 'verdict']
    show_cols = [c for c in show_cols if c in display.columns]

    st.dataframe(
    display[show_cols],
    use_container_width=True, height=450
    )

    # Download results
    dl_cols = show_cols
    st.download_button(
        "⬇️ Download Results CSV",
        results[dl_cols].to_csv(index=False).encode(),
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        use_container_width=True
    )
