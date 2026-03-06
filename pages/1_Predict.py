import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_artifacts, predict_single, risk_color
from src.data_generator import LOCATIONS, TRANSACTION_TYPES, BANKS
from src.train_model import train

st.set_page_config(page_title="Predict | UPI Fraud", page_icon="🔮", layout="wide")

st.markdown("""
<style>
  .result-box { border-radius:14px; padding:24px 28px; margin-top:12px; }
  .verdict    { font-size:1.8rem; font-weight:800; margin:0; }
  .prob-label { font-size:0.8rem; color:#8B9BB4; text-transform:uppercase; letter-spacing:.05em; }
  .prob-val   { font-size:1.4rem; font-weight:700; margin:0; }
  .flag-item  { background:#2C1B1B; border-left:3px solid #e74c3c;
                padding:8px 14px; border-radius:6px; margin:4px 0; font-size:.9rem; }
  .clear-item { background:#1B2C1B; border-left:3px solid #2ecc71;
                padding:8px 14px; border-radius:6px; margin:4px 0; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔮 Transaction Fraud Predictor")
st.caption("Enter transaction details — both models analyse it and give a fraud risk score")
st.divider()

# ── Auto-train if models missing ──────────────────────────────────────────────
artifacts = load_artifacts()
if artifacts is None:
    st.info("🤖 No trained model found. Training now — this takes about 10 seconds...")
    with st.spinner("Training RF + XGBoost..."):
        train()
    artifacts = load_artifacts()
    st.success("✅ Models trained and ready!")

# ── Pre-fill session state defaults ──────────────────────────────────────────
if 'ex_amount' not in st.session_state:
    st.session_state.ex_amount = 5000.0
if 'ex_hour' not in st.session_state:
    st.session_state.ex_hour = 14
if 'ex_location' not in st.session_state:
    st.session_state.ex_location = LOCATIONS[0]
if 'ex_new_device' not in st.session_state:
    st.session_state.ex_new_device = False
if 'ex_failed' not in st.session_state:
    st.session_state.ex_failed = 0
if 'ex_txn_type' not in st.session_state:
    st.session_state.ex_txn_type = TRANSACTION_TYPES[0]
if 'ex_sender_bank' not in st.session_state:
    st.session_state.ex_sender_bank = BANKS[0]
if 'ex_receiver_bank' not in st.session_state:
    st.session_state.ex_receiver_bank = BANKS[0]

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📝 Transaction Details")

    # ── Try example buttons ───────────────────────────────────────────────────
    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("🚨 Try Fraud Example", use_container_width=True):
            st.session_state.ex_amount       = 87000.0
            st.session_state.ex_hour         = 2
            st.session_state.ex_location     = "Unknown"
            st.session_state.ex_new_device   = True
            st.session_state.ex_failed       = 3
            st.session_state.ex_txn_type     = "P2P"
            st.session_state.ex_sender_bank  = BANKS[0]
            st.session_state.ex_receiver_bank= BANKS[1]
            st.rerun()
    with btn2:
        if st.button("✅ Try Legit Example", use_container_width=True):
            st.session_state.ex_amount       = 1500.0
            st.session_state.ex_hour         = 11
            st.session_state.ex_location     = "Mumbai"
            st.session_state.ex_new_device   = False
            st.session_state.ex_failed       = 0
            st.session_state.ex_txn_type     = "Bill Payment"
            st.session_state.ex_sender_bank  = BANKS[0]
            st.session_state.ex_receiver_bank= BANKS[0]
            st.rerun()

    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("💰 Amount (₹)", min_value=1.0, max_value=500000.0,
                                  value=st.session_state.ex_amount, step=100.0)
        location = st.selectbox("📍 Location", LOCATIONS,
                                  index=LOCATIONS.index(st.session_state.ex_location))
        sender_bank = st.selectbox("🏛️ Sender Bank", BANKS,
                                    index=BANKS.index(st.session_state.ex_sender_bank))
        is_new_device = st.toggle("📱 New / Unknown Device",
                                    value=st.session_state.ex_new_device)

    with col2:
        hour = st.slider("🕐 Hour of Day", 0, 23, st.session_state.ex_hour,
                          help="0 = midnight, 14 = 2pm")
        txn_type = st.selectbox("🔄 Transaction Type", TRANSACTION_TYPES,
                                  index=TRANSACTION_TYPES.index(st.session_state.ex_txn_type))
        receiver_bank = st.selectbox("🏛️ Receiver Bank", BANKS,
                                      index=BANKS.index(st.session_state.ex_receiver_bank))
        failed_attempts = st.slider("⚠️ Failed PIN Attempts", 0, 4,
                                     st.session_state.ex_failed)

    st.write("")
    model_choice = st.radio("🤖 Model", ["Both (Recommended)", "Random Forest only", "XGBoost only"],
                             horizontal=True)

    run = st.button("🔍 Analyse Transaction", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
with right:
    st.markdown("### 📊 Analysis Result")

    if not run:
        st.markdown("""
        <div style="background:#1C2333;border-radius:12px;padding:40px;text-align:center;color:#8B9BB4">
          <div style="font-size:3rem">🔮</div>
          <p>Fill in the transaction details on the left<br>and click <b>Analyse Transaction</b></p>
        </div>""", unsafe_allow_html=True)
    else:
        result = predict_single(
            artifacts, amount, hour, location, txn_type,
            sender_bank, receiver_bank, int(is_new_device), failed_attempts
        )

        r_color = risk_color(result['risk_level'])
        is_fraud = result['verdict'] == 'FRAUD'

        # Verdict banner
        st.markdown(f"""
        <div class="result-box" style="background:{'#2C1B1B' if is_fraud else '#1B2C1B'};
             border: 2px solid {r_color}">
          <p class="verdict" style="color:{r_color}">
            {'🚨 FRAUDULENT TRANSACTION' if is_fraud else '✅ LEGITIMATE TRANSACTION'}
          </p>
          <p style="color:#8B9BB4;margin:4px 0 0">
            Risk Level: <b style="color:{r_color}">{result['risk_level']}</b>
          </p>
        </div>""", unsafe_allow_html=True)

        st.write("")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['avg_prob'],
            number={'suffix': "%", 'font': {'size': 40, 'color': r_color}},
            title={'text': "Fraud Risk Score", 'font': {'color': '#FAFAFA', 'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8B9BB4'},
                'bar': {'color': r_color, 'thickness': 0.3},
                'bgcolor': '#1C2333',
                'steps': [
                    {'range': [0, 30],  'color': '#1B2C1B'},
                    {'range': [30, 60], 'color': '#2C2B1B'},
                    {'range': [60, 100],'color': '#2C1B1B'},
                ],
                'threshold': {'line': {'color': r_color, 'width': 4},
                              'thickness': 0.8, 'value': result['avg_prob']}
            }
        ))
        fig.update_layout(height=260, paper_bgcolor='rgba(0,0,0,0)',
                          font_color='#FAFAFA', margin=dict(t=20, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Model breakdown
        if model_choice != "XGBoost only":
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown(f'<p class="prob-label">🌲 Random Forest</p>'
                             f'<p class="prob-val" style="color:{risk_color("HIGH" if result["rf_prob"]>=60 else "MEDIUM" if result["rf_prob"]>=30 else "LOW")}">'
                             f'{result["rf_prob"]}%</p>', unsafe_allow_html=True)
            with mc2:
                st.markdown(f'<p class="prob-label">⚡ XGBoost</p>'
                             f'<p class="prob-val" style="color:{risk_color("HIGH" if result["xgb_prob"]>=60 else "MEDIUM" if result["xgb_prob"]>=30 else "LOW")}">'
                             f'{result["xgb_prob"]}%</p>', unsafe_allow_html=True)

        # Explanation
        st.write("")
        st.markdown("**🧠 Why this result?**")
        for flag in result['explanation']['flags']:
            st.markdown(f'<div class="flag-item">{flag}</div>', unsafe_allow_html=True)
        for clr in result['explanation']['clears']:
            st.markdown(f'<div class="clear-item">{clr}</div>', unsafe_allow_html=True)

        # Session history
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            'Amount': f"₹{amount:,.0f}", 'Hour': hour, 'Location': location,
            'Risk': f"{result['avg_prob']}%", 'Verdict': result['verdict']
        })

# ── Session history ───────────────────────────────────────────────────────────
if 'history' in st.session_state and st.session_state.history:
    st.divider()
    st.markdown("### 🕓 Session History")
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, use_container_width=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()