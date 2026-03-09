import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import load_data, load_artifacts, predict_batch, risk_color
from src.data_generator import generate_transactions, LOCATIONS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card {
    background: #1C2333; border-radius: 12px; padding: 20px 24px;
    border-left: 4px solid; margin-bottom: 8px;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
  .kpi-label { font-size: 0.85rem; color: #8B9BB4; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
  .fraud-row  { color: #e74c3c !important; font-weight: 600; }
  .section-title { font-size: 1.1rem; font-weight: 600; color: #FAFAFA;
                   border-bottom: 1px solid #2C3E50; padding-bottom: 6px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.title("UPI Fraud Detection")
    st.caption("Big Data Capstone Project")
    st.divider()

    st.subheader("⚙️ Dataset Controls")
    n_rows    = st.slider("Transactions to generate", 1000, 50000, 10000, step=1000)
    fraud_pct = st.slider("Fraud percentage", 5, 30, 10) / 100

    if st.button("🔄 Generate New Dataset", use_container_width=True):
        with st.spinner("Generating..."):
            generate_transactions(n=n_rows, fraud_pct=fraud_pct)
            st.cache_data.clear()
        st.success(f"✅ Generated {n_rows:,} transactions!")
        st.rerun()

    st.divider()
    st.subheader("🔍 Filters")
    loc_filter  = st.multiselect("Location", options=LOCATIONS, default=LOCATIONS)
    time_filter = st.multiselect("Time of Day",
                                 options=['Morning','Afternoon','Evening','Night'],
                                 default=['Morning','Afternoon','Evening','Night'])

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_data():
    return load_data()

df_raw = get_data()

# Apply filters
df = df_raw.copy()
if loc_filter:
    df = df[df['location'].isin(loc_filter)]
if time_filter and 'time_of_day' in df.columns:
    df = df[df['time_of_day'].isin(time_filter)]

if df.empty:
    st.warning("No data matches your filters. Adjust the sidebar filters.")
    st.stop()

fraud_df = df[df['is_fraud'] == 1]
legit_df = df[df['is_fraud'] == 0]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("##  UPI Fraud Detection — Dashboard")
st.caption(f"Showing {len(df):,} transactions · Filters applied")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

def kpi(col, label, value, color, delta=None):
    col.markdown(f"""
    <div class="kpi-card" style="border-color:{color}">
      <p class="kpi-label">{label}</p>
      <p class="kpi-value" style="color:{color}">{value}</p>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Transactions", f"{len(df):,}", "#2E86C1")
kpi(k2, "Fraud Cases",        f"{len(fraud_df):,}", "#E74C3C")
kpi(k3, "Fraud Rate",         f"{len(fraud_df)/len(df)*100:.1f}%", "#E67E22")
kpi(k4, "Avg Fraud Amount",
    f"₹{fraud_df['amount'].mean():,.0f}" if len(fraud_df) else "—", "#8E44AD")

st.write("")

# ── Row 1: Pie + Hour bar ─────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown('<p class="section-title">📌 Fraud vs Legitimate Split</p>', unsafe_allow_html=True)
    pie_data = df['is_fraud'].value_counts().reset_index()
    pie_data.columns = ['Type', 'Count']
    pie_data['Type'] = pie_data['Type'].map({0: 'Legitimate', 1: 'Fraud'})
    fig = px.pie(pie_data, values='Count', names='Type',
                 color='Type',
                 color_discrete_map={'Legitimate': '#2ecc71', 'Fraud': '#e74c3c'},
                 hole=0.4)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', legend=dict(orientation='h', y=-0.1),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown('<p class="section-title">🕐 Fraud by Hour of Day</p>', unsafe_allow_html=True)
    hour_data = df.groupby(['hour_of_day', 'is_fraud']).size().reset_index(name='count')
    hour_data['Type'] = hour_data['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    fig = px.bar(hour_data, x='hour_of_day', y='count', color='Type', barmode='overlay',
                 color_discrete_map={'Legitimate': '#2ecc71', 'Fraud': '#e74c3c'},
                 labels={'hour_of_day': 'Hour', 'count': 'Transactions'}, opacity=0.8)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', legend=dict(orientation='h', y=-0.2),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Amount dist + Location ────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown('<p class="section-title">💰 Transaction Amount Distribution</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=legit_df['amount'].clip(upper=50000), name='Legitimate',
                               marker_color='#2ecc71', opacity=0.65, nbinsx=50))
    fig.add_trace(go.Histogram(x=fraud_df['amount'].clip(upper=200000), name='Fraud',
                               marker_color='#e74c3c', opacity=0.75, nbinsx=50))
    fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA',
                      xaxis_title='Amount (₹)', yaxis_title='Count',
                      legend=dict(orientation='h', y=-0.2), margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.markdown('<p class="section-title">📍 Fraud Count by Location</p>', unsafe_allow_html=True)
    loc_data = fraud_df['location'].value_counts().reset_index()
    loc_data.columns = ['Location', 'Fraud Count']
    fig = px.bar(loc_data, x='Fraud Count', y='Location', orientation='h',
                 color='Fraud Count', color_continuous_scale='Reds')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', coloraxis_showscale=False,
                      margin=dict(t=20, b=20), yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Device + Failed attempts ──────────────────────────────────────────
c5, c6 = st.columns(2)

with c5:
    st.markdown('<p class="section-title">📱 New Device Impact on Fraud</p>', unsafe_allow_html=True)
    dev = df.groupby(['is_new_device', 'is_fraud']).size().reset_index(name='count')
    dev['Device']  = dev['is_new_device'].map({0: 'Known Device', 1: 'New Device'})
    dev['Type']    = dev['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    fig = px.bar(dev, x='Device', y='count', color='Type', barmode='group',
                 color_discrete_map={'Legitimate': '#2ecc71', 'Fraud': '#e74c3c'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', legend=dict(orientation='h', y=-0.2),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with c6:
    st.markdown('<p class="section-title">⚠️ Failed Attempts vs Fraud Rate</p>', unsafe_allow_html=True)
    fail = df.groupby('failed_attempts').apply(
        lambda g: pd.Series({'fraud_rate': g['is_fraud'].mean() * 100, 'count': len(g)})
    ).reset_index()
    fig = px.bar(fail, x='failed_attempts', y='fraud_rate',
                 color='fraud_rate', color_continuous_scale='RdYlGn_r',
                 labels={'failed_attempts': 'Failed Attempts', 'fraud_rate': 'Fraud Rate (%)'},
                 text=fail['fraud_rate'].apply(lambda x: f"{x:.1f}%"))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', coloraxis_showscale=False,
                      margin=dict(t=20, b=20))
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# ── Recent Transactions Table ─────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">🧾 Recent Transactions</p>', unsafe_allow_html=True)

recent = df.tail(50)[['transaction_id', 'amount', 'hour_of_day', 'location',
                        'transaction_type', 'sender_bank', 'is_new_device',
                        'failed_attempts', 'is_fraud']].copy()
recent['amount']    = recent['amount'].apply(lambda x: f"₹{x:,.2f}")
recent['is_fraud']  = recent['is_fraud'].map({0: '✅ Legit', 1: '🚨 Fraud'})
recent['is_new_device'] = recent['is_new_device'].map({0: 'No', 1: 'Yes'})
recent.columns = ['ID', 'Amount', 'Hour', 'Location', 'Type', 'Bank',
                   'New Device', 'Failed Attempts', 'Status']

st.dataframe(
    recent.style.apply(
        lambda row: ['color: #e74c3c; font-weight: bold'
                     if row['Status'] == '🚨 Fraud' else '' for _ in row],
        axis=1
    ),
    use_container_width=True, height=400
)
