import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data
from src.data_generator import LOCATIONS, TRANSACTION_TYPES, BANKS

st.set_page_config(page_title="Data Explorer | UPI Fraud", page_icon="📊", layout="wide")

st.markdown("""
<style>
  .section-title { font-size:1.05rem; font-weight:600; color:#FAFAFA;
                   border-bottom:1px solid #2C3E50; padding-bottom:6px; margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Data Explorer")
st.caption("Filter, browse and analyse transaction data interactively")
st.divider()

@st.cache_data(ttl=60)
def get_data():
    return load_data()

df = get_data()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔍 Filter Transactions")

    status = st.radio("Show", ["All", "Fraud only", "Legitimate only"], horizontal=False)

    amount_min, amount_max = float(df['amount'].min()), float(df['amount'].max())
    amount_range = st.slider("Amount Range (₹)",
                              amount_min, min(amount_max, 200000.0),
                              (amount_min, min(amount_max, 200000.0)))

    hour_range = st.slider("Hour of Day", 0, 23, (0, 23))

    loc_sel  = st.multiselect("Location",         LOCATIONS,         default=LOCATIONS)
    type_sel = st.multiselect("Transaction Type", TRANSACTION_TYPES, default=TRANSACTION_TYPES)
    bank_sel = st.multiselect("Sender Bank",      BANKS,             default=BANKS)
    device   = st.radio("Device", ["All", "New Device only", "Known Device only"])

# ── Apply filters ────────────────────────────────────────────────────────────
filt = df.copy()

if status == "Fraud only":
    filt = filt[filt['is_fraud'] == 1]
elif status == "Legitimate only":
    filt = filt[filt['is_fraud'] == 0]

filt = filt[
    (filt['amount']       >= amount_range[0]) &
    (filt['amount']       <= amount_range[1]) &
    (filt['hour_of_day']  >= hour_range[0])   &
    (filt['hour_of_day']  <= hour_range[1])   &
    (filt['location'].isin(loc_sel))           &
    (filt['transaction_type'].isin(type_sel))  &
    (filt['sender_bank'].isin(bank_sel))
]
if device == "New Device only":
    filt = filt[filt['is_new_device'] == 1]
elif device == "Known Device only":
    filt = filt[filt['is_new_device'] == 0]

if filt.empty:
    st.warning("No transactions match your filters.")
    st.stop()

# ── Summary stats ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Filtered Rows",   f"{len(filt):,}")
k2.metric("Fraud Count",     f"{filt['is_fraud'].sum():,}")
k3.metric("Fraud Rate",      f"{filt['is_fraud'].mean()*100:.1f}%")
k4.metric("Avg Amount",      f"₹{filt['amount'].mean():,.0f}")
k5.metric("Max Amount",      f"₹{filt['amount'].max():,.0f}")

st.write("")

# ── Download button ───────────────────────────────────────────────────────────
csv_bytes = filt.to_csv(index=False).encode()
st.download_button("⬇️ Download Filtered Data as CSV", csv_bytes,
                    file_name="filtered_transactions.csv", mime="text/csv")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown('<p class="section-title">📈 Fraud Rate by Transaction Type</p>', unsafe_allow_html=True)
    g = filt.groupby('transaction_type')['is_fraud'].agg(['mean','sum','count']).reset_index()
    g.columns = ['Type', 'Fraud Rate', 'Fraud Count', 'Total']
    g['Fraud Rate %'] = (g['Fraud Rate'] * 100).round(1)
    fig = px.bar(g, x='Type', y='Fraud Rate %', color='Fraud Rate %',
                 color_continuous_scale='Reds', text=g['Fraud Rate %'].apply(lambda x: f"{x}%"))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', coloraxis_showscale=False, margin=dict(t=20,b=20))
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown('<p class="section-title">🏛️ Fraud Rate by Bank</p>', unsafe_allow_html=True)
    b = filt.groupby('sender_bank')['is_fraud'].agg(['mean','count']).reset_index()
    b.columns = ['Bank', 'Fraud Rate', 'Count']
    b['Fraud Rate %'] = (b['Fraud Rate'] * 100).round(1)
    fig = px.bar(b, x='Bank', y='Fraud Rate %', color='Fraud Rate %',
                 color_continuous_scale='OrRd', text=b['Fraud Rate %'].apply(lambda x: f"{x}%"))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', coloraxis_showscale=False, margin=dict(t=20,b=20))
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.markdown('<p class="section-title">🌍 Fraud Heatmap: Hour × Location</p>', unsafe_allow_html=True)
    heat = filt[filt['is_fraud']==1].groupby(
        ['location','hour_of_day']).size().reset_index(name='count')
    heat_piv = heat.pivot(index='location', columns='hour_of_day', values='count').fillna(0)
    fig = px.imshow(heat_piv, color_continuous_scale='Reds',
                    labels=dict(x='Hour of Day', y='Location', color='Fraud Count'))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#FAFAFA', margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.markdown('<p class="section-title">💰 Fraud Amount Distribution</p>', unsafe_allow_html=True)
    fig = px.box(filt, x=filt['is_fraud'].map({0:'Legitimate',1:'Fraud'}),
                 y='amount', color=filt['is_fraud'].map({0:'Legitimate',1:'Fraud'}),
                 color_discrete_map={'Legitimate':'#2ecc71','Fraud':'#e74c3c'},
                 log_y=True, labels={'x':'Type','y':'Amount (₹, log scale)'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#FAFAFA', showlegend=False, margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Data Table ────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">🗂️ Transaction Data</p>', unsafe_allow_html=True)

page_size = 50
total_pages = max(1, (len(filt) - 1) // page_size + 1)
page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

display = filt.iloc[page * page_size: (page + 1) * page_size].copy()
display['amount']        = display['amount'].apply(lambda x: f"₹{x:,.2f}")
display['is_fraud']      = display['is_fraud'].map({0: '✅ Legit', 1: '🚨 Fraud'})
display['is_new_device'] = display['is_new_device'].map({0: 'No', 1: 'Yes'})

st.dataframe(
    display.style.apply(
        lambda row: ['color:#e74c3c;font-weight:600'
                     if row.get('is_fraud') == '🚨 Fraud' else '' for _ in row], axis=1
    ),
    use_container_width=True, height=450
)
st.caption(f"Page {page+1} of {total_pages} · {len(filt):,} total rows")
