import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time

st.set_page_config(page_title="Trader Sentiment Analysis", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    sentiment = pd.read_csv("fear_greed_index.csv")
    trades = pd.read_csv("historical_data.csv")
    
    # Preprocessing
    sentiment['date'] = pd.to_datetime(sentiment['date'])
    trades['Timestamp IST'] = pd.to_datetime(
        trades['Timestamp IST'],
        dayfirst=True,
        errors='coerce'
    )
    trades['time'] = trades['Timestamp IST'].dt.time
    trades['trade_date'] = pd.to_datetime(trades['Timestamp IST'].dt.date)
    trades['Timestamp'] = pd.to_numeric(trades['Timestamp'], errors='coerce').astype('Int64')
    
    # Merge datasets
    df = pd.merge(
        trades,
        sentiment,
        left_on='trade_date',
        right_on='date',
        how='left'
    )
    
    # Drop duplicate 'date' column and original timestamp
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    if 'Timestamp IST' in df.columns:
        df = df.drop(columns=['Timestamp IST'])
        
    df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce').fillna(0)
    df['win'] = df['Closed PnL'] > 0
    
    return df

try:
    df_all = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("Dashboard Controls")

# Date Range Filter
min_date = df_all['trade_date'].min().date()
max_date = df_all['trade_date'].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Interval Selection
interval = st.sidebar.selectbox("Aggregation Level", ["Daily", "Weekly", "Monthly"], index=2)

# --- Filter Data ---
df = df_all.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['trade_date'].dt.date >= start_date) & (df['trade_date'].dt.date <= end_date)]

# --- Metrics Calculation ---
total_pnl = df['Closed PnL'].sum()
win_rate = df['win'].mean() * 100 if len(df) > 0 else 0
total_trades = len(df)
avg_sentiment = df['value'].mean()

# --- Dashboard Layout ---
st.title("📊 Trader Sentiment Analysis Dashboard")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total PnL", f"${total_pnl:,.2f}")
with col2:
    st.metric("Win Rate", f"{win_rate:.1f}%")
with col3:
    st.metric("Total Trades", f"{total_trades:,}")
with col4:
    st.metric("Avg Fear & Greed", f"{avg_sentiment:.1f}")

st.divider()

# --- Visualizations ---

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader(f"Performance Overview ({interval})")
    if interval == "Daily":
        pnl_data = df.groupby('trade_date')['Closed PnL'].sum().reset_index()
        fig_pnl = px.line(pnl_data, x='trade_date', y='Closed PnL', title="Daily PnL Over Time")
    elif interval == "Weekly":
        pnl_data = df.groupby(df['trade_date'].dt.to_period('W').astype(str))['Closed PnL'].sum().reset_index()
        pnl_data.columns = ['Week', 'Closed PnL']
        fig_pnl = px.bar(pnl_data, x='Week', y='Closed PnL', title="Weekly Aggregated PnL", color='Closed PnL', color_continuous_scale='RdYlGn')
    else: # Monthly
        pnl_data = df.groupby(df['trade_date'].dt.to_period('M').astype(str))['Closed PnL'].sum().reset_index()
        pnl_data.columns = ['Month', 'Closed PnL']
        fig_pnl = px.bar(pnl_data, x='Month', y='Closed PnL', title="Monthly Aggregated PnL", color='Closed PnL', color_continuous_scale='RdYlGn')
    
    st.plotly_chart(fig_pnl, use_container_width=True)

with row1_col2:
    st.subheader("Market Sentiment Correlation")
    sent_pnl = df.groupby('trade_date').agg({'Closed PnL': 'sum', 'value': 'mean', 'win': 'mean'}).reset_index()
    fig_sent = px.scatter(sent_pnl, x='value', y='Closed PnL', color='win',
                         hover_data=['trade_date'],
                         labels={'value': 'Fear & Greed Index', 'Closed PnL': 'Daily PnL', 'win': 'Win Ratio'},
                         title="PnL vs. Fear & Greed Index")
    st.plotly_chart(fig_sent, use_container_width=True)

st.divider()

# --- End of Dashboard Analysis Options ---
st.header("🔍 Deep Dive Analysis")

analysis_tab = st.radio("Select Analysis Metric", ["Win Rate Analysis", "Profit & Loss Details", "Sentiment Insights"], horizontal=True)

if analysis_tab == "Win Rate Analysis":
    st.subheader("Win vs Loss Distribution")
    win_loss = df['win'].value_counts().reset_index()
    win_loss.columns = ['Result', 'Count']
    win_loss['Result'] = win_loss['Result'].apply(lambda x: "Win" if x else "Loss")
    fig_wl = px.pie(win_loss, values='Count', names='Result', color='Result',
                   color_discrete_map={'Win':'#2ecc71', 'Loss':'#e74c3c'}, hole=0.4)
    st.plotly_chart(fig_wl)
    
elif analysis_tab == "Profit & Loss Details":
    st.subheader("PnL Distribution per Trade")
    fig_dist = px.histogram(df, x='Closed PnL', nbins=50, title="Trade PnL Distribution",
                           color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.write("Summary Statistics:")
    st.write(df['Closed PnL'].describe())

elif analysis_tab == "Sentiment Insights":
    st.subheader("PnL across Sentiment Classifications")
    if 'classification' in df.columns:
        sent_class = df.groupby('classification')['Closed PnL'].agg(['sum', 'count', 'mean']).reset_index()
        fig_class = px.bar(sent_class, x='classification', y='sum', color='mean',
                          title="Total PnL by Market Sentiment Type",
                          labels={'sum': 'Total PnL', 'mean': 'Avg PnL per Trade'})
        st.plotly_chart(fig_class, use_container_width=True)
    else:
        st.warning("Sentiment classification data not available for the selected range.")

# --- Raw Data Table (Truncated to avoid error) ---
with st.expander("View Underlying Data (Sample: Top 500 rows)"):
    st.dataframe(df.head(500).style.format(subset=['Size USD', 'Closed PnL', 'Execution Price'], formatter="{:.2f}"))

st.markdown("---")
st.caption("Dashboard updated with deep dive analysis for Win Rate, PnL, and Sentiment.")
