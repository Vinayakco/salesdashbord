import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .kpi-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #4CAF50;
    }
    .kpi-value { font-size: 28px; font-weight: bold; color: #1f1f1f; }
    .kpi-label { font-size: 14px; color: #6c6c6c; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    monthly = pd.read_csv('dashboard_data/monthly_sales.csv', parse_dates=['Order Date'])
    raw = pd.read_csv('dashboard_data/sales_raw.csv', parse_dates=['Order Date'])
    forecasts = pd.read_csv('dashboard_data/forecasts.csv')
    metrics = pd.read_csv('dashboard_data/model_metrics.csv')
    weekly = pd.read_csv('dashboard_data/weekly_sales.csv', parse_dates=['Order Date'])
    anomalies_iso = pd.read_csv('dashboard_data/anomalies_isolation.csv', parse_dates=['Order Date'])
    anomalies_z = pd.read_csv('dashboard_data/anomalies_zscore.csv', parse_dates=['Order Date'])
    clusters = pd.read_csv('dashboard_data/clusters.csv')
    pca_clusters = pd.read_csv('dashboard_data/pca_clusters.csv')
    return monthly, raw, forecasts, metrics, weekly, anomalies_iso, anomalies_z, clusters, pca_clusters

monthly_sales, sales_raw, forecasts, metrics, weekly, anomalies_iso, anomalies_z, clusters, pca_clusters = load_data()

st.title("📊 Retail Sales Analytics Dashboard")
st.caption("End-to-End Sales Forecasting & Demand Intelligence System")

st.sidebar.header("Filters")
region_filter = st.sidebar.multiselect("Region", sales_raw['Region'].unique(), default=list(sales_raw['Region'].unique()))
category_filter = st.sidebar.multiselect("Category", sales_raw['Category'].unique(), default=list(sales_raw['Category'].unique()))
filtered_raw = sales_raw[(sales_raw['Region'].isin(region_filter)) & (sales_raw['Category'].isin(category_filter))]

total_sales = filtered_raw['Sales'].sum()
total_orders = len(filtered_raw)
avg_order_value = filtered_raw['Sales'].mean()
best_model_mape = metrics['MAPE'].min()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">₹{total_sales:,.0f}</div><div class="kpi-label">Total Sales</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{total_orders:,}</div><div class="kpi-label">Total Orders</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">₹{avg_order_value:,.0f}</div><div class="kpi-label">Avg Order Value</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{best_model_mape:.1f}%</div><div class="kpi-label">Best Model MAPE</div></div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Overview", "🔮 Forecast Explorer", "⚠️ Anomaly Report", "📦 Product Segments"])

with tab1:
    filtered_raw['Year'] = filtered_raw['Order Date'].dt.year
    yearly = filtered_raw.groupby('Year')['Sales'].sum().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(yearly, x='Year', y='Sales', title="Total Sales by Year", color='Sales', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.line(monthly_sales, x='Order Date', y='Total_Sales', title="Monthly Sales Trend")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Sales by Region & Category")
    grouped = filtered_raw.groupby(['Region', 'Category'])['Sales'].sum().reset_index()
    fig3 = px.bar(grouped, x='Region', y='Sales', color='Category', barmode='group')
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    segment_choice = st.selectbox("Select Category or Region", forecasts['segment'].unique())
    horizon = st.slider("Forecast Horizon (months ahead)", 1, 3, 3)
    segment_data = forecasts[(forecasts['segment'] == segment_choice) & (forecasts['month_ahead'] <= horizon)]

    fig4 = px.line(segment_data, x='month_ahead', y='forecast', markers=True, title=f"Forecast for {segment_choice}")
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(segment_data, use_container_width=True)

    xgb_metrics = metrics[metrics['Model'] == 'XGBoost'].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("MAE (XGBoost - Best Model)", f"{xgb_metrics['MAE']:.2f}")
    c2.metric("RMSE (XGBoost - Best Model)", f"{xgb_metrics['RMSE']:.2f}")

with tab3:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=weekly['Order Date'], y=weekly['Total_Sales'], mode='lines', name='Weekly Sales'))
    fig5.add_trace(go.Scatter(x=anomalies_iso['Order Date'], y=anomalies_iso['Total_Sales'], mode='markers', name='Isolation Forest', marker=dict(color='red', size=10)))
    fig5.add_trace(go.Scatter(x=anomalies_z['Order Date'], y=anomalies_z['Total_Sales'], mode='markers', name='Z-Score', marker=dict(color='green', size=10, symbol='x')))
    fig5.update_layout(title="Weekly Sales with Detected Anomalies")
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Isolation Forest Anomalies")
        st.dataframe(anomalies_iso[['Order Date', 'Total_Sales']], use_container_width=True)
    with c2:
        st.subheader("Z-Score Anomalies")
        st.dataframe(anomalies_z[['Order Date', 'Total_Sales', 'z_score']], use_container_width=True)

with tab4:
    fig6 = px.scatter(pca_clusters, x='PC1', y='PC2', color=pca_clusters['cluster'].astype(str), text='Sub-Category', title="Product Sub-Category Clusters (PCA)")
    fig6.update_traces(textposition='top center')
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Sub-Categories by Cluster")
    st.dataframe(clusters[['Sub-Category', 'total_sales', 'growth_rate', 'volatility', 'cluster']].sort_values('cluster'), use_container_width=True)
