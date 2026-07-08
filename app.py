import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    monthly = pd.read_csv('dashboard_data/monthly_sales.csv', parse_dates=['Order Date'])
    raw = pd.read_csv('dashboard_data/sales_raw.csv', parse_dates=['Order Date'])
    return monthly, raw

monthly_sales, sales_raw = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"])

if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    sales_raw['Year'] = sales_raw['Order Date'].dt.year
    yearly = sales_raw.groupby('Year')['Sales'].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        fig, ax = plt.subplots()
        ax.bar(yearly['Year'].astype(str), yearly['Sales'])
        st.pyplot(fig)
    with col2:
        st.subheader("Monthly Sales Trend")
        fig2, ax2 = plt.subplots()
        ax2.plot(monthly_sales['Order Date'], monthly_sales['Total_Sales'])
        st.pyplot(fig2)

    st.subheader("Sales by Region & Category (Filterable)")
    region_filter = st.multiselect("Region", sales_raw['Region'].unique(), default=list(sales_raw['Region'].unique()))
    category_filter = st.multiselect("Category", sales_raw['Category'].unique(), default=list(sales_raw['Category'].unique()))
    filtered = sales_raw[(sales_raw['Region'].isin(region_filter)) & (sales_raw['Category'].isin(category_filter))]
    st.dataframe(filtered.groupby(['Region','Category'])['Sales'].sum().reset_index())

elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    forecasts = pd.read_csv('dashboard_data/forecasts.csv')
    metrics = pd.read_csv('dashboard_data/model_metrics.csv')

    segment_choice = st.selectbox("Select Category or Region", forecasts['segment'].unique())
    horizon = st.slider("Forecast Horizon (months ahead)", 1, 3, 3)

    segment_data = forecasts[(forecasts['segment'] == segment_choice) & (forecasts['month_ahead'] <= horizon)]

    st.subheader(f"Forecast for {segment_choice} — Next {horizon} month(s)")
    fig, ax = plt.subplots()
    ax.plot(segment_data['month_ahead'], segment_data['forecast'], marker='o', color='green')
    ax.set_xlabel("Month Ahead")
    ax.set_ylabel("Forecasted Sales")
    st.pyplot(fig)

    st.dataframe(segment_data)

    xgb_metrics = metrics[metrics['Model'] == 'XGBoost'].iloc[0]
    col1, col2 = st.columns(2)
    col1.metric("MAE (XGBoost - Best Model)", f"{xgb_metrics['MAE']:.2f}")
    col2.metric("RMSE (XGBoost - Best Model)", f"{xgb_metrics['RMSE']:.2f}")

elif page == "Anomaly Report":
    st.title("⚠️ Anomaly Report")

    weekly = pd.read_csv('dashboard_data/weekly_sales.csv', parse_dates=['Order Date'])
    anomalies_iso = pd.read_csv('dashboard_data/anomalies_isolation.csv', parse_dates=['Order Date'])
    anomalies_z = pd.read_csv('dashboard_data/anomalies_zscore.csv', parse_dates=['Order Date'])

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(weekly['Order Date'], weekly['Total_Sales'], label='Weekly Sales', color='steelblue', alpha=0.7)
    ax.scatter(anomalies_iso['Order Date'], anomalies_iso['Total_Sales'], color='red', label='Isolation Forest', s=80, zorder=5)
    ax.scatter(anomalies_z['Order Date'], anomalies_z['Total_Sales'], color='green', label='Z-Score', marker='x', s=80, zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected Anomalies — Isolation Forest")
    st.dataframe(anomalies_iso[['Order Date', 'Total_Sales']])

    st.subheader("Detected Anomalies — Z-Score")
    st.dataframe(anomalies_z[['Order Date', 'Total_Sales', 'z_score']])

elif page == "Product Segments":
    st.title("📦 Product Demand Segments")

    clusters = pd.read_csv('dashboard_data/clusters.csv')
    pca_clusters = pd.read_csv('dashboard_data/pca_clusters.csv')

    fig, ax = plt.subplots(figsize=(9,6))
    scatter = ax.scatter(pca_clusters['PC1'], pca_clusters['PC2'], c=pca_clusters['cluster'], cmap='viridis', s=100)
    for i, row in pca_clusters.iterrows():
        ax.annotate(row['Sub-Category'], (row['PC1'], row['PC2']), fontsize=8, xytext=(5,5), textcoords='offset points')
    st.pyplot(fig)

    st.subheader("Sub-Categories by Cluster")
    st.dataframe(clusters[['Sub-Category', 'total_sales', 'growth_rate', 'volatility', 'cluster']].sort_values('cluster'))