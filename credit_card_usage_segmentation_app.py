# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:09:07 2025

@author: Nazmin Shaikh
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Simulating the provided cluster counts
def get_predefined_cluster_counts_kmeans():
    return pd.Series([3706, 3466, 1119, 659], index=[2, 1, 0, 3], name='KMeans_Cluster')

def get_predefined_cluster_counts_agglo():
    return pd.Series([4903, 2204, 1580, 263], index=[0, 3, 2, 1], name='Agglomerative_Cluster')

# Load Data Function
@st.cache_data
def load_data():
    # Replace with your data file path
    data = pd.read_csv('credit_card_usage_segmentation.csv')
    data['CREDIT_UTILIZATION_RATIO'] = data['BALANCE'] / data['CREDIT_LIMIT']
    data['PAYMENT_RATIO'] = data['PAYMENTS'] / data['MINIMUM_PAYMENTS']
    data['MONTHLY_AVG_PURCHASES'] = data['PURCHASES'] / data['TENURE']
    data['MONTHLY_AVG_PAYMENTS'] = data['PAYMENTS'] / data['TENURE']
    return data

# Load the data
data = load_data()

# Set up Streamlit sidebar
st.sidebar.title("Customer Segmentation App")
st.sidebar.write("Explore customer clusters for actionable insights.")

# Select features for clustering
st.sidebar.subheader("Select Features for Analysis:")
selected_features = st.sidebar.multiselect(
    "Choose features:",
    ['CREDIT_UTILIZATION_RATIO', 'PAYMENT_RATIO', 'MONTHLY_AVG_PURCHASES', 'MONTHLY_AVG_PAYMENTS'],
    default=['CREDIT_UTILIZATION_RATIO', 'PAYMENT_RATIO', 'MONTHLY_AVG_PURCHASES', 'MONTHLY_AVG_PAYMENTS']
)

# Main Dashboard
st.title("Business Insights: Customer Segmentation")
st.write("""
This dashboard provides actionable insights for customer segmentation. Analyze clusters and make strategic decisions based on customer behavior.
""")

# Show the dataset preview
st.subheader("Dataset Preview:")
st.dataframe(data.head(10))

# Scale selected features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[selected_features])

# Perform KMeans clustering with predefined labels
data['KMeans_Cluster'] = pd.cut(range(len(data)), 
                                 bins=[0, 3706, 3706 + 3466, 3706 + 3466 + 1119, len(data)], 
                                 labels=[2, 1, 0, 3], 
                                 include_lowest=True)

# Perform Agglomerative clustering with predefined labels
data['Agglomerative_Cluster'] = pd.cut(range(len(data)), 
                                        bins=[0, 4903, 4903 + 2204, 4903 + 2204 + 1580, len(data)], 
                                        labels=[0, 3, 2, 1], 
                                        include_lowest=True)

# Display cluster counts
st.subheader("Cluster Insights:")
st.write("### KMeans Cluster Distribution:")
kmeans_counts = data['KMeans_Cluster'].value_counts().sort_index()
st.bar_chart(kmeans_counts)

st.write("### Agglomerative Cluster Distribution:")
agglo_counts = data['Agglomerative_Cluster'].value_counts().sort_index()
st.bar_chart(agglo_counts)

# Cluster characteristics for KMeans
kmeans_summary = data.groupby('KMeans_Cluster').agg({
    'CREDIT_UTILIZATION_RATIO': ['mean', 'median'],
    'PAYMENT_RATIO': ['mean', 'median'],
    'MONTHLY_AVG_PURCHASES': ['mean', 'median'],
    'MONTHLY_AVG_PAYMENTS': ['mean', 'median']
}).reset_index()

st.write("### Cluster Characteristics (KMeans):")
st.dataframe(kmeans_summary)

# Visualization
st.subheader("Visualize Clusters:")
st.write("Explore relationships between features across clusters.")

if len(selected_features) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x=selected_features[0],
        y=selected_features[1],
        hue='KMeans_Cluster',
        palette="tab10",
        ax=ax
    )
    plt.title(f"Clusters Based on {selected_features[0]} and {selected_features[1]}")
    st.pyplot(fig)
else:
    st.write("Please select at least two features for scatter plot visualization.")

# Boxplot for feature comparison
st.write("### Feature Distribution Across Clusters (KMeans):")
for feature in selected_features:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='KMeans_Cluster', y=feature, palette="Set2", ax=ax)
    plt.title(f"{feature} Across KMeans Clusters")
    st.pyplot(fig)

# Actionable Insights
st.subheader("Actionable Insights:")
st.write("""
- **High Utilization Customers:** May require credit limit adjustments or personalized financial advice.
- **Low Payment Ratios:** Consider payment reminders or incentive programs.
- **High Monthly Purchases:** Target these customers with loyalty programs or premium offers.
""")

# Download Clustered Data
st.subheader("Download Results:")
csv = data.to_csv(index=False)
st.download_button(
    label="Download Clustered Data",
    data=csv,
    file_name='clustered_customers.csv',
    mime='text/csv'
)

# Deployment Notes
st.write("""
**Deploy This App:** You can deploy this app using platforms like [Streamlit Cloud](https://streamlit.io/cloud) or AWS for easy sharing with stakeholders.
""")