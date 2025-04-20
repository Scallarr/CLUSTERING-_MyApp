# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 14:25:38 2025

@author: Kasidit
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("🌸 Iris Dataset - K-Means Clustering App")
st.markdown("""
This app performs **K-Means clustering** on the classic **Iris flower dataset**.
Use the slider to choose the number of clusters, and see the results in real time.
""")

# Load Iris dataset
iris = load_iris()
X = iris.data
features = iris.feature_names
df = pd.DataFrame(X, columns=features)

# Feature scaling
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# User input: number of clusters
k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(scaled_X)

# Add cluster info to DataFrame
df['Cluster'] = clusters

# Show data table with clusters
st.subheader("🔍 Data with Cluster Labels")
st.write(df)

# Visualize clusters (using first two features)
st.subheader("📊 Cluster Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(scaled_X[:, 0], scaled_X[:, 1], c=clusters, cmap='viridis', s=80)
ax.set_xlabel(features[0])
ax.set_ylabel(features[1])
ax.set_title(f"K = {k}")
st.pyplot(fig)

# Option to download
st.download_button(
    label="📥 Download Clustered Data",
    data=df.to_csv(index=False),
    file_name='iris_clustered.csv',
    mime='text/csv'
)


# Load from a saved dataset or generate synthetic data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

# Predict using the loaded model
y_kmeans = loaded_model.predict(X)
