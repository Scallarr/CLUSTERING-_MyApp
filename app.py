# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 14:25:38 2025

@author: Kasidit
"""

# app.py
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 21:19:26 2025
@author: Nongnuch
"""

# app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set Streamlit page config
st.set_page_config(page_title="Iris k-Means Clustering", layout="centered")

# Title
st.title("🌸 Iris Dataset - k-Means Clustering App")

# Subheader
st.subheader("🔍 Clustering the Iris Dataset")

# Load the iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
df = pd.DataFrame(X, columns=feature_names)

# Feature Scaling
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Select number of clusters
k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)

# Train model or load
model = KMeans(n_clusters=k, random_state=42)
model.fit(scaled_X)
labels = model.predict(scaled_X)
centers = model.cluster_centers_

# Show dataframe with cluster labels
df['Cluster'] = labels
st.write(df)

# Plotting
st.subheader("📊 Cluster Visualization (2D)")
fig, ax = plt.subplots()
scatter = ax.scatter(scaled_X[:, 0], scaled_X[:, 1], c=labels, cmap='viridis', s=80)
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.6, label='Centroids')
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_title(f"k = {k}")
ax.legend()
st.pyplot(fig)

# Option to download results
st.download_button(
    label="📥 Download Clustered Data",
    data=df.to_csv(index=False),
    file_name='iris_clustered.csv',
    mime='text/csv'
)
