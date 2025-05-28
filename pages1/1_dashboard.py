import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Dataset dan Visualisasi")

# Load data
df = pd.read_csv("data/dataset.csv")
st.write("📄 **Dataset**")
st.dataframe(df)

# Statistik
st.write("📊 **Statistik Data**")
st.write(df.describe())

# Korelasi
st.write("🔍 **Heatmap Korelasi**")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
