import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Formulir Prediksi")

# Input manual fitur (misalnya dataset memiliki 2 fitur: x1 dan x2)
x1 = st.number_input("Masukkan nilai fitur 1 (x1):", value=0.0)
x2 = st.number_input("Masukkan nilai fitur 2 (x2):", value=0.0)

# Prediksi
if st.button("Prediksi"):
    df = pd.read_csv("data/dataset.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    model = RandomForestClassifier()
    model.fit(X, y)
    
    pred = model.predict(np.array([[x1, x2]]))
    st.success(f"✅ Hasil Prediksi: {pred[0]}")
