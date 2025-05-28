import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("Pelatihan Model")

df = pd.read_csv("data/dataset.csv")

# Misal target kolomnya 'target'
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

st.write("🎯 **Akurasi Model**")
st.write(f"Akurasi: {model.score(X_test, y_test):.2f}")

st.write("📄 **Classification Report**")
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))
