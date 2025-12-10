import sys
import os

# Agregar la ruta raíz del proyecto al Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import joblib
from src.models.predict import predict_text

st.set_page_config(page_title="Mental Health Monitor", layout="wide")

st.title("🧠 Mental Health Monitoring – Fase 1 (MVP)")
st.write("Clasificación básica de ansiedad/estrés con NLP tradicional.")

user_input = st.text_area("Escribe un texto para analizar:")

if st.button("Analizar"):
    if user_input.strip():
        pred = predict_text(user_input)
        st.success(f"Predicción: **{pred}**")
    else:
        st.warning("Escribe un texto primero.")
