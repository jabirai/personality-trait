# streamlit_app.py
import streamlit as st
import requests
import plotly.graph_objs as go

st.title("HVP | Personality Trait Prediction")
user_input = st.text_input("Enter a text here:")

if user_input:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict", json={"input_text": user_input})
        response.raise_for_status()
        result = response.json()

        trait_scores = result["traits"]
        y_out = result["raw_output"]

        st.write("User input:\n\n", f"{user_input}")

        fig = go.Figure(data=go.Scatter(x=list(trait_scores.keys()), y=list(
            trait_scores.values()), mode='lines+markers', name='Line Plot'))
        st.plotly_chart(fig)

        st.write("HVP Scores:\n", trait_scores)

    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
