
import streamlit as st
import pandas as pd
import joblib

# Load the saved machine learning model
model = joblib.load("heart_disease_model.pkl")

st.title("Heart Disease Prediction")
st.write(
    "This academic machine learning application predicts "
    "whether heart disease may be present based on four input features."
)

st.warning(
    "Educational project only. This prediction is not a medical diagnosis."
)

# Get user inputs
age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
sex = st.selectbox(
    "Sex (0 = Female, 1 = Male)",
    options=[0, 1]
)
cp = st.selectbox(
    "Chest Pain Type (dataset codes 1–4)",
    options=[1, 2, 3, 4]
)
thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150
)

# Predict when the button is clicked
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("The model predicts: Heart Disease")
    else:
        st.success("The model predicts: No Heart Disease")
