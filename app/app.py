import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("./model.pkl", "rb"))
scaler = pickle.load(open("./scaler.pkl", "rb"))

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.sidebar.title("About")
st.sidebar.info(
    "This app predicts diabetes risk using Machine Learning and explains predictions using SHAP."
)

st.title("🧠 Explainable AI - Diabetes Prediction")
st.write("Enter patient details to predict diabetes risk")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):

    input_data = pd.DataFrame([[
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({(1-probability)*100:.2f}%)")

    st.subheader("Prediction Confidence")
    st.progress(int(probability * 100))

    # ---------------- SHAP PART ----------------
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_scaled)

    st.subheader("Explainable AI - Feature Impact")

    # Use class 1 (diabetes)
    shap_val = shap_values[:, :, 1]

    # Create plot
    fig, ax = plt.subplots()
    shap.bar_plot(shap_val[0], feature_names=feature_names)
    st.pyplot(fig)