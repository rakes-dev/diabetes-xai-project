import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("./model.pkl", "rb"))
scaler = pickle.load(open("./scaler.pkl", "rb"))

def generate_explanation(shap_values, feature_values, feature_names):
    explanation = []
    
    for i in range(len(feature_names)):
        value = shap_values[i]
        feature = feature_names[i]
        actual_val = feature_values[i]

        if value > 0:
            explanation.append(f"🔺 {feature} ({actual_val}) is increasing diabetes risk")
        else:
            explanation.append(f"🔻 {feature} ({actual_val}) is reducing diabetes risk")

    return explanation

def top_risk_factors(shap_values, feature_names, top_n=3):
    importance = []

    for i in range(len(feature_names)):
        val = float(shap_values[i])  # ensure scalar
        importance.append((feature_names[i], val))

    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return importance[:top_n]

def validate_inputs(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    errors = []

    if glucose < 70 or glucose > 300:
        errors.append("Glucose should be between 70–300")

    if bp < 40 or bp > 180:
        errors.append("Blood Pressure should be between 40–180")

    if bmi < 10 or bmi > 60:
        errors.append("BMI should be between 10–60")

    if age < 1 or age > 120:
        errors.append("Age should be between 1–120")

    if insulin < 0 or insulin > 900:
        errors.append("Insulin looks unrealistic")

    if skin < 0 or skin > 100:
        errors.append("Skin Thickness looks unrealistic")

    if dpf < 0 or dpf > 3:
        errors.append("Diabetes Pedigree Function should be 0–3")

    return errors

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

errors = validate_inputs(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age)

if errors:
    for err in errors:
        st.warning(err)
    st.stop()

# Predict button
if st.button("Predict"):

    input_data = pd.DataFrame([[
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # if prediction[0] == 1:
    #     st.error(f"⚠️ High Risk of Diabetes ({probability*100:.2f}%)")
    # else:
    #     st.success(f"✅ Low Risk of Diabetes ({(1-probability)*100:.2f}%)")

    st.subheader("Prediction Confidence")
    st.progress(int(probability * 100))

    # Convert to DataFrame (important!)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_scaled_df)

    # Extract SHAP values correctly
    shap_val = shap_values[0].values

    # If multi-dimensional (binary class case), pick class 1
    if len(shap_val.shape) > 1:
        shap_val = shap_val[:, 1]

    # -------- REPORT --------
    st.markdown("## 🏥 AI Medical Report")

    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"Low Risk of Diabetes ({(1-probability)*100:.2f}%)")

    # Top Factors
    st.markdown("### 🔍 Key Risk Factors")
    top_features = top_risk_factors(shap_val, feature_names)

    for name, val in top_features:
        direction = "⬆️ Increasing Risk" if val > 0 else "⬇️ Decreasing Risk"
        st.write(f"**{name}** → {direction}")

    # Explanation
    st.markdown("### 🧠 Explanation")
    explanations = generate_explanation(
        shap_val,
        input_data.iloc[0].values,
        feature_names
    )

    for exp in explanations[:5]:
        st.write(exp)

    # -------- SHAP VISUAL --------
    st.subheader("📊 Feature Impact Visualization")

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)

    st.pyplot(fig)

    # ---------------- SHAP PART ----------------
    # explainer = shap.TreeExplainer(model)

    # # Convert to DataFrame with column names
    # input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    # shap_values = explainer(input_scaled_df)

    # st.subheader("Explainable AI - Feature Impact")

    # fig = plt.figure()
    # shap.plots.waterfall(shap_values[0, :, 1], show=False)

    # st.pyplot(fig)