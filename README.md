# 🧠 Explainable AI for Diabetes Prediction

## 📌 Overview

This project is an **Explainable AI-based healthcare system** that predicts the likelihood of diabetes using Machine Learning and provides **interpretable insights** using SHAP (SHapley Additive exPlanations).

Unlike traditional black-box models, this system not only predicts outcomes but also explains **why** a prediction was made, improving transparency and trust in AI-based medical systems.

---

## 🎯 Objectives

* Predict diabetes using patient health data
* Provide **feature-level explanations** using SHAP
* Build a simple **interactive web application** using Streamlit
* Demonstrate real-world application of Explainable AI in healthcare

---

## 📊 Dataset

* **Pima Indians Diabetes Dataset**
* Features include:

  * Glucose
  * BMI
  * Age
  * Blood Pressure
  * Insulin
  * Skin Thickness
  * Diabetes Pedigree Function

---

## ⚙️ Technologies Used

* Python 🐍
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn
* SHAP (Explainable AI)
* Streamlit (Web App)

---

## 🤖 Machine Learning Model

* Random Forest Classifier
* Data Preprocessing:

  * Handling missing values
  * Feature scaling using StandardScaler

---

## 📈 Results

* Model Accuracy: **~76%**
* Key Influencing Features:

  * Glucose (highest impact)
  * BMI
  * Age

---

## 🧠 Explainable AI (SHAP)

SHAP is used to:

* Interpret individual predictions
* Identify feature contributions
* Improve model transparency

Example:

> Higher glucose levels significantly increase the likelihood of diabetes prediction.

---

## 🌐 Web Application

A Streamlit-based UI allows users to:

* Input patient data
* Get real-time predictions
* View prediction confidence
* Understand feature-level impact

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/diabetes-xai-project.git
cd diabetes-xai-project
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
diabetes-xai-project/
│
├── data/
├── notebooks/
├── app/
│   └── app.py
├── model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
```

---

## 🔮 Future Improvements

* Integrate real-time medical datasets
* Deploy using cloud platforms (AWS / GCP)
* Improve model accuracy using deep learning
* Add advanced SHAP visualizations

---

## 👨‍💻 Author

**Rakesh Sardar**
MCA Final Year Student
AI & Software Developer

---

## 📌 Conclusion

This project demonstrates how Explainable AI can be applied in healthcare to build **trustworthy and interpretable machine learning systems**, bridging the gap between AI models and real-world applications.
