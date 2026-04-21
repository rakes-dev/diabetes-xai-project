# 🧠 Explainable AI for Healthcare: Interpretable Diabetes Risk Prediction using SHAP

## 📌 Overview

This project presents an **Explainable AI (XAI)-driven healthcare system** for predicting diabetes risk using machine learning, with a strong focus on **model interpretability and transparency**.

While traditional machine learning models often act as black boxes, this system integrates **SHAP (Shapley Additive Explanations)** to provide **feature-level insights**, enabling users to understand *why* a prediction was made.

> ⚠️ This project emphasizes **trustworthy AI in healthcare**, where interpretability is as important as prediction accuracy.

---

## 🎯 Research Motivation

In healthcare applications, model predictions must be **interpretable, reliable, and transparent**.

This project addresses:

* Lack of trust in black-box AI models
* Need for **explainability in medical decision support systems**
* Importance of understanding feature influence in predictions

---

## 🎯 Objectives

* Develop a machine learning model for diabetes prediction
* Integrate **Explainable AI techniques (SHAP)** for interpretability
* Provide **patient-level prediction explanations**
* Build an interactive system demonstrating **human-centered AI**

---

## 📊 Dataset

* **Pima Indians Diabetes Dataset**
* Features:

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
* Streamlit (Deployment Interface)

---

## 🤖 Methodology

### 🔹 Data Preprocessing

* Missing value handling (median imputation)
* Feature scaling using StandardScaler

### 🔹 Model Development

* Random Forest Classifier
* Train-test split and evaluation

### 🔹 Explainability Layer ⭐

* SHAP used to:

  * Quantify feature contributions
  * Interpret individual predictions
  * Provide local and global explanations

---

## 📈 Results

* Model Accuracy: **~76%**
* Key Influencing Features:

  * Glucose (highest impact)
  * BMI
  * Age

> 📌 SHAP analysis confirms domain-relevant insights, enhancing model credibility.

---

## 🧠 Explainable AI (SHAP)

This project goes beyond prediction by enabling:

* **Local Interpretability** → Why a specific patient is predicted diabetic
* **Global Interpretability** → Which features matter most overall

Example Insight:

> Elevated glucose levels significantly increase predicted diabetes risk, as validated through SHAP feature contributions.

---

## 🌐 Interactive Web Application

A **Streamlit-based interface** allows users to:

* Input patient health data
* Receive real-time predictions
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

## 🔬 Research Contribution

This project contributes by:

* Demonstrating **Explainable AI in healthcare systems**
* Bridging the gap between **prediction accuracy and interpretability**
* Providing a foundation for **trustworthy AI-based decision support systems**

---

## 🔮 Future Work

* Integration with real-world clinical datasets
* Deployment on cloud platforms
* Comparison with deep learning models
* Advanced explainability techniques beyond SHAP

---

## 👨‍💻 Author

**Rakesh Sardar**
MCA Final Year Student
Aspiring AI & Software Developer

---

## 📌 Conclusion

This project highlights the importance of **Explainable AI in sensitive domains like healthcare**, where understanding model decisions is critical for trust, adoption, and ethical AI deployment.
