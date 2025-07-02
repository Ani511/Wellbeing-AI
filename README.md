# 💚 Wellbeing AI — Predictive Health Risk Scoring with Explainable ML & Empathetic Recommendations

> 🧠 Using CDC’s BRFSS Dataset · XGBoost · LIME/ELI5 · Streamlit · Power BI  
> 📅 Project Date: May 2025 &nbsp;&nbsp;&nbsp;&nbsp;🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)

---

## 🩺 Problem Statement

Preventive healthcare in underserved and vulnerable populations is often hindered by the lack of explainable risk assessments based on **Social Determinants of Health (SDOH)**. Traditional models ignore non-clinical data such as mental health, socioeconomic status, lifestyle, or access to care.

This project aims to build an AI-powered tool that uses publicly available health survey data to:

- Predict an individual's risk of developing **chronic conditions**
- Provide **interpretable insights** behind the prediction
- Suggest **empathetic, actionable health guidance**
- Support **population-level policy decisions** through BI dashboards

---

## 🌍 Real-World Impact

- 🎯 **Personalized Prevention**: Deliver risk scores to individuals before clinical symptoms appear
- 👥 **Equity-Oriented**: Account for social determinants often overlooked in traditional models
- 🏥 **Supports Public Health Agencies**: Visualize clusters of at-risk populations to guide interventions
- 🤖 **Human-Centered AI**: Transparent, interpretable models reduce “black-box” distrust

---

## 📚 Dataset

- 📦 **Source**: [CDC Behavioral Risk Factor Surveillance System (BRFSS)](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)
- 🧮 **Size**: 400,000+ anonymized survey records (2020–2022)
- 🧠 **Features**: 300+ variables on physical & mental health, lifestyle, access to care, geography, and demographics

---

## 🧾 Project Summary

We built an end-to-end ML system using BRFSS data to predict chronic disease risk and explain predictions using human-readable methods.

### 🎯 Goals:
- Clean, structure, and model the BRFSS dataset
- Predict whether a person is at risk for chronic conditions like diabetes, heart disease, etc.
- Use **explainable AI** to break down *why* that risk exists
- Generate **natural language health recommendations**
- Visualize high-risk groups using **clustering** and **Power BI**

---

## ✅ Results

- Achieved **balanced accuracy >80%** on chronic disease prediction using calibrated XGBoost
- Successfully identified and visualized **key social determinants** contributing to risk (e.g., exercise, depression, income, healthcare access)
- Delivered **interpretable explanations per individual** using LIME & ELI5
- Built a working prototype of a **Streamlit app** for personal health feedback
- Designed **Power BI dashboards** to visualize risk across different regions and demographics

---

## 🔑 Key Features

- 🔍 **Risk Prediction**: Binary classification for chronic disease presence using XGBoost
- ⚖️ **Class Imbalance Handling**: Resampling and calibration to account for underrepresented outcomes
- 💬 **Explainability Layer**: LIME & ELI5 breakdown of prediction factors
- 🤖 **LLM Recommender**: Natural language health tips based on risk and feature influence
- 📊 **Power BI Dashboards**: Public health-level visual insights on SDOH clustering
- 🧪 **Fully Modular**: Clean separation of EDA, modeling, explainability, and dashboarding notebooks

---

## 💡 Example Use Cases

- 👨‍⚕️ A family doctor gives a patient a risk score with clear reasoning: “You may be at higher risk for diabetes because of recent inactivity, BMI, and mental health scores.”
- 🏥 A public health department clusters at-risk individuals by state and income level to optimize outreach
- 🧠 A health policy researcher investigates which lifestyle factors most influence risk for women under 40

---

## 🧠 Technical Stack

| Area                  | Tools Used                                             |
|-----------------------|--------------------------------------------------------|
| **ML & Modeling**     | `XGBoost`, `Scikit-learn`, `CalibratedClassifierCV`   |
| **Explainability**    | `ELI5`, `LIME`, `SHAP (optional)`                      |
| **Frontend**          | `Streamlit`                                           |
| **Visualization & BI**| `Power BI`, `Seaborn`, `Plotly`                        |
| **Data**              | BRFSS (CDC) 2020–2022 – ~400K records, 300+ features  |

---

## 🗂️ Project Structure

- `EDA_Preprocessing.ipynb` — Data cleaning, feature selection, correlation maps
- `Modelling_Healthscore.ipynb` — XGBoost training, calibration, evaluation
- `mExplainability.ipynb` — ELI5 + LIME interpretability for predictions
- `LLM_Recommeder.ipynb` — Personalized, empathetic text suggestions
- `outputs ` - Contains final model outputs including ELI5 and LIME explanations, top feature summaries, a batch of personalized health recommendations, and the calibrated XGBoost pipeline.  
  *(Note: The final `risk_scores_with_recc.csv` file is not uploaded due to size limits.)*
- `README.md` — Project overview (this file)

---

## ⚠️ Note on Dataset

Due to GitHub’s file size constraints, the original dataset is not included.

You can download it from Kaggle:  
🔗 [CDC BRFSS Dataset](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system)

---

## 🧾 Input & Output

### 🔹 Input:
- Raw `.csv` files from CDC BRFSS survey (2020–2022)
- Features include: age, gender, BMI, physical activity, depression, access to healthcare, etc.

### 🔹 Output:
- Binary prediction: `"At Risk"` / `"Not At Risk"` for chronic conditions
- Explainable feature contributions (e.g., `"Low physical activity: +0.12 risk"`)
- Natural-language recommendation (LLM output):  
  > “To reduce your risk, consider increasing physical activity and regular check-ups…”

---

## 🛠️ How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Ani511/Wellbeing-AI.git
   cd Wellbeing-AI

2. Install dependencies:
    pip install -r requirements.txt
    Download & add BRFSS CSVs into a data/ folder

3. Open & run notebooks:
- jupyter notebook modeling/Modelling_Healthscore.ipynb
- run app.py
- streamlit run app.py

---
### 📊 Power BI Dashboards
To complement the ML pipeline, we designed Power BI dashboards using cleaned BRFSS data to help visualize:

- 📍 SDOH patterns across states and age groups
- 💡 Distribution of key risk factors (e.g., physical inactivity, depression, obesity)
- 🧑‍🤝‍🧑 Clustering of at-risk populations by income, access to care, and lifestyle
- 🏥 Potential intervention zones for preventive healthcare outreach

These dashboards support public health researchers and policymakers in identifying vulnerable groups and targeting data-driven wellness campaigns.

⚠️ Due to size limits and platform restrictions, .pbix files are not included in the repository but are available upon request.

--- 
### 🧠 Highlights & Conclusion
✅ Human-centered risk scoring from publicly available data
✅ Modular notebook structure and clear code flow
✅ Strong emphasis on interpretability and health empathy
✅ Scalable for use in mobile health apps or public policy systems

This project demonstrates how responsible ML can support both individual well-being and large-scale health strategy.
