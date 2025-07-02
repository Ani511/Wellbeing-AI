import streamlit as st
st.set_page_config(page_title="Health Risk Predictor", layout="centered")
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular
import os

# ---------------------- # 1. Load Model + Setup
@st.cache_resource
def load_assets():
    model = joblib.load("outputs/xgb_calibrated_pipeline.pkl")
    sample = pd.read_csv("data/lime_clean_sample.csv")
    
    cat_cols = ['SEX', 'EDUCA', 'INCOME2', 'EMPLOY1', 'MARITAL', 'GENHLTH']
    num_cols = ['AGE', '_SMOKER3', 'YEAR']
    all_cols = num_cols + cat_cols

    # Ensure correct dtypes
    sample = sample[all_cols].copy()
    sample[cat_cols] = sample[cat_cols].astype("category")

    # Fit encoder manually (fix for get_feature_names_out)
    encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    encoder.fit(sample[cat_cols])

    # Transform for LIME
    transformed = model.named_steps['preprocessor'].transform(sample)
    onehot_features = encoder.get_feature_names_out(cat_cols)
    feature_names = num_cols + list(onehot_features)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=transformed,
        feature_names=feature_names,
        class_names=["Low Risk", "High Risk"],
        mode="classification"
    )

    return model, cat_cols, num_cols, all_cols, explainer, feature_names

model, cat_cols, num_cols, all_cols, explainer, feature_names = load_assets()

# ---------------------- # 2. Streamlit App Layout
st.title("🧠 Predict Your Health Risk Score")
st.markdown("Enter your details to get your **risk score**, explanation, and a warm, human recommendation.")

# ---------------------- # 3. Input Form
with st.form("input_form"):
    col1, col2 = st.columns(2)

    SEX = col1.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    AGE = col2.slider("Age", 18, 80, 30)

    EDUCA = col1.selectbox("Education Level", list(range(1, 7)), format_func=lambda x: [
        "Never attended school", "Grades 1–8", "Grades 9–11", "High School Grad", "Some College", "College Grad"
    ][x - 1])

    INCOME2 = col2.selectbox("Income Level", list(range(1, 9)), format_func=lambda x: f"Level {x}")

    EMPLOY1 = col1.selectbox("Employment Status", list(range(1, 9)), format_func=lambda x: [
        "Employed for wages", "Self-employed", "Out of work (< 1 year)", "Out of work (≥ 1 year)", "Homemaker", "Student", "Retired", "Unable to work"
    ][x - 1])
    MARITAL = col2.selectbox("Marital Status", list(range(1, 7)), format_func=lambda x: [
        "Married", "Divorced", "Widowed", "Separated", "Never married", "In a realtionship"
    ][x - 1])

    GENHLTH = col1.selectbox("Self-Rated General Health", list(range(1, 6)), format_func=lambda x: [
        "Excellent", "Very Good", "Good", "Fair", "Poor"
    ][x - 1])

    _SMOKER3 = col2.selectbox("Smoking Status", list(range(1, 5)), format_func=lambda x: [
        "Former Smoker", "Never Smoked", "Chain Smoker", "Smoke Sometimes"][x-1])
    YEAR = col1.selectbox("Survey Year", [2020, 2021, 2022])

    submit = st.form_submit_button("🔍 Predict Risk Score")

# ---------------------- # 4. Prediction + Explainability
if submit:
    input_df = pd.DataFrame([{
        'SEX': SEX, 'AGE': AGE, 'EDUCA': EDUCA, 'INCOME2': INCOME2,
        'EMPLOY1': EMPLOY1, 'MARITAL': MARITAL, 'GENHLTH': GENHLTH,
        '_SMOKER3': _SMOKER3, 'YEAR': YEAR
    }])
    input_df[cat_cols] = input_df[cat_cols].astype("category")

    prob = model.predict_proba(input_df)[0][1]
    score = round(prob * 100, 2)

    st.subheader("📊 Your Predicted Risk Score")
    st.metric("Health Risk Score", f"{score} / 100")

    # Explain with LIME
    input_transformed = model.named_steps['preprocessor'].transform(input_df)
    explanation = explainer.explain_instance(input_transformed[0], model.named_steps['classifier'].predict_proba, num_features=5)

    st.subheader("🧠 Top Risk Drivers (via LIME)")
    for feat, val in explanation.as_list():
        st.markdown(f"- **{feat}**  —  `{val:.2f}`")

    # ---------------------- # 5. Recommendation
    st.subheader("💬 Warm Personalized Advice")
    # Get base feature names from LIME explanation
    drivers = [f.split('=')[0].strip().lower() for f, _ in explanation.as_list()]
    advice = []

    drivers = [f.split('=')[0].strip().lower() for f, _ in explanation.as_list()]
    advice = []

    if "bmi" in drivers:
        advice.append("Your body does so much for you every day. Let’s gently care for it through nourishing meals and joyful movement, at your own pace.")
    if "totinda" in drivers or "inactive" in drivers:
        advice.append("Even a 10-minute walk or stretch session can uplift your energy and spirit. Movement doesn’t have to be intense to be healing.")
    if "diabete" in drivers:
        advice.append("Blood sugar management is a journey. Consider small steps—fiber-rich meals, fewer sugary drinks, and check-ins with your doctor.")
    if "income2" in drivers:
        advice.append("You’re doing your best in your circumstances. Explore free or affordable health services around you—support is meant for this.")
    if "educa" in drivers:
        advice.append("Curiosity is power. Even a short video or local session on health can unlock new choices for your wellbeing.")
    if "_smoker3" in drivers:
        advice.append("There’s no shame in your journey. If you're considering reducing smoking, even small steps make a huge difference.")
    if "genhlth" in drivers:
        advice.append("Your wellbeing matters. Rest, hydrate, and try to include one thing daily that brings you peace.")
    if "employ1" in drivers:
        advice.append("Work stress is real. Even a short daily ritual just for you—music, silence, tea—can recharge your inner self.")
    if "age" in drivers and AGE <= 25:
        advice.append("Starting young is a gift to your future self. What you’re doing now will echo beautifully in years to come.")
    if score > 80:
        advice.append("This score suggests you're at higher risk. It’s okay to feel concerned—consider a health check-up and self-kindness.")
    elif score < 25:
        advice.append("You're doing beautifully. Keep honoring your body and mind—they thank you every day.")
    if not advice:
        advice.append("You’re on the right path. Stay consistent, stay kind to yourself—you’re doing better than you think.")

    st.markdown("> " + " ".join(advice))
