# ---------------------------------------------------------------------------
# Streamlit HTN Risk Stratification Tool (calibrated LASSO, two model variants)
# ---------------------------------------------------------------------------
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="HTN Risk Stratification Tool", page_icon="🩺", layout="wide")
st.title("🩺 Hypertension Risk Stratification Tool")
st.markdown(
    "This tool estimates the probability of **hypertension (HTN)** in adults aged 18 to 69 "
    "years, using demographic and clinical variables, with or without a cholesterol result. "
    "It does not use a blood pressure measurement."
)

st.sidebar.header("Patient Information")
cholesterol_available = st.sidebar.radio("Do you have a cholesterol result?", ["Yes", "No"])


@st.cache_resource(show_spinner="Loading model...")
def load_model(filename: str):
    return joblib.load(HERE / filename)


filename = "lasso_htn_full.joblib" if cholesterol_available == "Yes" else "lasso_htn_nob8.joblib"
if not (HERE / filename).exists():
    st.error(f"Model file not found: {filename}")
    st.stop()
artifact = load_model(filename)
model, features, thresholds = artifact["model"], artifact["features"], artifact["thresholds"]
model_name = "Full model (with cholesterol)" if cholesterol_available == "Yes" else \
    "Simplified model (no cholesterol)"
st.caption(f"Model loaded: **{model_name}**")

YES_NO = {"Yes": 1, "No": 2}
CHOICES = {
    "c1": ("Sex", {"Male": 1, "Female": 2}),
    "region": ("Region", {"Central": 1, "Eastern": 2, "Northern": 3, "Western": 4}),
    "status": ("Residence", {"Urban": 1, "Rural": 2}),
    "edu": ("Highest education completed",
            {"None, or primary not completed": 1, "Primary completed": 2, "Secondary (O-level or A-level)": 3,
             "Tertiary (university or postgraduate)": 4}),
    "smoke": ("Smoking status", {"Never smoked": 1, "Former smoker": 2, "Current smoker": 3}),
    "t12": ("Uses smokeless tobacco (snuff, chewing tobacco)", YES_NO),
    "p1": ("Work involves vigorous-intensity activity", YES_NO),
    "p4": ("Work involves moderate-intensity activity", YES_NO),
    "p7": ("Walks or cycles to get to and from places", YES_NO),
    "p10": ("Does vigorous-intensity sport or recreation", YES_NO),
    "p13": ("Does moderate-intensity sport or recreation", YES_NO),
    "fbg61": ("Fasting blood glucose 6.1 mmol/L or higher", {"No": 0, "Yes": 1}),
}
# (label, min, max, default, step) -- limited to the range seen in the training data
NUMBERS = {
    "age": ("Age (years)", 18, 69, 45, 1),
    "m14": ("Waist circumference (cm)", 50.0, 170.0, 85.0, 0.1),
    "m15": ("Hip circumference (cm)", 50.0, 180.0, 95.0, 0.1),
    "HR": ("Heart rate (beats per minute)", 40, 130, 72, 1),
    # The survey meter recorded 2.59 to 10.36 mmol/L (100 to 400 mg/dL); 44% of
    # participants were at the 2.59 floor, so lower results are entered as 2.59.
    "b8": ("Total cholesterol (mmol/L; enter 2.59 if lower)", 2.59, 10.36, 4.0, 0.01),
    "sit_hours": ("Hours sitting or reclining on a typical day", 0.0, 15.0, 3.0, 0.5),
    "d1": ("Days per week eating fruit", 0, 7, 3, 1),
    "d3": ("Days per week eating vegetables", 0, 7, 4, 1),
    "d4": ("Servings of vegetables on a day you eat them (0 if none)", 0, 20, 2, 1),
}


def user_inputs() -> tuple[dict, dict]:
    data: dict[str, float] = {}
    shown: dict[str, object] = {}
    needs_body = any(v in features for v in ("m11", "m12", "bmi"))
    if needs_body:
        height = st.sidebar.number_input("Height (cm)", 130.0, 210.0, 165.0, 0.1)
        weight = st.sidebar.number_input("Weight (kg)", 29.0, 170.0, 65.0, 0.1)
        data.update(m11=height, m12=weight, bmi=weight / (height / 100) ** 2)
        if not 12 <= data["bmi"] <= 60:
            st.sidebar.warning("BMI is outside 12 to 60 kg/m2, the range the model was built "
                               "on. Check the height and weight.")
        shown.update({"Height (cm)": height, "Weight (kg)": weight,
                      "BMI (kg/m2)": round(data["bmi"], 1)})
    for v in features:
        if v in NUMBERS:
            label, lo, hi, default, step = NUMBERS[v]
            data[v] = st.sidebar.number_input(label, lo, hi, default, step)
            shown[label] = data[v]
        elif v in CHOICES:
            label, options = CHOICES[v]
            choice = st.sidebar.selectbox(label, list(options))
            data[v] = options[choice]
            shown[label] = choice
    return data, shown


input_data, shown = user_inputs()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Entered data")
    for k, v in shown.items():
        st.write(f"**{k}:** {v}")

with col2:
    st.subheader("Risk assessment")
    if st.button("Calculate HTN risk", type="primary"):
        input_df = pd.DataFrame([input_data])[features]
        proba = float(model.predict_proba(input_df)[0][1])
        if proba >= thresholds["high_ppv"]:
            st.error("**Risk level: High risk**")
        elif proba >= thresholds["high_sens"]:
            st.warning("**Risk level: Moderate risk**")
        else:
            st.success("**Risk level: Low risk**")
        shown_p = "<1%" if proba < 0.01 else ">99%" if proba > 0.99 else f"{proba * 100:.1f}%"
        st.metric(label="Predicted probability of HTN", value=shown_p)
        st.info(
            "**Disclaimer:** This is a screening tool, not a diagnostic test. Confirm with "
            "standard blood pressure measurement.\n"
            "- **High risk:** confirmatory blood pressure measurement and follow-up.\n"
            "- **Moderate risk:** lifestyle counselling and follow-up monitoring.\n"
            "- **Low risk:** reassurance and general health advice.\n"
            "- Developed by Dr. Bashir Ssuna with MUWRP."
        )
