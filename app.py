import requests, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from db import fetch_recent, init_db

load_dotenv()
init_db()

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AI Clinical Decision Support", layout="centered")
st.title("AI-Based Clinical Decision Support (Heart Disease)")
st.caption("Academic demo • CPU-only • XGBoost/LogReg + Calibration + optional SHAP")

# integer-coded categoricals with valid ranges
INT_FIELDS = {
    "sex": (0,1), "cp": (0,3), "fbs": (0,1), "restecg": (0,2),
    "exang": (0,1), "slope": (0,2), "ca": (0,3), "thal": (0,3)
}
FLOAT_FIELDS = {"age","trestbps","chol","thalach","oldpeak"}

defaults = {
    "age": 54.0, "sex": 1, "cp": 0, "trestbps": 130.0, "chol": 246.0,
    "fbs": 0, "restecg": 1, "thalach": 150.0, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
}

st.subheader("Enter patient features")
cols = st.columns(3)
vals = {}
order = ["age","sex","cp","trestbps","chol","fbs","restecg",
         "thalach","exang","oldpeak","slope","ca","thal"]

for i, name in enumerate(order):
    with cols[i % 3]:
        if name in INT_FIELDS:
            lo, hi = INT_FIELDS[name]
            vals[name] = st.number_input(name, min_value=lo, max_value=hi, value=int(defaults[name]), step=1)
        else:
            step = 0.1 if name == "oldpeak" else 1.0
            vals[name] = st.number_input(name, value=float(defaults[name]), step=step, format="%.2f")

if st.button("Predict"):
    try:
        r = requests.post(API_URL, json={"features": vals}, timeout=10)
        r.raise_for_status()
        res = r.json()
        prob = res["prob"]; label = res["label"]

        st.success(f"Prediction: {'Heart Disease' if label==1 else 'No Disease'} • Probability: {prob:.2f}")
        st.caption(f"Model version: {res.get('model_version','?')}")

        top = res.get("top_factors", [])
        if top:
            st.subheader("Top contributing features")
            df = pd.DataFrame(top)
            fig = plt.figure()
            plt.barh(df["feature"], df["impact"])
            plt.xlabel("SHAP impact (positive raises risk)")
            plt.gca().invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("Explanation not available for this instance.")
    except requests.RequestException as e:
        st.error(f"API error: {e}")

st.divider()
st.subheader("Recent predictions (local logs)")
try:
    rows = fetch_recent(limit=10)
    if rows:
        table = []
        for r in rows:
            table.append({
                "timestamp (UTC)": r["ts"],
                "prob": round(r["prob"], 3),
                "label": "Heart Disease" if r["label"] == 1 else "No Disease",
                "model": r["model_version"]
            })
        st.dataframe(pd.DataFrame(table))
    else:
        st.write("No predictions yet. Submit one above.")
except Exception as e:
    st.info("Logs not available yet.")
    st.caption(f"(Details: {e})")
