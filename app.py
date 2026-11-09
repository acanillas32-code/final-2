import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification.")
if not Path("wine_model.pkl").exists() or not Path("wine_quality.csv").exists():
    st.error("Required files are missing. Run `python train_model.py` first.")
    st.stop()
model = load_model()
data = pd.read_csv("wine_quality.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))
st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)
st.subheader("Predict uploaded wine samples")
required = list(model.feature_names_in_)
sample = data.drop(columns=["quality"]).head(5)
st.download_button("Download sample input CSV", sample.to_csv(index=False),
                 "sample_wine_input.csv", "text/csv")
uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        samples = pd.read_csv(uploaded)
        missing = [col for col in required if col not in samples.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(samples[required])
            results = samples.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if int(v) == 1 else "NEEDS IMPROVEMENT"
                for v in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
            st.download_button("Download prediction results", results.to_csv(index=False),
                             "wine_quality_predictions.csv", "text/csv")
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
st.warning("Preliminary screening only—not laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.")
