import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

st.title("🍷 Wine Quality Prediction System")
st.caption("Classroom prototype for preliminary red-wine quality classification.")

if not Path("wine_quality.csv").exists():
    st.error("wine_quality.csv was not found. Run `python train_model.py` first.")
    st.stop()
if not Path("wine_model.pkl").exists():
    st.error("wine_model.pkl was not found. Run `python train_model.py` first.")
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
st.write("Upload a CSV containing the 11 physicochemical predictor columns.")
features = data.drop(columns=["quality"])
st.download_button(
    "Download sample input CSV",
    features.head(5).to_csv(index=False),
    file_name="sample_wine_input.csv",

    mime="text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required = list(model.feature_names_in_)
        missing = [col for col in required if col not in new_data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[required])
            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                file_name="wine_quality_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Could not read or process the CSV: {e}")

st.warning(
    "Responsible use: predictions are preliminary screening results, not a substitute "
    "for laboratory testing, sensory evaluation, food-safety procedures, or professional QC."
)
