# Wine Quality Prediction System

A classroom prototype that classifies red-wine samples as **GOOD QUALITY** or **NEEDS IMPROVEMENT** using physicochemical measurements.

## Prediction rule
- `quality >= 6` → GOOD QUALITY (class 1)
- `quality < 6` → NEEDS IMPROVEMENT (class 0)
- The original `quality` column is excluded from model inputs to prevent target leakage.

## Dataset
UCI Machine Learning Repository — Wine Quality (red wine):
https://archive.ics.uci.edu/dataset/186/wine+quality

The training script downloads the red-wine CSV from UCI and saves `wine_quality.csv`.

## Technologies
Python, pandas, scikit-learn, Random Forest, Streamlit, joblib.

## Setup
```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open the local URL displayed by Streamlit (normally http://localhost:8501).

## App usage
1. Run `train_model.py` to download the dataset, train the model, and create `wine_model.pkl`.
2. Start the app with `streamlit run app.py`.
3. Download the sample input CSV from the app or upload a CSV with all 11 predictor columns.
4. Review predictions and download the results CSV.

## Generated files
- `wine_quality.csv` — downloaded dataset
- `wine_model.pkl` — trained pipeline
- `evaluation.txt` — accuracy, confusion matrix, and classification report

## Limitations
This model is trained on the UCI red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening aids, not final quality decisions or a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
