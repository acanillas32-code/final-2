# Wine Quality Prediction System

## Problem
A wine producer needs a consistent preliminary way to screen red-wine samples using physicochemical measurements.

## Dataset
UCI Wine Quality — Red Wine CSV: https://archive.ics.uci.edu/dataset/186/wine+quality
The training script downloads the red-wine CSV and saves `wine_quality.csv`.

## Prediction rule
- Original quality score >= 6: GOOD QUALITY (class 1)
- Original quality score < 6: NEEDS IMPROVEMENT (class 0)

The original `quality` column is excluded from model inputs to avoid target leakage.

## Install and run
```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL displayed by Streamlit (normally http://localhost:8501). Download the sample input CSV in the app and upload it to test predictions.

## Evaluation
The training script prints accuracy, confusion matrix, precision, recall, and F1-score, and writes `evaluation_results.csv`. Run it to obtain actual metrics; do not report invented values.

## SAD mini-analysis
- **Problem:** Preliminary screening of wine samples from recorded measurements.
- **Users:** Quality-control staff, production supervisor, or data analyst.
- **Inputs:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, and alcohol.
- **Process:** Validate columns -> preprocess data -> apply trained model -> classify.
- **Outputs:** GOOD QUALITY / NEEDS IMPROVEMENT label, evaluation summary, downloadable predictions.
- **Limitation:** Trained on a particular red-wine dataset; may not generalize to other wine types, producers, regions, or testing conditions.

## Reflection answers
1. **Target and excluded quality column:** GOOD QUALITY versus NEEDS IMPROVEMENT based on score >= 6. The original score is excluded because it defines the target and would leak the answer.
2. **GOOD QUALITY recall:** Recall measures the proportion of actual good-quality samples correctly identified. Low recall means some good samples are missed, potentially causing unnecessary further review.
3. **Different producer:** Measurement distributions and production conditions can differ from training data, so performance may change.
4. **Preliminary screening:** Predictions do not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
5. **Improve reliability:** Collect representative data across producers, regions, vintages, and conditions; verify labels; conduct external testing and cross-validation; monitor performance.

## GitHub
Create a public repository named `wine-quality-prediction`, then:
```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build wine quality prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/wine-quality-prediction.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. Never put passwords or access tokens in source code.
