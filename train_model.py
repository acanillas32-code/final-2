import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def main():
    df = pd.read_csv(URL, sep=";")
    df.to_csv("wine_quality.csv", index=False)
    print("Dataset shape:", df.shape)
    print(df.head())
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred,
        target_names=["Needs improvement", "Good quality"], zero_division=0))
    report = classification_report(y_test, pred,
        target_names=["Needs improvement", "Good quality"], zero_division=0,
        output_dict=True)
    pd.DataFrame(report).transpose().to_csv("evaluation_results.csv")
    joblib.dump(model, "wine_model.pkl")
    print("Saved wine_model.pkl and evaluation_results.csv")

if __name__ == "__main__":
    main()
