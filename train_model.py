import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def main():
    print("Downloading UCI red wine dataset...")
    df = pd.read_csv(DATA_URL, sep=";")
    df.to_csv("wine_quality.csv", index=False)
    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("\nQuality distribution:\n", df["quality"].value_counts().sort_index())

    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])  # Prevent target leakage

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    report = classification_report(
        y_test, pred, target_names=["Needs improvement", "Good quality"],
        zero_division=0
    )
    matrix = confusion_matrix(y_test, pred)

    print("\nAccuracy:", round(accuracy, 3))
    print("Confusion matrix:\n", matrix)
    print("Classification report:\n", report)

    with open("evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.3f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(matrix))
        f.write("\n\nClassification report:\n")
        f.write(report)

    joblib.dump(model, "wine_model.pkl")
    print("\nSaved wine_quality.csv, wine_model.pkl, and evaluation.txt")

if __name__ == "__main__":
    main()
