import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ---------- CONFIG ----------

DATA_PATH = os.path.join("data", "wdbc.data")
FIGURES_DIR = "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# ---------- UTILS ----------

def ensure_directories():
    """Create output folders if they do not exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def get_column_names():
    """Return column names for the WDBC dataset."""
    return [
        "id",
        "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
        "symmetry_worst", "fractal_dimension_worst",
    ]


# ---------- DATA LOADING & EDA ----------

def load_data():
    """Load the WDBC dataset from a local file."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Make sure wdbc.data is placed in the data/ folder."
        )

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = get_column_names()
    return df


def basic_info(df: pd.DataFrame):
    """Print basic info about the dataset."""
    print("First 5 rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nDiagnosis value counts:")
    print(df["diagnosis"].value_counts())
    print("\nMissing values per column:")
    print(df.isna().sum())


def plot_class_distribution(df: pd.DataFrame):
    """Bar plot of benign vs malignant counts."""
    counts = df["diagnosis"].value_counts().sort_index()  # B, M
    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution: Benign vs Malignant")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved class distribution plot to {out_path}")


def plot_histogram_radius_mean_by_class(df: pd.DataFrame):
    """Histogram of radius_mean for benign vs malignant."""
    benign = df[df["diagnosis"] == "B"]["radius_mean"]
    malignant = df[df["diagnosis"] == "M"]["radius_mean"]

    plt.figure()
    plt.hist(benign, bins=20, alpha=0.5, label="Benign")
    plt.hist(malignant, bins=20, alpha=0.5, label="Malignant")
    plt.xlabel("Radius (mean)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Mean Radius by Diagnosis")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "hist_radius_mean_by_diagnosis.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram of radius_mean by class to {out_path}")


def plot_boxplots_by_class(df: pd.DataFrame, features):
    """Boxplots for selected features (one figure per feature)."""
    for feature in features:
        data = [
            df[df["diagnosis"] == "B"][feature],
            df[df["diagnosis"] == "M"][feature],
        ]
        plt.figure()
        plt.boxplot(data, labels=["Benign", "Malignant"])
        plt.ylabel(feature.replace("_", " ").title())
        plt.title(f"{feature.replace('_', ' ').title()} by Diagnosis")
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"boxplot_{feature}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved boxplot for {feature} to {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap for the 10 mean features."""
    mean_features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
        "symmetry_mean", "fractal_dimension_mean",
    ]
    corr = df[mean_features].corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values)
    labels = [f.split("_")[0].title() for f in mean_features]
    plt.xticks(range(len(mean_features)), labels, rotation=90)
    plt.yticks(range(len(mean_features)), labels)
    plt.title("Correlation Heatmap of Mean Features")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "correlation_heatmap_mean_features.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved correlation heatmap to {out_path}")


def run_eda(df: pd.DataFrame):
    """Run all basic EDA steps and save plots."""
    basic_info(df)
    plot_class_distribution(df)
    plot_histogram_radius_mean_by_class(df)
    plot_boxplots_by_class(
        df,
        features=["radius_mean", "area_mean", "concavity_mean"],
    )
    plot_correlation_heatmap(df)


# ---------- PREPROCESSING ----------

def preprocess(df: pd.DataFrame):
    """
    Preprocess the dataset:
    - Drop ID
    - Encode diagnosis (B -> 0, M -> 1)
    - Split into train/test
    - Standardize features
    """
    df = df.copy()
    df["target"] = df["diagnosis"].map({"B": 0, "M": 1})

    X = df.drop(columns=["id", "diagnosis", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTrain set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


# ---------- MODEL TRAINING & EVALUATION ----------

def evaluate_model(name, y_true, y_pred):
    """Print metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f} (malignant)")
    print(f"Recall:    {rec:.3f} (malignant)")
    print(f"F1-score:  {f1:.3f} (malignant)")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))


def train_logistic_regression(X_train_scaled, y_train):
    """Train a logistic regression model."""
    log_reg = LogisticRegression(
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    log_reg.fit(X_train_scaled, y_train)
    return log_reg


def train_random_forest(X_train, y_train):
    """Train a random forest classifier (on unscaled features)."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    return rf


def cross_validate_model(model, X, y, name):
    """Optional: simple cross-validation on the training data."""
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"\nCross-validation accuracy for {name}:")
    print("Scores:", np.round(scores, 3))
    print("Mean:", scores.mean().round(3), "Std:", scores.std().round(3))


# ---------- MAIN ----------

def main():
    ensure_directories()

    print("Loading data...")
    df = load_data()

    print("\nRunning exploratory data analysis...")
    run_eda(df)

    print("\nPreprocessing data...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess(df)

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    evaluate_model("Logistic Regression", y_test, y_pred_lr)

    # Optional CV for logistic regression (on scaled training data)
    # cross_validate_model(log_reg, X_train_scaled, y_train, "Logistic Regression")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    evaluate_model("Random Forest", y_test, y_pred_rf)

    # Optional CV for random forest (on unscaled training data)
    # cross_validate_model(rf, X_train, y_train, "Random Forest")

    print("\nDone.")


if __name__ == "__main__":
    main()
