import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os

PLOTS_DIR = "static/plots"
MODEL_PATH = "ml/model.pkl"
ENCODER_PATH = "ml/encoder.pkl"


def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre"])
    features = ["genre_encoded", "author_popularity", "year_published",
                "price", "page_count", "times_borrowed", "rating"]
    X = df[features]
    y = df["is_popular"]
    return X, y, le, features


def train_model(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("ml", exist_ok=True)

    X, y, le, features = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total": len(df),
        "popular_count": int(y.sum()),
        "not_popular_count": int((y == 0).sum()),
    }

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    _plot_genre_popularity(df)
    _plot_correlation(df)
    _plot_feature_importance(clf, features)
    _plot_roc(y_test, y_proba)

    return metrics


def _plot_genre_popularity(df):
    plt.figure(figsize=(10, 5))
    genre_stats = df.groupby("genre")["is_popular"].mean().sort_values(ascending=False)
    colors = sns.color_palette("Blues_r", len(genre_stats))
    bars = plt.bar(genre_stats.index, genre_stats.values * 100, color=colors, edgecolor="white", linewidth=0.5)
    plt.title("share of popular books by genre (%)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("genre", fontsize=12)
    plt.ylabel("share of popular books (%)", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    for bar, val in zip(bars, genre_stats.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/genre_popularity.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_correlation(df):
    plt.figure(figsize=(9, 7))
    num_cols = ["author_popularity", "year_published", "price",
                "page_count", "times_borrowed", "rating", "is_popular"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}
    )
    plt.title("feature correlation matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_feature_importance(clf, features):
    plt.figure(figsize=(9, 5))
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    colors = sns.color_palette("viridis", len(sorted_features))
    bars = plt.barh(sorted_features[::-1], sorted_importances[::-1],
                    color=colors[::-1], edgecolor="white")
    plt.title("feature importance", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("importance", fontsize=12)
    for bar, val in zip(bars, sorted_importances[::-1]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()


def _plot_roc(y_test, y_proba):
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, color="#4C72B0", lw=2.5,
             label=f"roc curve (auc = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="random model")
    plt.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
    plt.xlabel("false positive rate", fontsize=12)
    plt.ylabel("true positive rate", fontsize=12)
    plt.title("roc curve", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()


def predict_book(genre, author_popularity, year_published, price, page_count, times_borrowed, rating):
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    try:
        genre_encoded = le.transform([genre])[0]
    except ValueError:
        genre_encoded = 0

    X = pd.DataFrame([{
        "genre_encoded": genre_encoded,
        "author_popularity": float(author_popularity),
        "year_published": int(year_published),
        "price": float(price),
        "page_count": int(page_count),
        "times_borrowed": int(times_borrowed),
        "rating": float(rating),
    }])

    prediction = clf.predict(X)[0]
    probability = clf.predict_proba(X)[0][1]

    if prediction == 1:
        if probability > 0.85:
            recommendation = "recommended to purchase 10-15 copies"
        elif probability > 0.65:
            recommendation = "recommended to purchase 5-10 copies"
        else:
            recommendation = "recommended to purchase 3-5 copies"
    else:
        recommendation = "1-2 copies is enough"

    return {
        "prediction": int(prediction),
        "label": "popular" if prediction == 1 else "not popular",
        "probability": round(float(probability) * 100, 1),
        "recommendation": recommendation,
    }
