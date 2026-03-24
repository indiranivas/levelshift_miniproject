"""
ml_pipeline.py
──────────────
ML Classification Pipeline for Resume Shortlisting.

Models: Logistic Regression | Decision Tree | Random Forest | SVM
Tasks : Train/test split → Cross-validation → GridSearchCV → Evaluation → Save

Run:  python ml_pipeline.py
      (requires processed_resumes.csv from data_preprocessing.py)
"""

import os
import sys
import warnings

# Force UTF-8 output on Windows to avoid UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing   import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
PROCESSED_CSV  = 'processed_resumes.csv'
MODEL_OUT      = 'best_model.pkl'
SCORE_THRESHOLD = 20.0   # resumes scoring ≥ this are "shortlisted"

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(
            f"{PROCESSED_CSV} not found. Run data_preprocessing.py first."
        )
    df = pd.read_csv(PROCESSED_CSV)
    print(f"✅ Loaded {PROCESSED_CSV}: {df.shape[0]} rows")
    return df


# ─────────────────────────────────────────────────────────────
# 2. PREPARE FEATURES & LABEL
# ─────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame):
    """
    X = TF-IDF on cleaned_resume + numerical features (experience, skills_count, score)
    y = binary shortlisted label (1 if score >= threshold)
    """
    # Binary label
    if 'score' in df.columns:
        y = (df['score'] >= SCORE_THRESHOLD).astype(int)
    else:
        # Fallback: use experience > median as proxy label
        y = (df['experience'] >= df['experience'].median()).astype(int)
    print(f"📌 Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df['cleaned_resume'].fillna(''))

    # Numerical features
    num_cols = [c for c in ['experience', 'skills_count', 'score'] if c in df.columns]
    X_num = csr_matrix(df[num_cols].fillna(0).values)

    X = hstack([X_text, X_num])
    print(f"📐 Feature matrix shape: {X.shape}")
    return X, y, tfidf


# ─────────────────────────────────────────────────────────────
# 3. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results = {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall'   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1'       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else 'N/A',
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    results['CV F1 (mean)'] = round(cv_scores.mean(), 4)
    results['CV F1 (std)']  = round(cv_scores.std(), 4)

    return results, model, y_pred


# ─────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────
def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("\n🔧 GridSearchCV – Random Forest …")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth'   : [None, 10, 20],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"   Best params: {grid.best_params_}")
    print(f"   Best CV F1:  {grid.best_score_:.4f}")
    return grid.best_estimator_


# ─────────────────────────────────────────────────────────────
# 5. VISUALISE RESULTS
# ─────────────────────────────────────────────────────────────
def plot_comparison(results_df: pd.DataFrame):
    metrics   = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    plot_data = results_df.set_index('Model')[metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data.T.plot(kind='bar', ax=ax)
    ax.set_title('Model Comparison – Evaluation Metrics')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=120)
    plt.close()
    print("📊 Saved model_comparison.png")


def plot_confusion_matrix(model, X_test, y_test, name="Best Model"):
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Rejected", "Shortlisted"],
        cmap='Blues'
    )
    disp.ax_.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    plt.close()
    print("📊 Saved confusion_matrix.png")


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 55)
    print("  TalentAI – ML Classification Pipeline")
    print("=" * 55)

    df = load_data()
    X, y, tfidf = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    # Base models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree'      : DecisionTreeClassifier(random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM'                : SVC(kernel='linear', probability=True, random_state=42),
    }

    all_results = []
    trained     = {}

    print("\n── Training & Evaluating Models ──────────────")
    for name, model in models.items():
        print(f"  ▶ {name} …", end=" ", flush=True)
        res, trained_model, _ = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        all_results.append(res)
        trained[name] = trained_model
        print("done")

    results_df = pd.DataFrame(all_results)
    print("\n── Results ───────────────────────────────────")
    print(results_df.to_string(index=False))

    # Hyper-tune best base model (RF)
    best_rf = tune_random_forest(X_train, y_train)
    res_tuned, _, y_pred_tuned = evaluate_model(
        'RF (Tuned)', best_rf, X_train, X_test, y_train, y_test
    )
    all_results.append(res_tuned)
    results_df = pd.DataFrame(all_results)

    # Select overall best by F1
    best_row   = results_df.loc[results_df['F1'].idxmax()]
    best_name  = best_row['Model']
    best_model = best_rf if 'RF (Tuned)' in best_name else trained[best_name]

    print(f"\n🏆 Best model: {best_name}  (F1={best_row['F1']})")

    # Full classification report
    y_pred_best = best_model.predict(X_test)
    print("\n── Classification Report ─────────────────────")
    print(classification_report(y_test, y_pred_best,
                                target_names=["Rejected", "Shortlisted"]))

    # Save artefacts
    joblib.dump(best_model, MODEL_OUT)
    joblib.dump(tfidf,      'ml_tfidf_vectorizer.pkl')
    print(f"💾 Saved {MODEL_OUT}  +  ml_tfidf_vectorizer.pkl")

    # Plots
    plot_comparison(results_df[results_df['ROC-AUC'] != 'N/A'].copy()
                    .assign(**{'ROC-AUC': lambda d: d['ROC-AUC'].astype(float)}))
    plot_confusion_matrix(best_model, X_test, y_test, best_name)

    return best_model, results_df


if __name__ == "__main__":
    run_pipeline()
