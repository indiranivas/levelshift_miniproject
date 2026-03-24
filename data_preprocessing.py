"""
data_preprocessing.py
---------------------
Fixed & enhanced preprocessing pipeline for the Resume Dataset.

Fixes from notebook audit:
  [OK] SettingWithCopyWarning  -> use .loc[]
  [OK] Drop Resume_html early  -> save memory
  [OK] Null + duplicate checks
  [OK] Experience extraction scoped to Experience section only
  [OK] TF-IDF fit on full corpus with stop_words='english'
  [OK] NLP cleaning via nlp.py (lemmatization + stopwords)
  [OK] skills_count feature added
  [OK] Label encoding for Category
  [OK] StandardScaler on numerical features

Run:  python data_preprocessing.py
"""

import re
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser as dateparser
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

import nlp  # local nlp.py

warnings.filterwarnings('ignore')

# Force UTF-8 output on Windows to avoid UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "Resume.csv")
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "processed_resumes.csv")


# -----------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load dataset, trying common column variations."""
    if not os.path.exists(path):
        for alt in ["UpdatedResumeDataSet.csv", "resume_dataset.csv", "resumes.csv"]:
            alt_path = os.path.join(os.path.dirname(path), alt)
            if os.path.exists(alt_path):
                path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. "
                "Please place Resume.csv in the project folder."
            )
    try:
        df = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
    except TypeError:
        df = pd.read_csv(path, encoding='latin-1')
    print(f"[OK] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# -----------------------------------------------------------------
# 2. BASIC QUALITY CHECKS
# -----------------------------------------------------------------
def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    print("\n-- Data Quality ------------------------------------------")
    print(f"Nulls per column:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    text_col = _get_text_col(df)
    before = len(df)
    df = df.dropna(subset=[text_col]).copy()
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"\nDropped {before - len(df)} null/duplicate rows -> {len(df)} remain")
    return df


def _get_text_col(df: pd.DataFrame) -> str:
    for col in ["Resume_str", "Resume", "resume", "text", "content"]:
        if col in df.columns:
            return col
    raise KeyError("Could not find resume text column in dataset.")


def _get_category_col(df: pd.DataFrame) -> str:
    for col in ["Category", "category", "label", "Label"]:
        if col in df.columns:
            return col
    return None


# -----------------------------------------------------------------
# 3. DROP UNUSED COLUMNS
# -----------------------------------------------------------------
def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["Resume_html", "ID", "id"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"\n[OK] Dropped columns: {drop_cols}")
    return df


# -----------------------------------------------------------------
# 4. TEXT CLEANING
# -----------------------------------------------------------------
def clean_resumes(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _get_text_col(df)
    print(f"\n[..] Cleaning '{text_col}' column ... ", end="", flush=True)
    df.loc[:, 'cleaned_resume'] = df[text_col].apply(nlp.clean_text)
    print("done")
    return df


# -----------------------------------------------------------------
# 5. EXPERIENCE EXTRACTION (FIXED)
# -----------------------------------------------------------------
def extract_experience(text: str) -> float:
    """
    Extract years of experience scoped to the Experience section only.
    Strategy 1: Direct mention "X+ years"
    Strategy 2: Sum date ranges within the Experience section.
    """
    if not isinstance(text, str):
        return 0.0

    exp_text = nlp.extract_experience_section(text).lower()
    current_date = datetime.now()

    match = re.search(r'(\d+)\+?\s+year', exp_text)
    if match:
        years = float(match.group(1))
        if 0 < years < 50:
            return years

    total_days = 0
    date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\d{1,2}/\d{4}|\d{4})'
        r'\s*(?:to|-)\s*'
        r'(current|present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|\d{1,2}/\d{4}|\d{4})',
        exp_text, re.IGNORECASE
    )

    for start_str, end_str in date_ranges:
        try:
            start_date = dateparser.parse(start_str.strip(), fuzzy=True)
            if 'current' in end_str.lower() or 'present' in end_str.lower():
                end_date = current_date
            else:
                end_date = dateparser.parse(end_str.strip(), fuzzy=True)
            diff = (end_date - start_date).days
            if 30 < diff < 365 * 45:
                total_days += diff
        except Exception:
            continue

    return round(total_days / 365, 1)


def add_experience_feature(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _get_text_col(df)
    print("[..] Extracting experience ... ", end="", flush=True)
    df.loc[:, 'experience'] = df[text_col].apply(extract_experience)
    print("done")
    return df


# -----------------------------------------------------------------
# 6. SKILLS COUNT FEATURE
# -----------------------------------------------------------------
def add_skills_count(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _get_text_col(df)
    print("[..] Counting skills ... ", end="", flush=True)
    df.loc[:, 'skills_count'] = df[text_col].apply(nlp.count_skills)
    print("done")
    return df


# -----------------------------------------------------------------
# 7. MATCH SCORE (FIXED TF-IDF)
# -----------------------------------------------------------------
def calculate_match_scores(
    df: pd.DataFrame,
    job_description: str,
    vectorizer: TfidfVectorizer = None
) -> pd.DataFrame:
    """
    Compute cosine similarity between each resume and the job description.
    TF-IDF is fit on the FULL corpus (all resumes + JD).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = df['cleaned_resume'].tolist() + [nlp.clean_text(job_description)]
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

    tfidf_matrix  = vectorizer.fit_transform(corpus)
    jd_vector     = tfidf_matrix[-1]
    resume_matrix = tfidf_matrix[:-1]

    scores = cosine_similarity(resume_matrix, jd_vector).flatten() * 100
    df.loc[:, 'score'] = scores.round(2)
    return df, vectorizer


# -----------------------------------------------------------------
# 8. LABEL ENCODING + SCALING
# -----------------------------------------------------------------
def encode_and_scale(df: pd.DataFrame):
    cat_col = _get_category_col(df)
    le = LabelEncoder()
    if cat_col:
        df.loc[:, 'category_encoded'] = le.fit_transform(df[cat_col])
        print(f"\n[OK] Encoded '{cat_col}': {len(le.classes_)} classes")
    else:
        le = None

    scaler = StandardScaler()
    num_cols = [c for c in ['experience', 'skills_count', 'score'] if c in df.columns]
    if num_cols:
        df.loc[:, [f'{c}_scaled' for c in num_cols]] = scaler.fit_transform(df[num_cols])
        print(f"[OK] Scaled: {num_cols}")

    return df, le, scaler


# -----------------------------------------------------------------
# 9. MAIN
# -----------------------------------------------------------------
def run_pipeline(job_description: str = "") -> pd.DataFrame:
    print("=" * 55)
    print("  TalentAI - Data Preprocessing Pipeline")
    print("=" * 55)

    df = load_data()
    df = drop_unused(df)
    df = quality_checks(df)
    df = clean_resumes(df)
    df = add_experience_feature(df)
    df = add_skills_count(df)

    if job_description:
        df, vectorizer = calculate_match_scores(df, job_description)
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        print("[OK] Saved tfidf_vectorizer.pkl")
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        vectorizer.fit(df['cleaned_resume'])
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        print("[OK] Saved tfidf_vectorizer.pkl (no JD provided)")

    df, le, scaler = encode_and_scale(df)

    if le:
        joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Saved processed data -> {OUTPUT_PATH}")
    print(f"     Final shape: {df.shape}")
    print("\nSample output:")
    cols = ['cleaned_resume', 'experience', 'skills_count']
    if 'score' in df.columns:
        cols.append('score')
    print(df[cols].head(3).to_string())
    return df


if __name__ == "__main__":
    sample_jd = """
    We are looking for an experienced HR Manager with strong knowledge of
    recruitment, HRIS systems, employee relations, performance management,
    FMLA, compensation and benefits administration.
    """
    run_pipeline(sample_jd)

