"""
api.py
──────
FastAPI REST API for TalentAI Solutions.

Endpoints:
  POST /match-score  – cosine similarity score between resume and JD
  POST /predict      – shortlist prediction + extracted skills
  POST /chat         – AI recruiter assistant (GenAI response)

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""

import os
import re
import joblib
import warnings
import numpy as np
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import nlp
import genai_helper

warnings.filterwarnings('ignore')

app = FastAPI(
    title="TalentAI API",
    description="AI-Powered Smart Hiring & Candidate Intelligence API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Load artefacts at startup
# ─────────────────────────────────────────────────────────────
_model    = None
_tfidf    = None
_scaler   = None

def _load_artefacts():
    global _model, _tfidf, _scaler
    for fname, var_name in [
        ('best_model.pkl',           '_model'),
        ('ml_tfidf_vectorizer.pkl',  '_tfidf'),
        ('scaler.pkl',               '_scaler'),
    ]:
        if os.path.exists(fname):
            globals()[var_name] = joblib.load(fname)

_load_artefacts()


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────
class MatchRequest(BaseModel):
    resume: str
    job_description: str

class PredictRequest(BaseModel):
    resume: str
    job_description: str

class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = ""


# ─────────────────────────────────────────────────────────────
# Helper: TF-IDF Cosine Score
# ─────────────────────────────────────────────────────────────
def _cosine_score(resume_clean: str, jd_clean: str) -> float:
    vect = TfidfVectorizer(stop_words='english')
    try:
        mat = vect.fit_transform([resume_clean, jd_clean])
        score = cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100
    except Exception:
        score = 0.0
    return round(float(score), 2)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "TalentAI API is running 🚀", "docs": "/docs"}


@app.post("/match-score")
def match_score(req: MatchRequest):
    """
    Returns cosine similarity score (0-100) between resume and job description.
    """
    resume_clean = nlp.clean_text(req.resume)
    jd_clean     = nlp.clean_text(req.job_description)
    score = _cosine_score(resume_clean, jd_clean)
    return {
        "match_score": score,
        "label": "Strong Match" if score >= 40 else "Moderate Match" if score >= 20 else "Weak Match"
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predicts whether a candidate will be shortlisted.
    Returns: prediction, match_score, extracted_skills, experience.
    """
    resume_clean = nlp.clean_text(req.resume)
    jd_clean     = nlp.clean_text(req.job_description)
    score        = _cosine_score(resume_clean, jd_clean)
    skills       = nlp.extract_skills(req.resume)
    experience   = 0.0

    prediction   = "Unknown"
    confidence   = None

    if _model is not None and _tfidf is not None:
        try:
            from scipy.sparse import hstack, csr_matrix
            X_text = _tfidf.transform([resume_clean])
            X_num  = csr_matrix([[experience, len(skills), score]])
            X      = hstack([X_text, X_num])
            pred   = int(_model.predict(X)[0])
            proba  = float(_model.predict_proba(X)[0][pred]) if hasattr(_model, 'predict_proba') else None
            prediction = "Shortlisted" if pred == 1 else "Rejected"
            confidence = round(proba * 100, 1) if proba else None
        except Exception as e:
            prediction = "Shortlisted" if score >= 20 else "Rejected"
    else:
        # Fallback rule-based when model not yet trained
        prediction = "Shortlisted" if score >= 20 else "Rejected"

    return {
        "prediction" : prediction,
        "confidence" : confidence,
        "match_score": score,
        "skills"     : skills,
        "experience" : experience,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    AI Recruiter Assistant - answers hiring-related questions using Gemini.
    """
    answer = genai_helper.answer_hiring_question(
        question=req.question.strip(),
        context=req.context.strip() if req.context else ""
    )
    return {"question": req.question.strip(), "answer": answer}


@app.get("/health")
def health():
    return {
        "status"       : "ok",
        "model_loaded" : _model is not None,
        "tfidf_loaded" : _tfidf is not None,
    }
