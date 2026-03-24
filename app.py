"""
app.py
──────
TalentAI Solutions – Full Streamlit Web Application

Tabs:
  1. 🏠 Home
  2. 📄 Resume Screener (upload + match score + prediction)
  3. 🤖 AI Assistant   (GenAI summary / questions / feedback)
  4. 💬 RAG Chatbot    (hiring policy Q&A)
  5. 📊 Dashboard      (charts + analytics)

Run:  streamlit run app.py
"""

import os
import re
import io
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import db
import text_extractor
import nlp
import data_preprocessing as dp
import genai_helper

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentAI Solutions",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS – premium dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #fff; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.1); }
    [data-testid="stSidebar"] * { color: #eee !important; }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #7c83fd; }
    .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #aaa; }

    /* Score badge */
    .score-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 6px 0;
    }
    .score-high   { background: #00c853; color: #000; }
    .score-medium { background: #ff9800; color: #000; }
    .score-low    { background: #f44336; color: #fff; }

    /* Prediction pill */
    .pill-shortlisted { background: #00e676; color: #000; padding: 4px 14px; border-radius: 20px; font-weight: 700; }
    .pill-rejected    { background: #ff1744; color: #fff; padding: 4px 14px; border-radius: 20px; font-weight: 700; }

    /* Tab content */
    .stTabs [role="tablist"] button { color: #ccc !important; font-size: 0.95rem; }
    .stTabs [role="tablist"] button[aria-selected="true"] { color: #7c83fd !important; border-bottom: 2px solid #7c83fd; }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #7c83fd, #5c67f5);
        color: white; border: none; border-radius: 8px;
        padding: 8px 20px; font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(124,131,253,0.4); }

    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background: rgba(124,131,253,0.2);
        border: 1px solid #7c83fd;
        border-radius: 14px;
        padding: 3px 10px;
        margin: 3px 2px;
        font-size: 0.8rem;
        color: #c5c8ff;
    }

    /* Chat bubbles */
    .chat-user { background: rgba(124,131,253,0.2); border-radius: 12px 12px 4px 12px; padding: 10px 14px; margin: 6px 0; }
    .chat-ai   { background: rgba(255,255,255,0.06); border-radius: 12px 12px 12px 4px; padding: 10px 14px; margin: 6px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INIT DB
# ─────────────────────────────────────────────────────────────
db.create_db()

# ─────────────────────────────────────────────────────────────
# LOAD ML ARTEFACTS (optional – may not exist yet)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    artefacts = {}
    for key, fname in [
        ('model',  'best_model.pkl'),
        ('tfidf',  'ml_tfidf_vectorizer.pkl'),
        ('scaler', 'scaler.pkl'),
    ]:
        if os.path.exists(fname):
            artefacts[key] = joblib.load(fname)
    return artefacts

artefacts = load_artefacts()

# ─────────────────────────────────────────────────────────────
# HELPER: match score
# ─────────────────────────────────────────────────────────────
def compute_match_score(resume_text: str, jd_text: str) -> float:
    r_clean = nlp.clean_text(resume_text)
    j_clean = nlp.clean_text(jd_text)
    vect    = TfidfVectorizer(stop_words='english')
    try:
        mat   = vect.fit_transform([r_clean, j_clean])
        score = cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100
    except Exception:
        score = 0.0
    return round(float(score), 2)


# ─────────────────────────────────────────────────────────────
# HELPER: ML prediction
# ─────────────────────────────────────────────────────────────
def ml_predict(resume_text: str, score: float) -> dict:
    if 'model' not in artefacts or 'tfidf' not in artefacts:
        label = "Shortlisted" if score >= 20 else "Rejected"
        return {"label": label, "confidence": None, "model_used": False}

    from scipy.sparse import hstack, csr_matrix
    r_clean = nlp.clean_text(resume_text)
    skills  = nlp.extract_skills(resume_text)
    X_text  = artefacts['tfidf'].transform([r_clean])
    X_num   = csr_matrix([[0.0, len(skills), score]])
    X       = hstack([X_text, X_num])
    pred    = int(artefacts['model'].predict(X)[0])
    proba   = float(artefacts['model'].predict_proba(X)[0][pred]) \
              if hasattr(artefacts['model'], 'predict_proba') else None
    return {
        "label"      : "Shortlisted" if pred == 1 else "Rejected",
        "confidence" : round(proba * 100, 1) if proba else None,
        "model_used" : True,
    }


# ─────────────────────────────────────────────────────────────
# HELPER: GenAI call  (delegates to genai_helper.py – new google-genai SDK)
# ─────────────────────────────────────────────────────────────
def genai_call(prompt: str) -> str:
    return genai_helper.call_gemini(prompt)


# ─────────────────────────────────────────────────────────────
# HELPER: RAG Chatbot
# ─────────────────────────────────────────────────────────────
POLICY_DOCS = []  # In-memory list of (text, embedding_vec)

def add_policy_doc(text: str):
    """Add a policy document to the in-memory store."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    POLICY_DOCS.append(nlp.clean_text(text))

def rag_answer(question: str) -> str:
    if not POLICY_DOCS:
        # No docs uploaded – use GenAI directly
        prompt = f"""You are a helpful HR policy chatbot for TalentAI Solutions.
Answer this recruiter question clearly and concisely:

{question}"""
        return genai_call(prompt)

    # Retrieve most relevant chunk
    corpus    = POLICY_DOCS + [nlp.clean_text(question)]
    vect      = TfidfVectorizer(stop_words='english')
    mat       = vect.fit_transform(corpus)
    q_vec     = mat[-1]
    doc_mat   = mat[:-1]
    sims      = cosine_similarity(q_vec, doc_mat).flatten()
    best_idx  = int(np.argmax(sims))
    context   = POLICY_DOCS[best_idx][:1500]

    prompt = f"""You are an HR policy chatbot. Use the following company policy context to answer the question.

Context:
{context}

Question: {question}

Answer concisely and helpfully."""
    return genai_call(prompt)


# ─────────────────────────────────────────────────────────────
# SIDEBAR – Job Management
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.markdown("## 🤖 TalentAI Solutions")
    st.markdown("---")

    st.markdown("### ➕ Add Job Role")
    new_job = st.text_input("Job Title", key="new_job")
    new_jd  = st.text_area("Job Description", height=120, key="new_jd")
    if st.button("Add Job", key="add_job"):
        if new_job and new_jd:
            db.save_jobs(new_job, new_jd)
            st.success("✅ Job added!")
        else:
            st.warning("Enter both title and description.")

    st.markdown("---")
    st.markdown("### ⚙️ Quick Actions")
    if st.button("🗑 Delete All Resumes"):
        db.delete_db()
        st.warning("All resumes deleted.")


# ─────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📄 Resume Screener",
    "🤖 AI Assistant",
    "💬 RAG Chatbot",
    "📊 Dashboard",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 – HOME
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 🚀 TalentAI Solutions")
    st.markdown("### AI-Powered Smart Hiring & Candidate Intelligence Platform")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("📄", "Resume Screening", "Upload & Score"),
        ("🤖", "AI Assistant",     "GenAI Insights"),
        ("💬", "RAG Chatbot",      "Policy Q&A"),
        ("📊", "Dashboard",        "Analytics"),
    ]
    for col, (icon, name, desc) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2>{icon}</h2>
                <p><strong>{name}</strong><br>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### How it works
    1. **Add a Job Role** in the sidebar with a job description.
    2. **Upload resumes** in the *Resume Screener* tab (PDF, CSV, or TXT).
    3. Get an instant **match score** and **AI shortlisting prediction**.
    4. Use the **AI Assistant** for resume summaries, interview questions, and feedback.
    5. Ask the **RAG Chatbot** about company hiring policies.
    6. View **analytics and trends** in the Dashboard.
    """)

    # Model status
    st.markdown("---")
    st.markdown("### System Status")
    c1, c2 = st.columns(2)
    with c1:
        status = "✅ Loaded" if 'model' in artefacts else "⚠️ Not trained yet"
        st.metric("ML Model", status)
    with c2:
        api_status = "✅ Configured" if os.getenv("GOOGLE_API_KEY") else "⚠️ Not set"
        st.metric("Google AI API", api_status)


# ══════════════════════════════════════════════════════════════
# TAB 2 – RESUME SCREENER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📄 Resume Screener")

    jobs = db.get_jobs()
    if not jobs:
        st.warning("⚠️ Please add a Job Role from the sidebar first.")
        st.stop()

    job_dict = {f"{code} – {job}": code for code, job in jobs}
    selected_display  = st.selectbox("🎯 Select Job Role to Screen against", list(job_dict.keys()), key="screener_job")
    selected_job_code = job_dict[selected_display]
    job_description   = db.get_job_description(selected_job_code)

    st.markdown("---")
    with st.expander("📋 Review Selected Job Description", expanded=False):
        st.write(job_description)

    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF / CSV / TXT)",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
        key="resume_uploader"
    )

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            resume_text = text_extractor.extract_data(uploaded_file)
            score       = compute_match_score(resume_text, job_description)
            skills      = nlp.extract_skills(resume_text)
            pred        = ml_predict(resume_text, score)

            results.append({
                "file"       : uploaded_file.name,
                "score"      : score,
                "prediction" : pred["label"],
                "confidence" : pred["confidence"],
                "skills"     : skills,
                "text"       : resume_text,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        for r in results:
            badge_cls  = "score-high" if r["score"] >= 40 else "score-medium" if r["score"] >= 20 else "score-low"
            pill_cls   = "pill-shortlisted" if r["prediction"] == "Shortlisted" else "pill-rejected"
            conf_str   = f" ({r['confidence']}%)" if r['confidence'] else ""

            with st.expander(f"📄 {r['file']}  —  Score: {r['score']:.1f}%", expanded=False):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>{r['score']:.1f}</h2>
                        <p>Match Score (%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Prediction:** <span class='{pill_cls}'>{r['prediction']}{conf_str}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"{'✅ Model-based' if artefacts.get('model') else '📏 Score-based (rule)'}")

                with col_b:
                    if r["skills"]:
                        st.markdown("**Extracted Skills:**")
                        tags = "".join([f"<span class='skill-tag'>{s}</span>" for s in r["skills"]])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.write("_No matching skills detected_")

                with st.expander("Raw Resume Text"):
                    st.text_area("", r["text"], height=200, key=f"raw_{r['file']}")

        # Save button
        if st.button("💾 Save All to Database"):
            for r in results:
                db.save_to_db(
                    r["file"], r["text"], "", "", "", "",
                    "", ", ".join(r["skills"]), "", "", "",
                    selected_job_code
                )
            st.success("✅ All resumes saved!")

        # Quick summary table
        st.markdown("---")
        st.markdown("### 📊 Summary")
        summary_df = pd.DataFrame([{
            "File"       : r["file"],
            "Score (%)"  : r["score"],
            "Decision"   : r["prediction"],
            "Skills Found": len(r["skills"]),
        } for r in results])
        st.dataframe(summary_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 – AI ASSISTANT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 AI Recruiter Assistant")
    st.markdown("Upload a resume and let AI generate insights.")

    ai_file = st.file_uploader(
        "Upload Resume for AI Analysis",
        type=["pdf", "csv", "txt"],
        key="ai_uploader"
    )
    jobs = db.get_jobs()
    if jobs:
        job_dict = {f"{code} – {job}": code for code, job in jobs}
        selected_ai_job = st.selectbox("🎯 Target Job Role Context (Optional)", ["None"] + list(job_dict.keys()), key="ai_job")
        ai_jd = db.get_job_description(job_dict[selected_ai_job]) if selected_ai_job != "None" else ""
    else:
        ai_jd = ""

    if ai_file:
        resume_text = text_extractor.extract_data(ai_file)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📝 Summarize Resume"):
                with st.spinner("Generating summary …"):
                    prompt = f"""You are an expert recruiter. Summarize the following resume in 5-6 bullet points.
Highlight: key skills, total experience, education, and standout achievements.

Resume:
{resume_text[:3000]}"""
                    st.session_state["ai_summary"] = genai_call(prompt)

        with col2:
            if st.button("❓ Interview Questions"):
                with st.spinner("Generating questions …"):
                    jd_part = f"\nJob Description:\n{ai_jd[:500]}" if ai_jd else ""
                    prompt = f"""You are an expert technical interviewer.
Based on this resume, generate 8 targeted interview questions (mix of technical and behavioural).{jd_part}

Resume:
{resume_text[:2500]}"""
                    st.session_state["ai_questions"] = genai_call(prompt)

        with col3:
            if st.button("💬 Generate Feedback"):
                with st.spinner("Generating feedback …"):
                    jd_part = f"\nJob Description:\n{ai_jd[:500]}" if ai_jd else ""
                    prompt = f"""You are a senior recruiter. Provide structured candidate feedback:
1. Strengths (3 points)
2. Weaknesses / Gaps (3 points)
3. Overall Recommendation (Shortlist / Hold / Reject with justification){jd_part}

Resume:
{resume_text[:2500]}"""
                    st.session_state["ai_feedback"] = genai_call(prompt)

        # Display results
        for key, title in [
            ("ai_summary",   "📋 Resume Summary"),
            ("ai_questions", "❓ Interview Questions"),
            ("ai_feedback",  "💬 Candidate Feedback"),
        ]:
            if key in st.session_state:
                st.markdown(f"### {title}")
                st.markdown(st.session_state[key])
                st.markdown("---")

    # Prompt Engineering showcase
    with st.expander("🔬 Prompt Engineering Details"):
        st.markdown("""
        ### Version 1 — Basic prompt
        ```
        Summarize this resume.
        ```

        ### Version 2 — Structured prompt
        ```
        Summarize the following resume in 5-6 bullet points.
        Highlight: key skills, total experience, education, achievements.
        ```

        ### Version 3 — Role + Constraints + Format (current)
        ```
        You are an expert recruiter. Summarize the following resume in 5-6 bullet points.
        Highlight: key skills, total experience, education, and standout achievements.
        Resume: {resume_text}
        ```
        **Improvement**: Adding a role ("expert recruiter") and explicit output format
        significantly improves response quality and relevance.
        """)


# ══════════════════════════════════════════════════════════════
# TAB 4 – RAG CHATBOT
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 💬 RAG Recruiter Chatbot")
    st.markdown("Ask questions about hiring policies, role requirements, and candidate evaluation.")

    # Upload policy docs
    policy_file = st.file_uploader(
        "📂 Upload Hiring Policy Document (TXT/PDF) – optional",
        type=["txt", "pdf"],
        key="policy_uploader"
    )
    if policy_file:
        policy_text = text_extractor.extract_data(policy_file)
        add_policy_doc(policy_text)
        st.success(f"✅ Policy document loaded ({len(policy_text)} chars)")

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input(
        "Ask a question …",
        placeholder="What skills are required for a Python Developer role?",
        key="chat_input"
    )

    if st.button("Send 📨", key="chat_send") and user_question:
        with st.spinner("Thinking …"):
            answer = rag_answer(user_question)
        st.session_state.chat_history.append((user_question, answer))

    # Display chat history
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-user'>🧑 <strong>You:</strong> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-ai'>🤖 <strong>TalentAI:</strong> {a}</div>",  unsafe_allow_html=True)

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []

    # Sample questions
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - What skills are required for this role?
        - Why might a candidate be rejected?
        - What is the minimum experience required?
        - How is the shortlisting score calculated?
        - What is our interview process?
        """)


# ══════════════════════════════════════════════════════════════
# TAB 5 – DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📊 Hiring Analytics Dashboard")

    df_all = db.get_all_data()

    if df_all.empty:
        st.info("No resumes saved yet. Upload and save resumes in the Screener tab.")
    else:
        # KPI row
        total     = len(df_all)
        jobs_cnt  = len(db.get_jobs())
        kpi_row   = st.columns(4)
        kpis = [
            ("📄", total,    "Resumes Processed"),
            ("💼", jobs_cnt, "Active Job Roles"),
            ("✅", total,    "Candidates Evaluated"),
            ("📅", "Today",  "Last Updated"),
        ]
        for col, (icon, val, label) in zip(kpi_row, kpis):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <h2>{icon} {val}</h2><p>{label}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Load full processed dataset for charts
        proc_path = 'processed_resumes.csv'
        if os.path.exists(proc_path):
            df_proc = pd.read_csv(proc_path).fillna(0)

            row1 = st.columns(2)

            # Chart 1 – Category Distribution
            cat_col = next((c for c in ['Category','category'] if c in df_proc.columns), None)
            with row1[0]:
                if cat_col:
                    st.markdown("#### 🗂 Candidate Category Distribution")
                    cat_counts = df_proc[cat_col].value_counts().head(15)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#1e1e2e')
                    ax.set_facecolor('#1e1e2e')
                    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_counts)))
                    ax.barh(cat_counts.index, cat_counts.values, color=colors)
                    ax.set_xlabel('Count', color='white')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#444')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            # Chart 2 – Score Distribution
            with row1[1]:
                if 'score' in df_proc.columns:
                    st.markdown("#### 📈 Match Score Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#1e1e2e')
                    ax.set_facecolor('#1e1e2e')
                    ax.hist(df_proc['score'], bins=20, color='#7c83fd', edgecolor='white', alpha=0.8)
                    ax.axvline(20, color='#ff9800', linestyle='--', label='Shortlist threshold (20%)')
                    ax.set_xlabel('Score (%)', color='white')
                    ax.set_ylabel('Count', color='white')
                    ax.tick_params(colors='white')
                    ax.legend(facecolor='#333', labelcolor='white')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#444')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            row2 = st.columns(2)

            # Chart 3 – Experience vs Score
            with row2[0]:
                if 'experience' in df_proc.columns and 'score' in df_proc.columns:
                    st.markdown("#### ⏱ Experience vs Match Score")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#1e1e2e')
                    ax.set_facecolor('#1e1e2e')
                    sc = ax.scatter(
                        df_proc['experience'], df_proc['score'],
                        c=df_proc['score'], cmap='plasma', alpha=0.6, s=30, edgecolors='none'
                    )
                    plt.colorbar(sc, ax=ax, label='Score').ax.yaxis.label.set_color('white')
                    ax.set_xlabel('Experience (years)', color='white')
                    ax.set_ylabel('Match Score (%)', color='white')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#444')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            # Chart 4 – Skills Count Distribution
            with row2[1]:
                if 'skills_count' in df_proc.columns:
                    st.markdown("#### 🔧 Skills Count Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('#1e1e2e')
                    ax.set_facecolor('#1e1e2e')
                    ax.hist(df_proc['skills_count'], bins=15, color='#00e5ff', edgecolor='white', alpha=0.8)
                    ax.set_xlabel('Number of Skills', color='white')
                    ax.set_ylabel('Count', color='white')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#444')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            # Cluster plot if available
            if os.path.exists('cluster_plot.png'):
                st.markdown("---")
                st.markdown("#### 🔵 Candidate Clusters (K-Means + PCA)")
                st.image('cluster_plot.png', use_container_width=True)

            # Model comparison if available
            if os.path.exists('model_comparison.png'):
                st.markdown("---")
                st.markdown("#### 🏆 ML Model Comparison")
                st.image('model_comparison.png', use_container_width=True)

        else:
            st.info("Run `python data_preprocessing.py` to generate analytics data.")

        # Raw data viewer
        st.markdown("---")
        with st.expander("📋 View Saved Resume Database"):
            st.dataframe(df_all, use_container_width=True)