import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Attempt to download NLTK data silently (works offline if already cached)
for _pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass  # No internet – use whatever is already installed

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set()  # fallback: no stopword filtering

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize('test')  # warm-up check
except LookupError:
    lemmatizer = None  # fallback: no lemmatization


def _lemmatize(word: str) -> str:
    if lemmatizer is None:
        return word
    return lemmatizer.lemmatize(word)

# ─────────────────────────────────────────────
# Predefined skill keywords for extraction
# ─────────────────────────────────────────────
SKILL_KEYWORDS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "sql", "nosql", "html", "css", "php", "ruby", "swift", "kotlin", "go",
    # ML / AI
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    # Data
    "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
    "data analysis", "data science", "data engineering", "big data",
    "hadoop", "spark", "hive", "kafka",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "ci/cd", "jenkins", "terraform", "linux",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
    "sql server", "dynamodb",
    # Web Frameworks
    "flask", "fastapi", "django", "react", "angular", "vue", "node.js",
    "express", "spring", "rest api",
    # HR / Business Skills
    "recruitment", "hr", "human resources", "payroll", "fmla", "hris",
    "employee relations", "training", "performance management",
    "benefits administration", "compliance", "onboarding",
    # Soft Skills
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "time management", "analytical",
}


def clean_text(text: str) -> str:
    """Full NLP cleaning: lowercase → remove special chars → tokenize → stopword removal → lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [
        _lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(words)


def extract_section(text: str, section_name: str) -> str:
    """
    Extract a named section from resume text.
    Looks for section_name header, returns text until next header.
    """
    if not isinstance(text, str):
        return ""
    # Common section boundary markers
    pattern = rf'(?i){re.escape(section_name)}[\s\S]*?(?=\n[A-Z][A-Z\s]{{3,}}:|$)'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    return ""


def extract_experience_section(text: str) -> str:
    """Return only the Experience section of a resume."""
    section = extract_section(text, "Experience")
    if not section:
        section = extract_section(text, "Work Experience")
    if not section:
        section = extract_section(text, "Work History")
    if not section:
        section = extract_section(text, "Professional Experience")
    return section if section else text  # fall back to full text


def extract_skills(text: str) -> list:
    """Extract skills from text by matching against SKILL_KEYWORDS."""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        # Use word-boundary match for single-word skills
        if len(skill.split()) == 1:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                found.append(skill)
        else:
            if skill in text_lower:
                found.append(skill)
    return sorted(set(found))


def extract_education(text: str) -> str:
    """Return only the Education section of a resume."""
    section = extract_section(text, "Education")
    if not section:
        section = extract_section(text, "Education and Training")
    return section


def count_skills(text: str) -> int:
    """Return the number of identified skills in a text."""
    return len(extract_skills(text))