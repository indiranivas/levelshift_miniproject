# 🚀 AI-Powered Smart Hiring & Candidate Intelligence Platform

## 📌 Overview

This project is an end-to-end AI/ML system designed to automate and enhance the recruitment process. It processes resumes, matches candidates with job descriptions, and predicts candidate suitability using machine learning techniques.

The system helps recruiters save time, improve hiring accuracy, and gain insights into candidate profiles.

---

## 🎯 Objectives

* Automate resume screening
* Match candidates with job descriptions
* Extract key information (skills, experience)
* Predict candidate selection (Shortlist/Reject)
* Provide a user-friendly interface for recruiters

---

## 🏗️ Project Architecture

```
Resume → NLP Processing → Feature Engineering →
Matching Engine → ML Model → Output → Streamlit UI
```

---

## 📂 Dataset

* Resume Dataset (CSV format)

  * `Resume_str`: Resume text
  * `Category`: Job role

---

## ⚙️ Features

### 🔹 1. Data Preprocessing

* Removed missing values and duplicates
* Cleaned text using NLP techniques
* Standardized resume content

---

### 🔹 2. NLP Processing

* Tokenization
* Stopword removal
* Lemmatization
* Cleaned resume text generation

---

### 🔹 3. Feature Engineering

* Skills extraction (dictionary + regex)
* Experience extraction (date parsing)
* Skills count
* Encoded job categories
* Scaled numerical features

---

### 🔹 4. Matching Engine

* TF-IDF vectorization
* Cosine similarity
* Match score (0–100)

---

### 🔹 5. Machine Learning Model

* Models used:

  * Logistic Regression
  * Random Forest
  * SVM
* Predicts:
  👉 Candidate Shortlisted / Rejected

---

### 🔹 6. Advanced Enhancements

* BERT-based skill extraction (optional)
* Improved matching using n-grams
* Combined feature scoring

---

### 🔹 7. Streamlit Web App

* Upload resume
* Enter job description
* Display:

  * Match score
  * Extracted skills
  * Predicted result

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Transformers (BERT)
* Streamlit

---

## 📊 Workflow

1. Load and clean dataset
2. Apply NLP preprocessing
3. Extract features (skills, experience)
4. Compute match score
5. Train ML model
6. Predict candidate outcome
7. Display results in UI

---

## 📈 Example Output

* Match Score: 78.5%
* Skills: Python, SQL, Excel
* Prediction: Shortlisted

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install pandas numpy scikit-learn nltk streamlit transformers
```

### 2. Run the application

```
streamlit run app.py
```

---

## 📌 Future Improvements

* RAG-based chatbot for recruiter queries
* Better skill extraction using fine-tuned models
* Integration with real-time hiring systems
* Cloud deployment

---

## 👨‍💻 Author

Developed as part of AI/ML project for Smart Hiring System.

---

## ⭐ Conclusion

This project demonstrates how AI and machine learning can transform traditional hiring processes into intelligent, automated systems.
