import streamlit as st
import db
import text_extractor
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



db.create_db()
st.set_page_config(page_title="TalentAI", layout="wide")

st.title("🤖 TalentAI Solutions")


st.sidebar.header("➕ Add Job")

new_job = st.sidebar.text_input("Job Title")
new_jd = st.sidebar.text_area("Job Description")

if st.sidebar.button("Add Job"):
    if new_job and new_jd:
        db.save_jobs(new_job, new_jd)
        st.sidebar.success("Job added!")
    else:
        st.sidebar.warning("Enter job details")



jobs = db.get_jobs()

if not jobs:
    st.warning("No jobs available. Add from sidebar.")
    st.stop()

job_dict = {f"{code} - {job}": code for code, job in jobs}

selected_display = st.selectbox("Select Job Role", list(job_dict.keys()))
selected_job_code = job_dict[selected_display]

# Load JD
job_description = db.get_job_description(selected_job_code)


uploaded_files = st.file_uploader(
    "Upload Resume Folder (Select Multiple Files)",
    type=["pdf", "csv", "txt"],
    accept_multiple_files=True
)






if uploaded_files:
    for uploaded_file in uploaded_files:
        extracted_text = text_extractor.extract_data(uploaded_file)

        # Extract fields from CSV if applicable
        name = ""
        email = ""
        phone = ""
        education = ""
        experience_years = ""
        skills = ""
        projects = ""
        certifications = ""
        summary = ""
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if not df.empty:
                row = df.iloc[0]  # Assume first row is the candidate
                name = row.get('Name', '')
                email = row.get('Email', '')
                phone = row.get('Phone', '')
                education = row.get('Education', '')
                experience_years = row.get('Experience_Years', '')
                skills = row.get('Skills', '')
                projects = row.get('Projects', '')
                certifications = row.get('Certifications', '')
                summary = row.get('Summary', '')
            # Reset file pointer for text extraction
            uploaded_file.seek(0)

        st.subheader(f"📄 Extracted Resume: {uploaded_file.name}")
        st.text_area(f"Output for {uploaded_file.name}", extracted_text, height=300)

    if st.button("Save All"):
        for uploaded_file in uploaded_files:
            extracted_text = text_extractor.extract_data(uploaded_file)
            name = ""
            email = ""
            phone = ""
            education = ""
            experience_years = ""
            skills = ""
            projects = ""
            certifications = ""
            summary = ""
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if not df.empty:
                    row = df.iloc[0]
                    name = row.get('Name', '')
                    email = row.get('Email', '')
                    phone = row.get('Phone', '')
                    education = row.get('Education', '')
                    experience_years = row.get('Experience_Years', '')
                    skills = row.get('Skills', '')
                    projects = row.get('Projects', '')
                    certifications = row.get('Certifications', '')
                    summary = row.get('Summary', '')
                uploaded_file.seek(0)
            db.save_to_db(
                uploaded_file.name,
                extracted_text,
                name,
                email,
                phone,
                education,
                experience_years,
                skills,
                projects,
                certifications,
                summary,
                selected_job_code
            )
        st.success("All files saved successfully!")



if st.button("📊 Show Saved Data"):
    df = db.get_all_data()
    st.dataframe(df)


if st.button("📂 Load from formatted_resumes Folder"):
    folder_path = "formatted_resumes"
    if os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)
            if not df.empty:
                row = df.iloc[0]
                name = row.get('Name', '')
                email = row.get('Email', '')
                phone = row.get('Phone', '')
                education = row.get('Education', '')
                experience_years = row.get('Experience_Years', '')
                skills = row.get('Skills', '')
                projects = row.get('Projects', '')
                certifications = row.get('Certifications', '')
                summary = row.get('Summary', '')
                content = df.to_string(index=False)
                db.save_to_db(csv_file, content, name, email, phone, education, experience_years, skills, projects, certifications, summary, selected_job_code)
        st.success(f"Loaded {len(csv_files)} resumes from folder!")
    else:
        st.error("Folder not found.")





if st.button("�🗑 Delete All Data"):
    db.delete_db()
    st.warning("All data deleted")