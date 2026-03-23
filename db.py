import pandas as pd
import os
from datetime import datetime

JOBS_CSV = "jobs.csv"
RESUMES_CSV = "resumes.csv"

def create_db():
    if not os.path.exists(JOBS_CSV):
        df = pd.DataFrame(columns=['id', 'job_code', 'job', 'job_description', 'created_at'])
        df.to_csv(JOBS_CSV, index=False)
    
    if not os.path.exists(RESUMES_CSV):
        df = pd.DataFrame(columns=['id', 'file_name', 'content', 'name', 'email', 'phone', 'education', 'experience_years', 'skills', 'projects', 'certifications', 'summary', 'job_code', 'created_at'])
        df.to_csv(RESUMES_CSV, index=False)



def generate_job_code(job):
    words = job.split()
    prefix = "".join([w[0].upper() for w in words])

    if os.path.exists(JOBS_CSV):
        df = pd.read_csv(JOBS_CSV)
        count = len(df[df['job'] == job]) + 1
    else:
        count = 1

    return f"{prefix}_{str(count).zfill(3)}"



def save_jobs(job, job_description):
    job_code = generate_job_code(job)
    created_at = datetime.now().isoformat()
    
    if os.path.exists(JOBS_CSV):
        df = pd.read_csv(JOBS_CSV)
        new_id = len(df) + 1
    else:
        df = pd.DataFrame()
        new_id = 1
    
    new_row = pd.DataFrame({
        'id': [new_id],
        'job_code': [job_code],
        'job': [job],
        'job_description': [job_description],
        'created_at': [created_at]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(JOBS_CSV, index=False)


def get_jobs():
    if os.path.exists(JOBS_CSV):
        df = pd.read_csv(JOBS_CSV)
        return list(df[['job_code', 'job']].itertuples(index=False, name=None))
    else:
        return []



def get_job_description(job_code):
    if os.path.exists(JOBS_CSV):
        df = pd.read_csv(JOBS_CSV)
        result = df[df['job_code'] == job_code]['job_description']
        return result.iloc[0] if not result.empty else ""
    else:
        return ""



def save_to_db(file_name, content, name, email, phone, education, experience_years, skills, projects, certifications, summary, job_code):
    created_at = datetime.now().isoformat()
    
    if os.path.exists(RESUMES_CSV):
        df = pd.read_csv(RESUMES_CSV)
        new_id = len(df) + 1
    else:
        df = pd.DataFrame()
        new_id = 1
    
    new_row = pd.DataFrame({
        'id': [new_id],
        'file_name': [file_name],
        'content': [content],
        'name': [name],
        'email': [email],
        'phone': [phone],
        'education': [education],
        'experience_years': [experience_years],
        'skills': [skills],
        'projects': [projects],
        'certifications': [certifications],
        'summary': [summary],
        'job_code': [job_code],
        'created_at': [created_at]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(RESUMES_CSV, index=False)

def get_all_data():
    if os.path.exists(RESUMES_CSV):
        return pd.read_csv(RESUMES_CSV)
    else:
        return pd.DataFrame(columns=['id', 'file_name', 'content', 'name', 'email', 'job_code', 'created_at'])


def delete_db():
    if os.path.exists(RESUMES_CSV):
        df = pd.DataFrame(columns=['id', 'file_name', 'content', 'name', 'email', 'phone', 'education', 'experience_years', 'skills', 'projects', 'certifications', 'summary', 'job_code', 'created_at'])
        df.to_csv(RESUMES_CSV, index=False)