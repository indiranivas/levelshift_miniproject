
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() or ""
    
    return text


def extract_text_from_csv(file):
    df = pd.read_csv(file)
    
    text = df.to_string(index=False)
    
    return text


def extract_text_from_txt(file):
    return file.read().decode("utf-8")


def extract_data(uploaded_file):
    if uploaded_file is not None:
        
        file_type = uploaded_file.name.split(".")[-1]
        
        if file_type == "pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
            
        elif file_type == "csv":
            extracted_text = extract_text_from_csv(uploaded_file)
            
        elif file_type == "txt":
            extracted_text = extract_text_from_txt(uploaded_file)
            
        else:
            st.error("Unsupported file type")
            extracted_text = ""
    return extracted_text