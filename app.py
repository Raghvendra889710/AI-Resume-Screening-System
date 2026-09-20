import streamlit as st
import pandas as pd
import numpy as np
import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK packages are downloaded
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

setup_nltk()

def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file stream."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

def preprocess_text(text):
    """Cleans text: lowercase, punctuation removal, stopwords removal, lemmatization."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

# UI Layout
st.set_page_config(page_title="AI Resume Screening System", layout="centered")
st.title("📄 AI Resume Screening System")
st.write("Upload candidate resumes (PDF format) and paste a job description to calculate match scores automatically.")

# Form Controls
job_description = st.text_area("1. Paste Job Description Here:", height=150)
uploaded_files = st.file_uploader("2. Upload PDF Resumes:", type=["pdf"], accept_multiple_files=True)
threshold = st.slider("3. Select Shortlist Threshold Score (%)", min_value=0, max_value=100, value=15)

if st.button("Screen Resumes"):
    if not job_description.strip():
        st.warning("Please enter a valid job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
    else:
        # Extract Text
        resumes_data = {}
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes_data[file.name] = text

        clean_jd = preprocess_text(job_description)
        candidate_names = list(resumes_data.keys())
        clean_resumes = [preprocess_text(text) for text in resumes_data.values()]

        # TF-IDF Matching
        corpus = [clean_jd] + clean_resumes
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Build DataFrame
        results_df = pd.DataFrame({
            "Candidate / File Name": candidate_names,
            "Match Score (%)": np.round(similarity_scores * 100, 2)
        }).sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

        shortlisted_df = results_df[results_df["Match Score (%)"] >= threshold]

        # Display Results
        st.subheader("All Evaluated Resumes")
        st.dataframe(results_df, use_container_width=True)

        st.subheader(f"Shortlisted Candidates (Score ≥ {threshold}%)")
        st.dataframe(shortlisted_df, use_container_width=True)

        # Plot bar chart
        st.subheader("Candidate Match Visualization")
        st.bar_chart(results_df.set_index("Candidate / File Name")["Match Score (%)"])
