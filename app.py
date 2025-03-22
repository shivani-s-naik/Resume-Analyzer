import streamlit as st
import fitz  # PyMuPDF for PDF handling
import re
import string
import nltk
import requests
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Download stopwords if not already present
nltk.download("punkt")
nltk.download("stopwords")

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens)

# Function to analyze resume using AI model
def analyze_resume(resume_text, job_category):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please provide feedback on resumes."),
            ("user", "Give feedback on this resume: {resume_text} for a job in {job_category}.")
        ]
    )
    llm = Ollama(model="llama3.2:latest")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain.invoke({"resume_text": resume_text, "job_category": job_category})

# Streamlit UI
st.title("📄 Resume & Job Fit Analyzer")

# Initialize session state variables
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "predicted_category" not in st.session_state:
    st.session_state.predicted_category = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    # Read the uploaded PDF file
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        resume_text = "\n".join(page.get_text("text") for page in doc)
    
    st.session_state.resume_text = resume_text  # Store text in session state
    
    st.subheader("📜 Extracted Resume Text:")
    st.text_area("Resume Content", resume_text, height=250)

    # Load ML model & vectorizer
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("resume_classifier.pkl")

    # Clean and vectorize text
    clean_resume_text = clean_text(resume_text)

    if st.button("🔍 Predict Resume Category"):
        try:
            # Make prediction
            input_vector = vectorizer.transform([clean_resume_text])
            predicted_category = model.predict(input_vector)[0]

            # Store in session state
            st.session_state.predicted_category = predicted_category

            # Display prediction
            st.success(f"✅ Predicted Resume Category: **{predicted_category}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# Show job analysis section only after prediction
if st.session_state.predicted_category:
    st.subheader("💼 Job Fit Analysis")
    st.write(f"Would you like to enter a specific job description or use the predicted category (**{st.session_state.predicted_category}**)?")

    user_choice = st.radio("Choose an option:", ["Use Predicted Category", "Enter Job Description"])

    if user_choice == "Enter Job Description":
        job_description = st.text_area("Enter the Job Description:")
        if job_description.strip():
            job_description_clean = clean_text(job_description)
            job_vector = vectorizer.transform([job_description_clean])
            predicted_category = model.predict(job_vector)[0]  # Update category
            st.session_state.predicted_category = predicted_category
            st.info(f"📌 Job category updated to: **{predicted_category}**")
        else:
            st.warning("⚠️ Please enter a job description!")

    if st.button("🚀 Analyze Resume Fit"):
        st.write("🔍 Analyzing your resume against the job category...")
        analysis = analyze_resume(st.session_state.resume_text, st.session_state.predicted_category)

        # Store result in session state to persist
        st.session_state.analysis_result = analysis

# Display analysis result after processing
if st.session_state.analysis_result:
    st.subheader("📢 AI Feedback on Your Resume:")
    st.write(st.session_state.analysis_result)

# Download extracted text
if st.session_state.resume_text:
    st.download_button(
        label="📥 Download Extracted Text",
        data=st.session_state.resume_text,
        file_name="resume_text.txt",
        mime="text/plain"
    )

