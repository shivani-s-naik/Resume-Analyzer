# 📝 Resume Analyzer with LangChain & Streamlit

A simple yet powerful resume analysis tool that leverages cutting-edge NLP techniques and machine learning to analyze resumes and provide insights.

## ✨ Features
- **Resume Information Extraction:** Automatically extracts essential details from uploaded resumes.
- **Resume Classification:** Uses TF-IDF and a RandomForest classifier trained on a Kaggle dataset to categorize resumes.
- **Resume-Job Fit Analysis:** Utilizes LLaMA 3.2 via Ollama to assess the suitability of resumes for given job descriptions.
- **Feedback Generation:** Provides actionable feedback based on the predicted resume category or a custom job description.

---

## 🚀 Tech Stack
- **LangChain:** For chaining and orchestrating LLM tasks.
- **Streamlit:** Web-based UI for user interaction.
- **LLaMA 3.2 via Ollama:** Natural language understanding and generation.
- **TF-IDF & RandomForest:** Resume classification.
- **Python:** Core programming language.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resume-analyzer.git
   cd resume-analyzer

2. Set up a virtual environment:
   python3 -m venv env
   source env/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. LLM Setup
   Download and install OLLAMA https://ollama.com/
   Depending on your system's RAM, select and download the suitable model https://github.com/ollama/ollama

## 📝 Usage
1. Run the Streamlit application:
   streamlit run app.py

2. Open your browser at http://localhost:8501.

3. Upload a resume and select a job description to analyze the fit
