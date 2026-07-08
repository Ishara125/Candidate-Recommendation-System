import os
import re
import fitz  # PyMuPDF
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Folder paths
RAW_CV_DIR = "Data/raw_cvs"
RAW_JOB_DIR = "Data/raw_jobs"
PROCESSED_CV_DIR = "Data/processed_cvs"
PROCESSED_JOB_DIR = "Data/processed_jobs"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""

    document = fitz.open(pdf_path)

    for page in document:
        text += page.get_text()

    document.close()
    return text


def clean_text(text):
    """Clean, tokenize, remove stopwords, and lemmatize text."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def process_cv_pdfs(input_dir, output_dir):
    """Process raw CV PDF files and save cleaned TXT files."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace(".pdf", ".txt").replace(".PDF", ".txt")
            output_path = os.path.join(output_dir, output_filename)

            raw_text = extract_text_from_pdf(input_path)
            cleaned_text = clean_text(raw_text)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(cleaned_text)

            print(f"Processed CV: {filename}")


def process_job_txt_files(input_dir, output_dir):
    """Process raw job description TXT files and save cleaned TXT files."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            cleaned_text = clean_text(raw_text)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(cleaned_text)

            print(f"Processed Job: {filename}")


process_cv_pdfs(RAW_CV_DIR, PROCESSED_CV_DIR)
process_job_txt_files(RAW_JOB_DIR, PROCESSED_JOB_DIR)

print("Preprocessing completed successfully.")