# Preprocessing Pipeline

## Data Collection

Sample candidate CVs were collected and stored locally in the raw CV folder. Raw CV PDF files were not uploaded to GitHub to protect candidate privacy.

Job descriptions were created for the following roles:

- QA Engineer
- Full Stack Developer
- Frontend Developer
- Business Analyst

The job descriptions were stored as `.txt` files in the raw job descriptions folder.

## Preprocessing Steps

The preprocessing script performs the following steps:

1. Extracts text from candidate CV PDF files.
2. Reads job description text files.
3. Converts all text to lowercase.
4. Removes numbers, punctuation, and special characters.
5. Tokenizes the text by splitting it into words.
6. Removes English stopwords.
7. Applies lemmatization to reduce words to their base form.
8. Saves the cleaned CV text files into the processed CV folder.
9. Saves the cleaned job description text files into the processed job folder.

## Output

The final cleaned text files are ready for vectorization using TF-IDF, Word2Vec, BERT, or Sentence Transformers.