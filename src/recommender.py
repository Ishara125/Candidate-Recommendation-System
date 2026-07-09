import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Folder paths
PROCESSED_CV_DIR = "Data/processed_cvs"
PROCESSED_JOB_DIR = "Data/processed_jobs"
RESULTS_DIR = "results"
TOP_N = 5


def read_text_files(folder_path):
    """
    Read all .txt files from a given folder.
    Returns file names and text content.
    """
    file_names = []
    documents = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            file_names.append(filename)
            documents.append(text)

    return file_names, documents


def create_tfidf_vectors(cv_texts, job_texts):
    """
    Convert CVs and job descriptions into numerical TF-IDF vectors.
    Both CVs and job descriptions are fitted using the same vectorizer.
    """
    all_documents = cv_texts + job_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    cv_vectors = tfidf_matrix[:len(cv_texts)]
    job_vectors = tfidf_matrix[len(cv_texts):]

    return cv_vectors, job_vectors


def compute_similarity(job_vectors, cv_vectors):
    """
    Compute cosine similarity between job descriptions and candidate CVs.
    """
    similarity_matrix = cosine_similarity(job_vectors, cv_vectors)
    return similarity_matrix


def recommend_top_candidates(job_names, cv_names, similarity_matrix, top_n=5):
    """
    Rank candidates for each job description and recommend Top N candidates.
    """
    recommendations = []

    for job_index, job_name in enumerate(job_names):
        scores = similarity_matrix[job_index]

        ranked_indices = scores.argsort()[::-1][:top_n]

        for rank, cv_index in enumerate(ranked_indices, start=1):
            similarity_score = scores[cv_index]
            match_percentage = similarity_score * 100

            recommendations.append({
                "Job Description": job_name,
                "Rank": rank,
                "Candidate CV": cv_names[cv_index],
                "Similarity Score": round(float(similarity_score), 4),
                "Match Percentage": round(float(match_percentage), 2)
            })

    return pd.DataFrame(recommendations)


def display_recommendations(recommendations_df, top_n):
    """
    Display recommendation results in a clean grouped format.
    """
    print("\n" + "=" * 60)
    print(f"TOP {top_n} CANDIDATE RECOMMENDATIONS")
    print("=" * 60)

    job_names = recommendations_df["Job Description"].unique()

    for job_name in job_names:
        print(f"\nJob Description: {job_name}")
        print("-" * 60)

        job_results = recommendations_df[
            recommendations_df["Job Description"] == job_name
        ]

        for _, row in job_results.iterrows():
            print(
                f"Rank {row['Rank']}: "
                f"{row['Candidate CV']} | "
                f"Match: {row['Match Percentage']}%"
            )

    print("\n" + "=" * 60)


def save_results(recommendations_df):
    """
    Save recommendation results into a CSV file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_path = os.path.join(RESULTS_DIR, "recommendations.csv")
    recommendations_df.to_csv(output_path, index=False)

    print(f"\nResults saved successfully to: {output_path}")


def main():
    print("Starting Candidate Recommendation System...")

    print("\nReading processed CV files...")
    cv_names, cv_texts = read_text_files(PROCESSED_CV_DIR)

    print("Reading processed job description files...")
    job_names, job_texts = read_text_files(PROCESSED_JOB_DIR)

    print(f"\nNumber of candidate CVs found: {len(cv_names)}")
    print(f"Number of job descriptions found: {len(job_names)}")

    print("\nCreating TF-IDF vector representations...")
    cv_vectors, job_vectors = create_tfidf_vectors(cv_texts, job_texts)

    print("Computing cosine similarity scores...")
    similarity_matrix = compute_similarity(job_vectors, cv_vectors)

    print(f"Generating Top {TOP_N} candidate recommendations...")
    recommendations_df = recommend_top_candidates(
        job_names=job_names,
        cv_names=cv_names,
        similarity_matrix=similarity_matrix,
        top_n=TOP_N
    )

    display_recommendations(recommendations_df, top_n=TOP_N)
    save_results(recommendations_df)

    print("\nCandidate recommendation process completed successfully.")


if __name__ == "__main__":
    main()