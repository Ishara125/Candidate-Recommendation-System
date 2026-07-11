import pandas as pd
from pathlib import Path


# Top candidates to evaluate from recommendation results
TOP_K = 5

# File paths used in the evaluation process
GROUND_TRUTH_PATH = Path("Data/evaluation_ground_truth.csv")
RECOMMENDATIONS_PATH = Path("results/recommendations.csv")
EVALUATION_OUTPUT_PATH = Path("results/evaluation_metrics.csv")
SUMMARY_OUTPUT_PATH = Path("results/evaluation_summary.txt")


def calculate_metrics(expected_candidates, predicted_candidates):
    """
    Calculate Precision, Recall, and F1-score.

    expected_candidates: candidates manually selected as correct matches
    predicted_candidates: candidates recommended by the system
    """

    # Count candidates that appear in both expected and predicted lists
    true_positives = len(expected_candidates.intersection(predicted_candidates))

    # Precision = correctly recommended candidates / total recommended candidates
    if len(predicted_candidates) > 0:
        precision = true_positives / len(predicted_candidates)
    else:
        precision = 0

    # Recall = correctly recommended candidates / total expected correct candidates
    if len(expected_candidates) > 0:
        recall = true_positives / len(expected_candidates)
    else:
        recall = 0

    # F1-score gives a balanced value between precision and recall
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score, true_positives


def main():
    print("Starting evaluation and analysis...")

    # Read manually created expected results
    ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)

    # Read system generated recommendation results
    recommendations_df = pd.read_csv(RECOMMENDATIONS_PATH)

    evaluation_results = []

    # Evaluate recommendations for each job description
    for _, row in ground_truth_df.iterrows():
        job_description = row["Job Description"]

        # Convert expected candidate list into a set
        expected_candidates = set(str(row["Expected Candidate CVs"]).split("|"))

        # Get system recommendations only for the selected job description
        job_recommendations = recommendations_df[
            recommendations_df["Job Description"] == job_description
        ]

        # Select Top-K recommendations based on rank
        top_recommendations = job_recommendations.sort_values("Rank").head(TOP_K)

        # Convert predicted candidate list into a set
        predicted_candidates = set(top_recommendations["Candidate CV"].astype(str))

        # Calculate evaluation metrics
        precision, recall, f1_score, true_positives = calculate_metrics(
            expected_candidates,
            predicted_candidates
        )

        evaluation_results.append({
            "Job Description": job_description,
            "Expected Candidate CVs": "|".join(expected_candidates),
            "Predicted Candidate CVs": "|".join(predicted_candidates),
            "True Positives": true_positives,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-score": round(f1_score, 3)
        })

    # Convert results into a DataFrame
    evaluation_df = pd.DataFrame(evaluation_results)

    # Save per-job evaluation results
    evaluation_df.to_csv(EVALUATION_OUTPUT_PATH, index=False)

    # Calculate overall average metrics
    average_precision = round(evaluation_df["Precision"].mean(), 3)
    average_recall = round(evaluation_df["Recall"].mean(), 3)
    average_f1_score = round(evaluation_df["F1-score"].mean(), 3)

    # Save summary analysis as a text file
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("Evaluation and Analysis Summary\n")
        file.write("================================\n\n")

        file.write(f"Top-K value used: {TOP_K}\n\n")

        file.write("Overall Evaluation Results\n")
        file.write("--------------------------\n")
        file.write(f"Average Precision: {average_precision}\n")
        file.write(f"Average Recall: {average_recall}\n")
        file.write(f"Average F1-score: {average_f1_score}\n\n")

        file.write("Strengths\n")
        file.write("---------\n")
        file.write("- The system can rank candidates based on CV and job description similarity.\n")
        file.write("- The system works well when CVs and job descriptions contain similar skill keywords.\n")
        file.write("- The recommendation output is easy to evaluate using Precision, Recall, and F1-score.\n\n")

        file.write("Weaknesses\n")
        file.write("----------\n")
        file.write("- The system depends heavily on the words used in CVs and job descriptions.\n")
        file.write("- TF-IDF may not understand the deeper meaning of skills or experience.\n")
        file.write("- The evaluation dataset is small, so the result may not represent real-world accuracy fully.\n\n")

        file.write("Future Improvements\n")
        file.write("-------------------\n")
        file.write("- Increase the number of CVs and job descriptions.\n")
        file.write("- Use advanced NLP models such as BERT or Sentence Transformers.\n")
        file.write("- Add separate matching for skills, education level, and work experience.\n")

    print("Evaluation completed successfully.")
    print("\nPer-job evaluation results:")
    print(evaluation_df)

    print("\nOverall Results:")
    print("Average Precision:", average_precision)
    print("Average Recall:", average_recall)
    print("Average F1-score:", average_f1_score)


if __name__ == "__main__":
    main()