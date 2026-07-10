# Evaluation and Analysis

## Overview

This section evaluates the performance of the Candidate Recommendation System. The system-generated recommendations are compared with manually prepared expected results to measure how accurately the system recommends suitable candidates for each job description.

## Evaluation Dataset

A small evaluation dataset was created using the available job descriptions and candidate CV files. For each job description, expected suitable candidates were manually selected and stored as the ground truth dataset.

The evaluation dataset is stored in:

```text
Data/evaluation_ground_truth.csv
```

## System Recommendation Results

The system-generated recommendation results are stored in:

```text
results/recommendations.csv
```

The system recommends the Top 5 candidates for each job description based on TF-IDF vectorization and cosine similarity.

## Evaluation Metrics

The following metrics were used to evaluate the recommendation results:

### Precision

Precision measures how many of the recommended candidates were actually relevant.

### Recall

Recall measures how many of the expected relevant candidates were successfully recommended by the system.

### F1-score

F1-score gives a balanced value between Precision and Recall.

## Results

The evaluation script calculates the Precision, Recall, and F1-score for each job description. It also calculates the overall average Precision, Recall, and F1-score.

The evaluation results are saved in:

```text
results/evaluation_metrics.csv
results/evaluation_summary.txt
```

## Strengths

- The system can rank candidates based on similarity between CV content and job description content.
- It works well when candidate CVs and job descriptions contain similar skill keywords.
- The output is stored in CSV format, so it is easy to evaluate and analyze.

## Weaknesses

- The system depends heavily on the exact words used in CVs and job descriptions.
- TF-IDF may not understand deeper meaning or context.
- The evaluation dataset is small, so the results may not fully represent real-world performance.
- Some suitable candidates may be missed if their CVs use different wording for the same skills.

## Future Improvements

- Increase the number of candidate CVs and job descriptions.
- Use advanced NLP models such as BERT or Sentence Transformers.
- Add separate matching for skills, education level, and work experience.
- Improve the ground truth dataset by manually reviewing each CV and job description.