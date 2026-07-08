# Candidate Recommendation System

## Project Overview

This project is a Candidate Recommendation System that matches candidate CVs with suitable job roles based on skills, experience, education background, and CV-job description similarity.

The system preprocesses CVs and job descriptions, converts the text into vector representations, calculates similarity scores, and recommends the best-matching candidates for each job role.

## Main Features

- Read candidate CVs
- Read job descriptions
- Clean and preprocess text data
- Convert text into vector format
- Calculate similarity between CVs and job descriptions
- Rank and recommend top candidates

## Project Structure

```text
Data/
  raw_cvs/
  raw_jobs/
  processed_cvs/
  processed_jobs/

src/
docs/
results/
README.md