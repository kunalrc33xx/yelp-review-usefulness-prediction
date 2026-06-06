# Yelp Review Usefulness Prediction

Predicting whether a Yelp review will be marked "useful" by other users. Built as part of a graduate-level data mining course at the University of Maryland (BUDT758T).

## Overview

This project frames review usefulness as a binary classification problem on 1.57 million Yelp reviews. Six ML models were trained and compared, with the goal of maximizing True Positive Rate at a False Positive Rate cap of 10% (TPR@FPR<=10%).

Best result: **LightGBM** achieved **TPR 0.6803 at FPR <= 10%** and **ROC-AUC 0.9040**.

## My Role - Data Engineer and Infrastructure

- Set up the full Kaggle notebook environment with memory-optimized data loading
- Engineered supplementary structured features alongside the TF-IDF and VADER pipelines
- Built a leakage-free feature pipeline using 3-fold stratified cross-validation
- Handled data at scale: 1.57M rows, 356 total features per sample

## Feature Engineering

| Feature Type | Count | Description |
|---|---|---|
| TF-IDF text features | 300 | Top unigrams/bigrams from review text |
| VADER sentiment features | 8 | Positive, negative, neutral, compound scores |
| Structured engineered features | 56 | Review length, word count, user history, business stats |
| Total | 356 | Combined sparse and dense feature matrix |

## Models Compared

- LightGBM (winner)
- XGBoost
- Random Forest
- HistGradientBoosting
- Logistic Regression
- Decision Tree

## Results

| Metric | Score |
|---|---|
| TPR at FPR <= 10% | 0.6803 |
| ROC-AUC | 0.9040 |

## Tech Stack

Python, LightGBM, XGBoost, scikit-learn, VADER (NLTK), pandas, scipy sparse matrices, Kaggle

## Dataset

Yelp Open Dataset - 1.57M reviews used for training and evaluation.# yelp-review-usefulness-prediction
ML pipeline to predict highly useful Yelp reviews usng LightGBM, TF-IDF, and VADER sentiment analysis on1.57M reviews
