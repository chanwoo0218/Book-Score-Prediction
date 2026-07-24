# Book Score Prediction

> **사용자 특성과 도서 메타데이터를 결합해 사용자가 특정 도서에 부여할 1-10점 평점을 예측하고, 추천 데이터의 희소성·콜드스타트·분할 문제를 고려한 회귀 프로젝트입니다.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/) ![CatBoost](https://img.shields.io/badge/Final%20Model-CatBoost-yellow) ![Task](https://img.shields.io/badge/Task-Rating%20Regression-purple)

## At a Glance

| Item | Description |
|---|---|
| Project type | Recommender-system regression |
| Period | 2026.01.06 - 2026.02.25 |
| Activity | DF winter short-term program |
| Processed data | 131,865 explicit ratings |
| Users / books | 11,452 users / 16,930 books |
| Target | `Book-Rating` from 1 to 10 |
| Evaluation scenario | Warm-start user holdout |
| Final model | CatBoost with categorical, text, and numeric features |
| Core metrics | RMSE, MAE, R² |

## Problem

Rating prediction differs from ordinary tabular regression because each observation represents an interaction between a user and an item. The data is sparse, high-cardinality IDs carry behavioral information, and evaluation quality depends heavily on whether users and books are seen during training.

This project asks:

> **Can user information and book metadata improve rating prediction beyond a simple per-user mean baseline?**

The project uses only explicit ratings from 1 to 10 and evaluates the model in a warm-start setting where test users have historical observations in the training data.

## Dataset

The repository includes the personally processed dataset at:

```text
data/processed/df_final1.csv
```

Dataset summary:

| Item | Value |
|---|---:|
| Rows | 131,865 |
| Columns | 10 |
| Unique users | 11,452 |
| Unique books | 16,930 |
| Rating range | 1-10 |
| Mean rating | 7.77 |

Columns:

- `ID`
- `User-ID`
- `Book-ID`
- `Book-Rating`
- `Age`
- `Book-Title`
- `Book-Author`
- `Year-Of-Publication`
- `Publisher`
- `Location_country`

## Data Processing

1. Inspected data types, summary statistics, and missing values.
2. Normalized title, author, publisher, and location text.
3. Extracted country information from the location field.
4. Treated missing and invalid publication years.
5. Removed rating `0`, which represents unobserved or implicit feedback rather than a true zero-star preference.
6. Repeatedly applied **3×3 p-core filtering** so each retained user and book has at least three interactions.
7. Constructed a user-based warm-start train/validation/test split.

## Key Decisions

### Why remove rating 0?

A zero in this dataset does not have the same meaning as an explicit rating between 1 and 10. Keeping it as a numeric target would make the model learn a distorted rating scale.

### Why p-core filtering?

Users or books with only one or two ratings provide too little information for stable behavioral estimation. Iterative filtering reduces extreme sparsity and makes the evaluation scenario more coherent.

### Why a user-based warm-start holdout?

A random row split may accidentally produce an unrealistic mix of known and unknown entities. The selected split preserves part of each eligible user's history for training and evaluates prediction on later held-out interactions from known users.

### Why compare against UserMean?

A personalized mean is a strong and interpretable recommendation baseline. A complex model is useful only when it improves on this simple behavioral signal.

## Modeling

### Linear / Ridge experiment

- Text fields: TF-IDF → Truncated SVD
- Numeric fields: missing-value processing and standardization
- High-cardinality identifiers: encoded for linear modeling
- Linear Regression and Ridge compared through cross-validation

### Final CatBoost experiment

- Categorical features: `User-ID`, `Book-ID`, `Year_Decade`
- Text-derived features: TF-IDF with up to 3,000 terms per field, reduced to 30 SVD dimensions
- Numeric features: age, publication year, text lengths, and punctuation counts
- Early stopping on the validation set
- Retraining on train + validation using the selected iteration

## Results

All results below use the documented warm-start test split.

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| UserMean baseline | 1.6412 | 1.2406 | - |
| Ridge final | 1.55868 | 1.20898 | 0.24077 |
| **CatBoost final** | **1.55290** | **1.19926** | **0.24639** |

Compared with UserMean, the final CatBoost model reduced:

- RMSE by approximately `0.0883`
- MAE by approximately `0.0413`

The gain is moderate, which is expected in sparse rating prediction where user-specific averages already form a strong baseline.

## Project Work

- Interpreted rating 0 from a recommendation-domain perspective and restricted the target to explicit feedback.
- Designed text normalization, country extraction, missing-value handling, and p-core filtering.
- Constructed a warm-start user-based evaluation split.
- Converted heterogeneous user and book metadata into model-ready numeric, categorical, and text-derived features.
- Compared linear, Ridge, UserMean, and CatBoost approaches using RMSE and MAE.
- Evaluated not only absolute performance but also improvement over a personalized baseline.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── README.md
│   ├── processed
│   │   └── df_final1.csv
│   └── sample
│       └── df_final1_sample.csv
├── docs
│   ├── README.md
│   └── presentation_summary.md
└── src
    ├── preprocess_submission.py
    ├── train.py
    └── train_catboost_submission.py
```

## How to Run

```bash
git clone https://github.com/chanwoo0218/Book-Score-Prediction.git
cd Book-Score-Prediction
pip install -r requirements.txt
```

Train with the included processed dataset:

```bash
python src/train_catboost_submission.py \
  --data data/processed/df_final1.csv
```

To rebuild the processed data from a compatible raw training file:

```bash
python src/preprocess_submission.py \
  --input data/raw/train.csv \
  --output data/processed/df_final1.csv
```

## Limitations

- The evaluation is warm-start and does not measure performance for completely unseen users or books.
- The model predicts explicit ratings but does not directly optimize ranking quality or recommendation diversity.
- Text and ID features may encode popularity patterns that change over time.
- The current split does not explicitly model chronological changes in user preference.

## Future Work

- Separate cold-start evaluation for unseen users and books
- Ranking metrics such as NDCG, Recall@K, and MAP@K
- Matrix-factorization and neural collaborative-filtering baselines
- Hybrid ranking model combining collaborative and content signals
- Calibration and uncertainty analysis for predicted ratings

## Portfolio

The Korean-language project explanation and learning reflections are available on the [Notion portfolio page](https://app.notion.com/p/cd282d8994c283e9be85812f29e72d85).