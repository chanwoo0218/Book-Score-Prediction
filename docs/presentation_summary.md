# Presentation Summary

## Objective

Predict explicit book ratings from user and book metadata under a warm-start recommendation setting.

## Key preprocessing choices

- Remove rating `0`, treating it as implicit feedback rather than a 0-point score.
- Normalize text and extract country from user location.
- Remove invalid publication years.
- Apply iterative 3×3 user/item p-core filtering.
- Hold out a fraction of each eligible user's ratings so that train and test contain the same users.

## Model progression

1. Linear Regression with TF-IDF and TruncatedSVD features
2. Ridge Regression with tuned text dimensions
3. CatBoost with User-ID, Book-ID, decade, numeric metadata, and compressed text features
4. User-mean baseline on the identical test split

## Final reported metrics

| Model | RMSE | MAE |
|---|---:|---:|
| Ridge | 1.55868 | 1.20898 |
| CatBoost | **1.55290** | **1.19926** |

## Public-release note

The full processed dataset was not committed because its upstream redistribution terms were not clear. A schema-compatible sample and the complete preprocessing logic are provided. The submitted presentation remains in the downloadable reviewed package; this Markdown file captures the findings used to rewrite the README.
