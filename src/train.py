"""Public reproducible pipeline based on the documented Book Score Prediction workflow."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

NA_TOKENS = {"", "nan", "none", "null", "n/a", "na", "unknown", "-1"}


def normalize_text(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return np.nan if text in NA_TOKENS else text


def iterative_kcore(df: pd.DataFrame, user_col: str, item_col: str, minimum: int = 3) -> pd.DataFrame:
    out = df.copy()
    while True:
        before = len(out)
        user_ok = out[user_col].map(out[user_col].value_counts()) >= minimum
        item_ok = out[item_col].map(out[item_col].value_counts()) >= minimum
        out = out[user_ok & item_ok].copy()
        if len(out) == before:
            return out


def warm_start_split(df: pd.DataFrame, user_col: str, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []
    for _, group in df.groupby(user_col):
        if len(group) < 2:
            continue
        n_test = max(1, int(round(len(group) * test_ratio)))
        n_test = min(n_test, len(group) - 1)
        test_indices.extend(rng.choice(group.index.to_numpy(), size=n_test, replace=False).tolist())
    test_mask = df.index.isin(test_indices)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def oof_target_encode(train: pd.DataFrame, valid: pd.DataFrame, col: str, target: str, folds: int = 5):
    global_mean = train[target].mean()
    encoded_train = pd.Series(index=train.index, dtype=float)
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fit_idx, val_idx in kf.split(train):
        fit = train.iloc[fit_idx]
        val = train.iloc[val_idx]
        mapping = fit.groupby(col)[target].mean()
        encoded_train.loc[val.index] = val[col].map(mapping).fillna(global_mean)
    full_mapping = train.groupby(col)[target].mean()
    encoded_valid = valid[col].map(full_mapping).fillna(global_mean)
    return encoded_train, encoded_valid, full_mapping, global_mean


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = ["Book-Title", "Book-Author", "Publisher", "Location"]
    for col in text_cols:
        df[col] = df[col].map(normalize_text)
    df["Location_country"] = df["Location"].str.split(",").str[-1].str.strip()
    df["Year-Of-Publication"] = pd.to_numeric(df["Year-Of-Publication"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Book-Rating"] = pd.to_numeric(df["Book-Rating"], errors="coerce")
    df = df[df["Book-Rating"].between(1, 10)].copy()
    required = ["User-ID", "Book-ID", "Book-Title", "Book-Author", "Publisher", "Location_country", "Book-Rating"]
    df = df.dropna(subset=required)
    return iterative_kcore(df, "User-ID", "Book-ID", minimum=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    df = prepare(pd.read_csv(args.data, low_memory=False))
    train, test = warm_start_split(df, "User-ID")

    text_cols = ["Book-Title", "Book-Author", "Publisher", "Location_country"]
    train_text = train[text_cols].fillna("").agg(" ".join, axis=1)
    test_text = test[text_cols].fillna("").agg(" ".join, axis=1)
    tfidf = TfidfVectorizer(min_df=3, max_features=30000, ngram_range=(1, 2))
    Xtr_text = tfidf.fit_transform(train_text)
    Xte_text = tfidf.transform(test_text)
    n_components = min(128, max(2, Xtr_text.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr_svd = svd.fit_transform(Xtr_text)
    Xte_svd = svd.transform(Xte_text)

    numeric_cols = ["Age", "Year-Of-Publication"]
    train_num = train[numeric_cols].fillna(train[numeric_cols].median())
    test_num = test[numeric_cols].fillna(train[numeric_cols].median())
    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(train_num)
    Xte_num = scaler.transform(test_num)

    te_parts_train, te_parts_test = [], []
    for col in ["User-ID", "Book-ID"]:
        tr, te, _, _ = oof_target_encode(train, test, col, "Book-Rating")
        te_parts_train.append(tr.to_numpy()[:, None])
        te_parts_test.append(te.to_numpy()[:, None])

    X_train = np.hstack([Xtr_svd, Xtr_num, *te_parts_train])
    X_test = np.hstack([Xte_svd, Xte_num, *te_parts_test])
    model = Ridge(alpha=10.0)
    model.fit(X_train, train["Book-Rating"])
    pred = np.clip(model.predict(X_test), 1, 10)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(test["Book-Rating"], pred))),
        "mae": float(mean_absolute_error(test["Book-Rating"], pred)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "tfidf": tfidf, "svd": svd, "scaler": scaler}, output / "model.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
