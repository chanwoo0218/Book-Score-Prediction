from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

TARGET = "Book-Rating"
USER = "User-ID"
BOOK = "Book-ID"
TEXT_COLS = ["Book-Title", "Book-Author", "Publisher"]
NUM_COLS = ["Age", "Year-Of-Publication"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in TEXT_COLS:
        text = frame[col].fillna("").astype(str)
        frame[f"{col}_len"] = text.str.len()
        frame[f"{col}_excl"] = text.map(lambda value: value.count("!") + value.count("?"))
    year = pd.to_numeric(frame["Year-Of-Publication"], errors="coerce").fillna(0)
    frame["Year_Decade"] = year.map(
        lambda value: f"{int(value) // 10}0s" if 1900 < value < 2026 else "Unknown"
    )
    return frame


def user_holdout(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, group in frame.groupby(USER):
        idx = group.index.to_numpy()
        if len(idx) < 3:
            continue
        n_test = min(max(1, int(np.floor(len(idx) * test_size))), len(idx) - 1)
        picked = set(rng.choice(idx, size=n_test, replace=False).tolist())
        for row_idx in idx.tolist():
            (test_idx if row_idx in picked else train_idx).append(row_idx)
    return np.asarray(train_idx), np.asarray(test_idx)


def text_blocks(train: pd.DataFrame, test: pd.DataFrame, components: int = 30) -> tuple[np.ndarray, np.ndarray]:
    train_parts, test_parts = [], []
    for col in TEXT_COLS:
        tfidf = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.9)
        train_sparse = tfidf.fit_transform(train[col].fillna("").astype(str))
        test_sparse = tfidf.transform(test[col].fillna("").astype(str))
        n_comp = min(components, max(1, train_sparse.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        train_parts.append(svd.fit_transform(train_sparse))
        test_parts.append(svd.transform(test_sparse))
    return np.hstack(train_parts), np.hstack(test_parts)


def assemble(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    new_num = [f"{col}_len" for col in TEXT_COLS] + [f"{col}_excl" for col in TEXT_COLS]
    numeric = NUM_COLS + new_num
    categorical = [USER, BOOK, "Year_Decade"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_num = scaler.fit_transform(imputer.fit_transform(train[numeric]))
    test_num = scaler.transform(imputer.transform(test[numeric]))
    train_text, test_text = text_blocks(train, test)
    train_cat = train[categorical].fillna("MISSING").astype(str).to_numpy()
    test_cat = test[categorical].fillna("MISSING").astype(str).to_numpy()
    return np.hstack([train_cat, train_num, train_text]), np.hstack([test_cat, test_num, test_text]), [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/df_final1.csv.gz"))
    parser.add_argument("--model-out", type=Path, default=Path("models/catboost.cbm"))
    args = parser.parse_args()

    df = add_features(pd.read_csv(args.data))
    train_idx, test_idx = user_holdout(df)
    train, test = df.loc[train_idx].copy(), df.loc[test_idx].copy()
    X_train, X_test, cat_idx = assemble(train, test)

    model = CatBoostRegressor(
        iterations=1389,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=1.0,
        random_seed=42,
        loss_function="RMSE",
        verbose=100,
        thread_count=2,
        max_ctr_complexity=2,
    )
    model.fit(Pool(X_train, train[TARGET].to_numpy(), cat_features=cat_idx))
    pred = model.predict(X_test)
    print({
        "rmse": mean_squared_error(test[TARGET], pred) ** 0.5,
        "mae": mean_absolute_error(test[TARGET], pred),
        "r2": r2_score(test[TARGET], pred),
    })
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)


if __name__ == "__main__":
    main()
