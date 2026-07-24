from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NA_TOKENS = {"", "n/a", "na", "none", "null", "unknown"}


def normalize_text(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return np.nan if text in NA_TOKENS else text


def iterative_core_filter(frame: pd.DataFrame, min_user: int = 3, min_item: int = 3) -> pd.DataFrame:
    result = frame.copy()
    while True:
        before = len(result)
        user_counts = result["User-ID"].value_counts()
        result = result[result["User-ID"].isin(user_counts[user_counts >= min_user].index)]
        item_counts = result["Book-ID"].value_counts()
        result = result[result["Book-ID"].isin(item_counts[item_counts >= min_item].index)]
        if len(result) == before:
            return result.reset_index(drop=True)


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ["Location", "Book-Title", "Book-Author", "Publisher"]:
        frame[col] = frame[col].map(normalize_text)
    frame = frame.dropna(subset=["Book-Author", "Publisher"])
    frame["Location_country"] = frame["Location"].astype("string").str.rsplit(",", n=1).str[-1].str.strip().str.lower()
    frame.loc[frame["Location_country"].isin(NA_TOKENS), "Location_country"] = np.nan
    frame = frame.dropna(subset=["Location_country"])
    frame["Year-Of-Publication"] = pd.to_numeric(frame["Year-Of-Publication"], errors="coerce")
    frame.loc[frame["Year-Of-Publication"] == -1, "Year-Of-Publication"] = np.nan
    frame = frame.dropna(subset=["Year-Of-Publication"])
    frame = frame[frame["Book-Rating"].between(1, 10)]
    return iterative_core_filter(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/df_final1.csv.gz"))
    args = parser.parse_args()
    result = preprocess(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression="gzip")
    print(f"saved {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
