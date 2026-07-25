"""
Combines three review datasets into a single canonical CSV.

Sources
-------
- clothing  : app/data/reviews.csv              (~20k rows, women's clothing)
- ecommerce : app/data/amazon-review/reviews.csv (~6k rows, Amazon apparel)
- food      : app/data/amazon_listing_review.csv (~526k rows, Amazon fine food)

Output
------
app/data/combined_reviews.csv  — columns: text, rating, source
  - rating : original star rating (1,2,4,5) — 3-star neutral rows are dropped
  - source : clothing | ecommerce | food
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "app" / "data"
OUTPUT_PATH = DATA_DIR / "combined_reviews.csv"


def load_clothing() -> pd.DataFrame:
    path = DATA_DIR / "reviews.csv"
    df = pd.read_csv(path, usecols=["Review Text", "Rating"])
    df = df.rename(columns={"Review Text": "text", "Rating": "rating"})
    df["source"] = "clothing"
    return df


def load_ecommerce() -> pd.DataFrame:
    path = DATA_DIR / "amazon-review" / "reviews.csv"
    df = pd.read_csv(path, usecols=["reviewText", "rating"])
    df = df.rename(columns={"reviewText": "text"})
    df["rating"] = df["rating"].astype(float)
    df["source"] = "ecommerce"
    return df


def load_food() -> pd.DataFrame:
    path = DATA_DIR / "amazon_listing_review.csv"
    df = pd.read_csv(path, usecols=["Text", "Score"])
    df = df.rename(columns={"Text": "text", "Score": "rating"})
    df["source"] = "food"
    return df


def combine():
    print("Loading datasets …")
    parts = [load_clothing(), load_ecommerce(), load_food()]

    df = pd.concat(parts, ignore_index=True)
    print(f"Total rows before cleaning : {len(df):,}")

    df = df.dropna(subset=["text", "rating"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Drop neutral (3-star) reviews
    df = df[df["rating"] != 3]
    df["rating"] = df["rating"].astype(int)

    # Deduplicate on exact text
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"Duplicates removed          : {before - len(df):,}")

    print(f"Total rows after cleaning  : {len(df):,}")
    print()
    print("Source breakdown:")
    print(df["source"].value_counts())
    print()
    print("Rating distribution:")
    print(df["rating"].value_counts().sort_index())
    print()
    print("Label distribution (1-2 = negative, 4-5 = positive):")
    df["label"] = df["rating"].apply(lambda x: "positive" if x >= 4 else "negative")
    print(df["label"].value_counts())

    df = df.drop(columns=["label"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    combine()
