"""
Data pipeline script.
Orchestrates the flow: fetch -> load -> (ready for cleaning in Phase 3)
"""
import os
import pandas as pd
from fetch_data import fetch_data

RAW_DIR = os.path.join("data", "raw", "20_newsgroups")
PROCESSED_DIR = os.path.join("data", "processed")


def load_raw_data():
    """Load all newsgroup posts into a DataFrame."""
    rows = []
    categories = sorted(os.listdir(RAW_DIR))
    for i, category in enumerate(categories, 1):
        cat_path = os.path.join(RAW_DIR, category)
        if not os.path.isdir(cat_path):
            continue
        for filename in os.listdir(cat_path):
            filepath = os.path.join(cat_path, filename)
            with open(filepath, "r", encoding="latin-1") as f:
                text = f.read()
            rows.append({
                "category": category,
                "filename": filename,
                "text": text
            })
        print(f"[Loading] {i}/{len(categories)} - {category}")

    df = pd.DataFrame(rows)
    print(f"\nLoaded {len(df)} documents from {df['category'].nunique()} categories.\n")
    return df


def save_raw_dataframe(df):
    """Save loaded data as parquet for fast reloading."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "raw_corpus.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved raw corpus to {output_path}")


def run_pipeline():
    """Run the complete data pipeline."""
    print("=" * 50)
    print("Step 1: Fetching data...")
    print("=" * 50)
    fetch_data()

    print("\n" + "=" * 50)
    print("Step 2: Loading raw data into DataFrame...")
    print("=" * 50)
    df = load_raw_data()

    print("=" * 50)
    print("Step 3: Saving as parquet for fast reload...")
    print("=" * 50)
    save_raw_dataframe(df)

    print("\n Pipeline complete! Ready for Phase 3 cleaning.")
    return df


if __name__ == "__main__":
    run_pipeline()
