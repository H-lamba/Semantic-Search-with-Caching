import os
import pandas as pd

DATA_DIR = "data/raw/20_newsgroups"

rows = []
for category in sorted(os.listdir(DATA_DIR)):
    cat_path = os.path.join(DATA_DIR, category)
    if not os.path.isdir(cat_path):
        continue
    count = 0
    for filename in os.listdir(cat_path):
        filepath = os.path.join(cat_path, filename)
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()
        rows.append({
            "category": category,
            "filename": filename,
            "text": text
        })
        count += 1
    print(f"Loaded {category}: {count} files")

df = pd.DataFrame(rows)

# Quick verification
print(f"\nShape: {df.shape}")
print(f"\nCategories ({df['category'].nunique()}):")
print(df["category"].value_counts())
print(f"\nSample post:\n{df['text'].iloc[0][:500]}")
