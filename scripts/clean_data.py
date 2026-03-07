"""
Data Preprocessing & Cleaning
==============================
This script cleans the raw 20 Newsgroups corpus by handling ALL anomalies
discovered during EDA (see docs/eda_findings.md).

Design note:
    We deliberately chose NOT to remove stopwords, punctuation, or apply
    lemmatization. Modern dense embedding models (e.g., sentence-transformers)
    are trained on natural human language. Their attention mechanisms rely on
    syntax, grammar, prepositions, and word order to understand contextual
    meaning. Feeding a BERT-based model a chopped-up string of keywords
    destroys the semantic signal, resulting in poor quality embeddings and
    inaccurate fuzzy clustering downstream.

    Instead, we ONLY strip structural noise (email headers, quoted replies,
    signatures, HTML, URLs, emails) and preserve the natural body text as-is.

Anomalies handled:
  1.  Email headers (100% of posts)       -> stripped
  2.  Quoted replies (51.8%)              -> removed
  3.  Email signatures (35.0%)            -> removed
  4.  Short/empty body posts              -> dropped
  5.  Exact duplicates (531)              -> deduplicated
  6.  Body duplicates / cross-posts (663) -> deduplicated
  7.  Non-ASCII characters (0.4%)         -> removed
  8.  Cross-posted messages (30.5%)       -> kept (deduped above)
  9.  HTML-like tags (60.5%)              -> stripped
  10. Email addresses in body (80.5%)     -> removed

Run from project root:
    python scripts/clean_data.py
"""

import os
import re
import pandas as pd

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "20_newsgroups")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MIN_BODY_LENGTH = 50  # Drop posts with cleaned body shorter than this


# Strip email headers and footers
# Headers are everything before the first blank line (From, Subject, etc.).
# 100% of posts have them — they add no semantic value and would pollute embeddings.
def strip_headers(text):
    """Remove email headers (everything before the first blank line)."""
    if "\n\n" in text:
        return text.split("\n\n", 1)[1]
    return text


# --- Pruning decisions ---

# Remove quoted replies (lines starting with > or |).
# 51.8% of posts contain these — they duplicate content and would bias
# clustering toward heavily-discussed threads.
def remove_quoted_replies(text):
    """Remove lines that are quoted replies from previous messages."""
    lines = text.split("\n")
    cleaned = [line for line in lines if not line.strip().startswith((">", "|"))]
    return "\n".join(cleaned)


# Remove email signatures (everything after '-- ' marker).
# 35% of posts have these — personal info, phone numbers, ASCII art that
# is irrelevant to topic semantics.
def remove_signatures(text):
    """Remove email signature blocks (after -- marker)."""
    patterns = ["\n-- \n", "\n--\n"]
    for pattern in patterns:
        if pattern in text:
            text = text.split(pattern, 1)[0]
    return text


# Remove email addresses from body (80.5% of posts have them).
def remove_email_addresses(text):
    """Remove email addresses from the text."""
    return re.sub(r"[\w.\-+]+@[\w.\-]+\.\w+", "", text)


# Strip HTML-like tags (60.5% of posts have angle-bracket content).
def remove_html_tags(text):
    """Remove HTML-like tags and angle bracket content."""
    return re.sub(r"<[^>]+>", "", text)


# Remove non-ASCII characters (0.4% of posts, garbled encoding artifacts).
def remove_non_ascii(text):
    """Remove non-ASCII characters."""
    return text.encode("ascii", errors="ignore").decode("ascii")


# Remove URLs — don't contribute to topic semantics.
def remove_urls(text):
    """Remove http/ftp URLs from text."""
    return re.sub(r"https?://\S+|ftp://\S+|www\.\S+", "", text)


# --- Text normalization ---
# We deliberately DO NOT apply the following:
#   - Stopword removal
#   - Lemmatization
#   - Punctuation stripping
#   - Aggressive tokenization
#
# Reason: Modern transformer-based embedding models (like all-MiniLM-L6-v2)
# are pre-trained on natural
# English text. Their self-attention mechanisms depend on full grammatical
# structure — articles, prepositions, punctuation, and word order — to
# produce accurate semantic representations.
#
# Stripping these features reduces text to a "bag of words" that was
# appropriate for older methods (TF-IDF, Naive Bayes) but actively harms
# dense embedding quality. This would cascade into poor clustering and
# inaccurate semantic cache matching downstream.
#
# We only apply light whitespace normalization to collapse excessive
# blank lines and trailing spaces, preserving natural sentence structure.
def normalize_whitespace(text):
    """Collapse excessive whitespace while preserving natural text structure."""
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ============================================================
# Full cleaning pipeline for a single document
# ============================================================
def clean_document(text):
    """Apply the full cleaning pipeline to a single document.

    Pipeline order (deliberate):
      1. Strip headers      - remove metadata before text processing
      2. Remove signatures  - remove personal info blocks
      3. Remove quotes      - remove duplicate reply content
      4. Remove emails      - strip personal email addresses
      5. Remove HTML tags   - strip angle bracket noise
      6. Remove URLs        - strip web links
      7. Remove non-ASCII   - clean encoding artifacts
      8. Normalize spaces   - collapse excess whitespace

    NOT applied (by design):
      - Stopword removal    - transformers need grammar structure
      - Lemmatization       - transformers handle word forms natively
      - Punctuation removal - transformers rely on punctuation for context
      - Lowercasing         - embedding models handle casing internally
    """
    text = strip_headers(text)
    text = remove_signatures(text)
    text = remove_quoted_replies(text)
    text = remove_email_addresses(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_non_ascii(text)
    text = normalize_whitespace(text)
    return text


# ============================================================
# Load, clean, deduplicate, filter, save
# ============================================================
def load_raw_data():
    """Load raw data from the 20 newsgroups directory."""
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
            rows.append({"category": category, "filename": filename, "text": text})
        print(f"[Loading] {i}/{len(categories)} - {category}")
    return pd.DataFrame(rows)


def run_cleaning():
    """Execute the full cleaning pipeline."""

    # Step 1: Load raw data
    print("=" * 60)
    print("Step 1: Loading raw data...")
    print("=" * 60)
    df = load_raw_data()
    print(f"Loaded {len(df)} documents.\n")

    # Step 2: Remove exact duplicates
    # DECISION: Remove exact duplicates first to avoid processing redundant text.
    # 531 exact duplicates found in EDA.
    print("=" * 60)
    print("Step 2: Removing exact duplicates...")
    print("=" * 60)
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"Removed {before - len(df)} exact duplicates. Remaining: {len(df)}\n")

    # Step 3: Clean each document (preserving natural language)
    print("=" * 60)
    print("Step 3: Cleaning documents (preserving natural sentences)...")
    print("=" * 60)
    total = len(df)
    cleaned_texts = []
    for i, text in enumerate(df["text"], 1):
        cleaned_texts.append(clean_document(text))
        if i % 2000 == 0 or i == total:
            print(f"[Cleaning] {i}/{total} ({i/total*100:.1f}%)")
    df["cleaned_text"] = cleaned_texts

    # Step 4: Remove body-level duplicates (cross-posts with same content)
    print(f"\n{'=' * 60}")
    print("Step 4: Removing body-level duplicates after cleaning...")
    print("=" * 60)
    before = len(df)
    df = df.drop_duplicates(subset=["cleaned_text"], keep="first").reset_index(drop=True)
    print(f"Removed {before - len(df)} body duplicates. Remaining: {len(df)}\n")

    # Step 5: Filter empty/very short documents
    # Drop documents with cleaned text shorter than 50 characters.
    print("=" * 60)
    print(f"Step 5: Filtering documents with cleaned text < {MIN_BODY_LENGTH} chars...")
    print("=" * 60)
    df["cleaned_len"] = df["cleaned_text"].str.len()
    before = len(df)
    df = df[df["cleaned_len"] >= MIN_BODY_LENGTH].reset_index(drop=True)
    print(f"Removed {before - len(df)} short/empty documents. Remaining: {len(df)}\n")

    # Step 6: Save cleaned corpus
    print("=" * 60)
    print("Step 6: Saving cleaned corpus...")
    print("=" * 60)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save as parquet (fast reload)
    output_parquet = os.path.join(PROCESSED_DIR, "cleaned_corpus.parquet")
    df[["category", "filename", "text", "cleaned_text"]].to_parquet(output_parquet, index=False)
    print(f"Saved to {output_parquet}")

    # Save as JSON (human-readable backup)
    output_json = os.path.join(PROCESSED_DIR, "cleaned_corpus.json")
    df[["category", "filename", "cleaned_text"]].to_json(output_json, orient="records", indent=2)
    print(f"Saved to {output_json}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("CLEANING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  Original documents:    19,997")
    print(f"  After deduplication:   {len(df) + (before - len(df))}")
    print(f"  After cleaning:        {len(df)}")
    print(f"  Categories remaining:  {df['category'].nunique()}")
    print(f"  Avg cleaned length:    {df['cleaned_len'].mean():.0f} chars")
    print(f"  Min cleaned length:    {df['cleaned_len'].min()} chars")
    print(f"  Max cleaned length:    {df['cleaned_len'].max()} chars")
    print(f"\n  Saved to: {PROCESSED_DIR}")

    # Show a sample of cleaned text
    print(f"\n{'=' * 60}")
    print("SAMPLE CLEANED TEXT (first document)")
    print("=" * 60)
    print(df["cleaned_text"].iloc[0][:500])

    df.drop(columns=["cleaned_len"], inplace=True)
    return df


if __name__ == "__main__":
    run_cleaning()
