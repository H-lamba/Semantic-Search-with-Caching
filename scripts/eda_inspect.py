import os
import re
import pandas as pd

# ============================================================
# Load Data (with progress)
# ============================================================
DATA_DIR = "data/raw/20_newsgroups"
rows = []
categories = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
for i, category in enumerate(categories, 1):
    cat_path = os.path.join(DATA_DIR, category)
    for filename in os.listdir(cat_path):
        filepath = os.path.join(cat_path, filename)
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()
        rows.append({"category": category, "filename": filename, "text": text})
    print(f"[Loading] {i}/{len(categories)} - {category}")

df = pd.DataFrame(rows)
print(f"\nTotal documents loaded: {len(df)}\n")

# ============================================================
print("[Analyzing 1/10] Email Headers...")
# ============================================================
header_keys = ["From:", "Subject:", "Organization:", "Lines:", "Newsgroups:",
               "Path:", "Xref:", "NNTP-Posting-Host:", "Reply-To:", "Distribution:"]
header_counts = {}
for key in header_keys:
    count = df["text"].apply(lambda t: key in t.split("\n\n")[0] if "\n\n" in t else False).sum()
    header_counts[key] = count
    print(f"  Posts with '{key}': {count}")

sample = df["text"].iloc[0]
header_part = sample.split("\n\n")[0] if "\n\n" in sample else "No header found"
print(f"\n  Sample header:\n  {header_part[:300]}\n")

# ============================================================
print("[Analyzing 2/10] Quoted Replies...")
# ============================================================
df["quoted_lines"] = df["text"].apply(
    lambda t: sum(1 for line in t.split("\n") if line.strip().startswith(">"))
)
posts_with_quotes = (df["quoted_lines"] > 0).sum()
print(f"  Posts with quoted lines: {posts_with_quotes} / {len(df)} ({posts_with_quotes/len(df)*100:.1f}%)")
print(f"  Avg quoted lines per post: {df['quoted_lines'].mean():.1f}")
print(f"  Max quoted lines in a post: {df['quoted_lines'].max()}")
print(f"  Posts with >20 quoted lines: {(df['quoted_lines'] > 20).sum()}\n")

# ============================================================
print("[Analyzing 3/10] Email Signatures...")
# ============================================================
df["has_sig"] = df["text"].apply(lambda t: "\n-- \n" in t or "\n--\n" in t)
print(f"  Posts with signatures: {df['has_sig'].sum()} / {len(df)} ({df['has_sig'].sum()/len(df)*100:.1f}%)\n")

# ============================================================
print("[Analyzing 4/10] Document Lengths...")
# ============================================================
df["text_len"] = df["text"].str.len()
print(f"  Min: {df['text_len'].min()} | Max: {df['text_len'].max()} | Mean: {df['text_len'].mean():.0f} | Median: {df['text_len'].median():.0f}")
print(f"  Empty (0 chars):       {(df['text_len'] == 0).sum()}")
print(f"  Very short (<50):      {(df['text_len'] < 50).sum()}")
print(f"  Very short (<100):     {(df['text_len'] < 100).sum()}")
print(f"  Very long (>10K):      {(df['text_len'] > 10000).sum()}\n")

# ============================================================
print("[Analyzing 5/10] Duplicate Documents...")
# ============================================================
exact_dupes = df["text"].duplicated().sum()
print(f"  Exact duplicate texts: {exact_dupes}")
df["body"] = df["text"].apply(lambda t: t.split("\n\n", 1)[1] if "\n\n" in t else t)
body_dupes = df["body"].duplicated().sum()
print(f"  Duplicate bodies (ignoring headers): {body_dupes}\n")

# ============================================================
print("[Analyzing 6/10] Non-ASCII Characters...")
# ============================================================
df["non_ascii"] = df["text"].apply(lambda t: sum(1 for c in t if ord(c) > 127))
has_non_ascii = (df["non_ascii"] > 0).sum()
heavy_non_ascii = (df["non_ascii"] > 50).sum()
print(f"  Posts with non-ASCII: {has_non_ascii} ({has_non_ascii/len(df)*100:.1f}%)")
print(f"  Posts with >50 non-ASCII chars: {heavy_non_ascii}\n")

# ============================================================
print("[Analyzing 7/10] Cross-Posted Messages...")
# ============================================================
df["ng_count"] = df["text"].apply(
    lambda t: len(re.findall(r"Newsgroups:\s*(.+)", t.split("\n\n")[0])[0].split(","))
    if "Newsgroups:" in (t.split("\n\n")[0] if "\n\n" in t else "")
    and re.findall(r"Newsgroups:\s*(.+)", t.split("\n\n")[0])
    else 1
)
cross_posted = (df["ng_count"] > 1).sum()
print(f"  Cross-posted: {cross_posted} ({cross_posted/len(df)*100:.1f}%)")
print(f"  Max groups in one post: {df['ng_count'].max()}\n")

# ============================================================
print("[Analyzing 8/10] Body Content Length...")
# ============================================================
df["body_len"] = df["body"].str.strip().str.len()
print(f"  Empty body:       {(df['body_len'] == 0).sum()}")
print(f"  Body < 20 chars:  {(df['body_len'] < 20).sum()}")
print(f"  Body < 50 chars:  {(df['body_len'] < 50).sum()}\n")

# ============================================================
print("[Analyzing 9/10] HTML / MIME Content...")
# ============================================================
has_html = df["text"].apply(lambda t: bool(re.search(r"<[a-zA-Z][^>]*>", t))).sum()
has_mime = df["text"].apply(lambda t: "Content-Type:" in t or "MIME-Version:" in t).sum()
has_base64 = df["text"].apply(lambda t: "base64" in t.lower()).sum()
print(f"  HTML tags: {has_html}")
print(f"  MIME headers: {has_mime}")
print(f"  Base64 mentions: {has_base64}\n")

# ============================================================
print("[Analyzing 10/10] Emails in Body Text...")
# ============================================================
df["email_in_body"] = df["body"].apply(lambda t: len(re.findall(r"[\w.-]+@[\w.-]+", t)))
has_emails = (df["email_in_body"] > 0).sum()
print(f"  Posts with emails in body: {has_emails} ({has_emails/len(df)*100:.1f}%)\n")

# ============================================================
print("=" * 60)
print("SUMMARY OF ALL ANOMALIES FOUND")
print("=" * 60)
print(f"  Total documents:               {len(df)}")
print(f"  With email headers:            {header_counts.get('From:', 0)}")
print(f"  With quoted replies:           {posts_with_quotes}")
print(f"  With signatures:               {df['has_sig'].sum()}")
print(f"  Very short (<100 chars):       {(df['text_len'] < 100).sum()}")
print(f"  Empty body:                    {(df['body_len'] == 0).sum()}")
print(f"  Exact duplicates:              {exact_dupes}")
print(f"  Body duplicates:               {body_dupes}")
print(f"  Non-ASCII content:             {has_non_ascii}")
print(f"  Cross-posted:                  {cross_posted}")
print(f"  HTML content:                  {has_html}")
print(f"  MIME encoded:                  {has_mime}")
print(f"  Emails in body:                {has_emails}")
print("\nDone!")
