# EDA Findings — 20 Newsgroups Dataset

## Dataset Overview
- **Total documents:** 19,997
- **Categories:** 20 (balanced at ~1,000 each; `soc.religion.christian` has 997)
- **Format:** Raw newsgroup posts with email-style headers + body text

## Anomalies Found

### 1. Email Headers (100% of posts)
Every post contains metadata headers (`From:`, `Subject:`, `Path:`, `Newsgroups:`, `Organization:`, `Lines:`, etc.). These must be stripped before embedding.

| Header | Posts |
|--------|-------|
| From / Subject / Newsgroups / Path | 19,997 |
| Organization | 19,145 |
| Lines | 19,937 |
| Xref | 6,050 |
| NNTP-Posting-Host | 4,236 |
| Distribution | 4,351 |
| Reply-To | 3,403 |

### 2. Quoted Replies (51.8%)
10,352 posts contain quoted lines (starting with `>`). Average 6.3 quoted lines per post, max 587. 1,516 posts have >20 quoted lines — heavily reply-based.

### 3. Email Signatures (35.0%)
7,002 posts have signature blocks after `--` or `-- `. These contain personal info, not semantic content.

### 4. Document Length
- Min: 345 chars | Max: 161,040 chars | Mean: 2,307 | Median: 1,600
- No empty or very short posts (raw text)
- 337 posts exceed 10,000 chars (very long)

### 5. Duplicates
- **531 exact duplicates** (identical full text)
- **663 body duplicates** (same body, different headers — likely cross-posts)

### 6. Non-ASCII Characters
- 73 posts (0.4%) contain non-ASCII characters — negligible impact

### 7. Cross-Posted Messages (30.5%)
6,106 posts were posted to multiple newsgroups. Max: 18 groups in one post.

### 8. Empty/Minimal Body Content
- **41 posts** have empty body after header separation
- **117 posts** have body < 50 chars — minimal useful content

### 9. HTML / MIME Content
- **12,107 posts** contain HTML-like tags (angle brackets in text)
- 36 posts have MIME headers, 0 base64 encoded

### 10. Email Addresses in Body (80.5%)
16,091 posts contain email addresses in the body text — personal info leakage.

## Cleaning Strategy (for Phase 3)
1. **Strip all email headers** (everything before first blank line)
2. **Remove quoted replies** (lines starting with `>`)
3. **Remove signatures** (everything after `--` marker)
4. **Remove HTML tags**
5. **Drop empty/very short body posts** (<50 chars after cleaning)
6. **Remove duplicate documents**
7. **Normalize text** — lowercase, strip punctuation
8. **Tokenize, remove stop words, lemmatize**
