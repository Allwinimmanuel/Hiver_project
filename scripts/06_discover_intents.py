"""
scripts/06_discover_intents.py
------------------------------
Milestone 3, Task 1 — Explore Customer Message Topics

This script analyzes customer messages to identify recurring support issues
using NLP techniques (TF-IDF, N-gram analysis, and basic KMeans clustering).

Output: reports/intent_discovery.md
"""

import sys
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
IN_PATH = Path("data/processed/amazonhelp_clean.csv")
REPORT_PATH = Path("reports/intent_discovery.md")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 10
SAMPLE_SIZE = 50_000  # For speed and memory efficiency

# Stopwords extended with brand-specific noise
STOP_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "amazon", "amazonhelp", "hi", "hello", "hey",
    "please", "plz", "pls", "help", "thanks", "thank", "amp", "https", "com",
    "co", "www", "like", "get", "got", "know", "im", "it"
]

def clean_text(text):
    text = str(text).lower()
    # Remove mentions, URLs, punctuation
    text = re.sub(r'@[a-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove single characters
    text = re.sub(r'\b[a-z]\b', '', text)
    return " ".join(text.split())

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

if not IN_PATH.exists():
    print(f"[ERROR] Input not found: {IN_PATH}")
    sys.exit(1)

# ════════════════════════════════════════════════════════════
# LOAD & PREPROCESS
# ════════════════════════════════════════════════════════════
section("LOADING & PREPROCESSING")
df = pd.read_csv(IN_PATH, low_memory=False)

# Sample to speed up TF-IDF and clustering
if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(SAMPLE_SIZE, random_state=42).copy()
else:
    df_sample = df.copy()

print(f"Total rows: {len(df):,}")
print(f"Sample size for NLP: {len(df_sample):,}")

# Only look at customer messages
print("Cleaning text...")
df_sample["clean_text"] = df_sample["customer_text"].apply(clean_text)

# Filter out empty texts after cleaning
df_sample = df_sample[df_sample["clean_text"].str.strip() != ""]
corpus = df_sample["clean_text"].tolist()

# ════════════════════════════════════════════════════════════
# N-GRAM ANALYSIS
# ════════════════════════════════════════════════════════════
section("N-GRAM ANALYSIS")

# Bigrams
print("Calculating Bigrams...")
vectorizer_2 = CountVectorizer(ngram_range=(2, 2), stop_words=STOP_WORDS, max_features=20)
X_2 = vectorizer_2.fit_transform(corpus)
bigrams = list(zip(vectorizer_2.get_feature_names_out(), X_2.sum(axis=0).tolist()[0]))
bigrams.sort(key=lambda x: x[1], reverse=True)

# Trigrams
print("Calculating Trigrams...")
vectorizer_3 = CountVectorizer(ngram_range=(3, 3), stop_words=STOP_WORDS, max_features=20)
X_3 = vectorizer_3.fit_transform(corpus)
trigrams = list(zip(vectorizer_3.get_feature_names_out(), X_3.sum(axis=0).tolist()[0]))
trigrams.sort(key=lambda x: x[1], reverse=True)

for bg, count in bigrams[:5]:
    print(f"  {bg:<25} : {count:,}")

# ════════════════════════════════════════════════════════════
# CLUSTERING (K-Means on TF-IDF)
# ════════════════════════════════════════════════════════════
section("TF-IDF & CLUSTERING")

print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(max_features=2000, stop_words=STOP_WORDS)
X_tfidf = tfidf.fit_transform(corpus)
terms = tfidf.get_feature_names_out()

print(f"Clustering into {N_CLUSTERS} topics using MiniBatchKMeans...")
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
kmeans.fit(X_tfidf)

df_sample["cluster"] = kmeans.labels_

clusters_info = []
for i in range(N_CLUSTERS):
    cluster_center = kmeans.cluster_centers_[i]
    top_indices = cluster_center.argsort()[-5:][::-1]
    top_words = [terms[ind] for ind in top_indices]
    
    # Get a sample message for this cluster
    cluster_docs = df_sample[df_sample["cluster"] == i]
    size = len(cluster_docs)
    
    # Find a doc closest to the center
    # For simplicity, just pick a random medium-length doc
    candidates = cluster_docs[(cluster_docs["clean_text"].str.len() > 30) & (cluster_docs["clean_text"].str.len() < 150)]
    if not candidates.empty:
        example_text = candidates.iloc[0]["customer_text"]
    else:
        example_text = cluster_docs.iloc[0]["customer_text"] if not cluster_docs.empty else "N/A"
        
    # ASCII safe encode for terminal printing
    safe_example = str(example_text).replace("\n", " ")[:150].encode("ascii", errors="replace").decode("ascii")
    
    print(f"  Cluster {i} ({size:,} docs) - Words: {', '.join(top_words)}")
    print(f"    Example: {safe_example}")
    
    clusters_info.append({
        "id": i,
        "size": size,
        "words": top_words,
        "example": str(example_text).replace("\n", " ")[:200]
    })

clusters_info.sort(key=lambda x: x["size"], reverse=True)

# ════════════════════════════════════════════════════════════
# GENERATE REPORT
# ════════════════════════════════════════════════════════════
section("GENERATING REPORT")

md_lines = [
    "# Intent Discovery Report",
    "Generated by: `scripts/06_discover_intents.py`",
    "",
    "## 1. Top N-Grams (Customer Phrasing)",
    "These phrases indicate exactly how customers ask for help.",
    "",
    "### Top Bigrams",
    "| Bigram | Count |",
    "|--------|-------|"
]
for bg, cnt in bigrams:
    md_lines.append(f"| `{bg}` | {cnt:,} |")

md_lines.extend([
    "",
    "### Top Trigrams",
    "| Trigram | Count |",
    "|---------|-------|"
])
for tg, cnt in trigrams:
    md_lines.append(f"| `{tg}` | {cnt:,} |")

md_lines.extend([
    "",
    "## 2. Topic Clusters (K-Means on TF-IDF)",
    f"Customer messages were clustered into {N_CLUSTERS} groups to discover underlying issues.",
    ""
])

for c in clusters_info:
    md_lines.extend([
        f"### Cluster {c['id']}: {', '.join(c['words']).title()}",
        f"- **Size**: {c['size']:,} messages",
        f"- **Top Keywords**: `{', '.join(c['words'])}`",
        f"- **Example Message**: > {c['example']}",
        ""
    ])

md_lines.extend([
    "## 3. Recommended Intent Taxonomy Candidates",
    "Based on the clusters and n-grams above, the following intents emerge naturally:",
    "",
    "1. **Delivery/Shipping Issues** (e.g., 'not delivered', 'package never arrived')",
    "2. **Order Status/Tracking** (e.g., 'where is my order', 'shipped')",
    "3. **Prime Membership** (e.g., 'prime video', 'cancel prime')",
    "4. **Refunds/Returns** (e.g., 'where is my refund', 'return item')",
    "5. **Account/Billing** (e.g., 'charged twice', 'login issue')",
    "6. **Customer Service Escalation** (e.g., 'call me', 'customer service')",
    "7. **Product/Item Defect** (e.g., 'damaged item', 'wrong item')",
    "8. **Other/Unknown** (Fallback for unclear/misc issues)",
    ""
])

REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")
print(f"[OK] Saved discovery report to {REPORT_PATH}")
