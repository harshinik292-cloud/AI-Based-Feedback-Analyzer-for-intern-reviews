"""
ai_feedback_analyzer.py

Inputs:
 - CSV file (default: `intern_feedback.csv`) with columns:
    intern_id, intern_name, reviewer, date, feedback_text

Outputs:
 - CSV reports per-intern (aggregated metrics)
 - A summary printed to console
 - Optional: writes `reports/<intern_id>_report.txt` with human-readable summary
"""

import os
import re
import math
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

# NLP
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ML for topics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ensure nltk resources (call this in first-run environment)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


STOPWORDS = set(stopwords.words("english"))

# ---------- Utilities ----------
def clean_text(text):
    if pd.isna(text): 
        return ""
    text = str(text).strip()
    # simple cleaning
    text = re.sub(r"\s+", " ", text)
    return text

def split_sentences(text):
    if not text:
        return []
    return sent_tokenize(text)

# ---------- Sentiment ----------
sia = SentimentIntensityAnalyzer()

def sentiment_scores(text):
    if not text:
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

# ---------- Skill rubric scoring (heuristic) ----------
# Define keywords per skill. You can extend these lists.
SKILL_KEYWORDS = {
    "communication": ["communicat", "present", "clarity", "explain", "email", "articulate", "speak", "listening"],
    "technical": ["code", "implementation", "algorithm", "bug", "design", "debug", "engineering", "technical", "python", "java", "api"],
    "initiative": ["initiative", "proactive", "ownership", "drive", "self-start", "volunt", "suggest", "improv"],
    "teamwork": ["team", "collaborat", "peer", "help", "support", "cooperat", "feedback", "mentor"],
}

def skill_score_from_text(text, skill_keywords, debug=False):
    """
    Heuristic: score between 0-5 based on:
     - frequency of skill keywords (TF)
     - sentiment of sentences mentioning those keywords
    """
    sents = split_sentences(text)
    if len(sents) == 0:
        return 0.0
    total_score = 0.0
    total_weight = 0.0
    for sent in sents:
        words = [w.lower() for w in word_tokenize(sent)]
        # match keyword stems
        hits = 0
        for kw in skill_keywords:
            for w in words:
                if kw in w:
                    hits += 1
        if hits == 0:
            continue
        sent_sent = sentiment_scores(sent)["compound"]
        # map compound (-1..1) to 0..1
        sent_weight = (sent_sent + 1.0) / 2.0
        # score contribution: hits * sent_weight
        total_score += hits * sent_weight
        total_weight += hits
    if total_weight == 0:
        return 0.0
    # normalize to 0..5
    raw = total_score / total_weight  # in 0..1
    return round(raw * 5.0, 2)

# ---------- Topic extraction ----------
def extract_topics(feedback_texts, n_topics=4, top_n_terms=6):
    """
    Use TF-IDF + KMeans to identify clusters / topics across feedback_texts.
    Returns list of topics; each topic is list of top terms.
    """
    docs = [clean_text(t) for t in feedback_texts]
    if len(docs) == 0:
        return []
    # Use TF-IDF with basic tokenization
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words="english", ngram_range=(1,2))
    X = vectorizer.fit_transform(docs)
    if X.shape[0] < 2 or X.shape[1] == 0:
        # fallback: just return common words
        all_words = " ".join(docs).lower().split()
        freq = Counter([w for w in all_words if w not in STOPWORDS and len(w)>2])
        return [[w for w, _ in freq.most_common(top_n_terms)]]
    n_clusters = min(n_topics, X.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    topics = []
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :top_n_terms]]
        topics.append(top_terms)
    return topics

# ---------- Action item extraction ----------
ACTION_KEYWORDS = ["should", "needs to", "recommend", "please", "must", "could", "work on", "improve", "suggest", "focus on"]

def extract_action_items(text):
    sents = split_sentences(text)
    items = []
    for s in sents:
        low = s.lower()
        if any(k in low for k in ACTION_KEYWORDS):
            items.append(s.strip())
    # additional rule: imperative sentences (start with verb)
    for s in sents:
        words = word_tokenize(s)
        if len(words) > 0 and re.match(r"^[A-Za-z]{2,}$", words[0]) and words[0].lower() not in STOPWORDS:
            # crude heuristic — check if first word is a verb by POS? skipping POS due to dependencies
            # include if sentence shorter than 120 chars and contains a verb-like word 'improve', 'fix', etc.
            if any(kw in s.lower() for kw in ["improve", "fix", "work", "address", "practice", "prepare"]):
                items.append(s.strip())
    # dedupe
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out

# ---------- Per-intern aggregation & reporting ----------
def analyze_feedback(df):
    # Expect columns: intern_id, intern_name, reviewer, date, feedback_text
    df = df.copy()
    df["feedback_text"] = df["feedback_text"].fillna("").apply(clean_text)
    grouped = df.groupby(["intern_id","intern_name"], as_index=False)
    reports = []
    # global topic extraction
    all_texts = df["feedback_text"].tolist()
    global_topics = extract_topics(all_texts, n_topics=6, top_n_terms=6)
    for (iid, name), sub in grouped:
        texts = sub["feedback_text"].tolist()
        # Sentiment per feedback
        sentiments = [sentiment_scores(t) for t in texts]
        compound_scores = [s["compound"] for s in sentiments]
        avg_compound = float(np.mean(compound_scores)) if len(compound_scores)>0 else 0.0
        pos_pct = float(np.mean([s["pos"] for s in sentiments])) if len(sentiments)>0 else 0.0

        # Skill scores
        skill_scores = {}
        for skill, kws in SKILL_KEYWORDS.items():
            score = 0.0
            # combine all feedback for that intern
            text_combined = " ".join(texts)
            score = skill_score_from_text(text_combined, kws)
            skill_scores[skill] = score

        # Action items: combine and pick top 6
        action_items = []
        for t in texts:
            action_items += extract_action_items(t)
        action_items = action_items[:6]

        # Top positive & constructive quotes
        quotes = sorted(texts, key=lambda t: sentiment_scores(t)["compound"], reverse=True)
        top_positive = quotes[:3]
        top_negative = sorted(texts, key=lambda t: sentiment_scores(t)["compound"])[:3]

        # Recommendation heuristic
        # avg of skill scores (0..5)
        avg_skill = np.mean(list(skill_scores.values())) if len(skill_scores)>0 else 0.0
        recommendation = "Needs improvement"
        if avg_skill >= 4.0 and avg_compound > 0.3:
            recommendation = "Strong hire"
        elif avg_skill >= 3.0 and avg_compound > 0.0:
            recommendation = "Consider for role / keep on probation"
        elif avg_skill >= 2.0 and avg_compound > -0.1:
            recommendation = "Potential but monitor progress"
        else:
            recommendation = "Not recommended / needs heavy mentoring"

        # Compose report row
        reports.append({
            "intern_id": iid,
            "intern_name": name,
            "n_reviews": len(texts),
            "avg_sentiment_compound": round(avg_compound, 3),
            "pos_score_mean": round(pos_pct,3),
            "avg_skill_score_0_5": round(avg_skill,2),
            **{f"skill_{k}": v for k,v in skill_scores.items()},
            "top_positive_quote_1": top_positive[0] if len(top_positive)>0 else "",
            "top_positive_quote_2": top_positive[1] if len(top_positive)>1 else "",
            "top_negative_quote_1": top_negative[0] if len(top_negative)>0 else "",
            "top_negative_quote_2": top_negative[1] if len(top_negative)>1 else "",
            "action_items": " || ".join(action_items),
            "recommendation": recommendation,
            "global_topics": "; ".join([", ".join(t[:3]) for t in global_topics[:3]])  # short summary
        })
    reports_df = pd.DataFrame(reports)
    return reports_df, global_topics

# ---------- Main runner / demo ----------
def main(input_csv="intern_feedback.csv", output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    # basic validation
    required_cols = {"intern_id","intern_name","feedback_text"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    reports_df, global_topics = analyze_feedback(df)

    # Save summary CSV
    summary_csv = os.path.join(output_dir, "interns_summary.csv")
    reports_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to: {summary_csv}\n")

    print("Global topics (top words per cluster):")
    for i, t in enumerate(global_topics):
        print(f"  Topic {i+1}: {', '.join(t[:8])}")

    # Save per-intern readable report
    for _, row in reports_df.iterrows():
        fn = os.path.join(output_dir, f"{row['intern_id']}_report.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"Intern: {row['intern_name']} ({row['intern_id']})\n")
            f.write(f"Reviews: {row['n_reviews']}\n")
            f.write(f"Avg sentiment (compound): {row['avg_sentiment_compound']}\n")
            f.write(f"Avg skill score (0-5): {row['avg_skill_score_0_5']}\n")
            f.write("Skill breakdown:\n")
            for sk in SKILL_KEYWORDS.keys():
                f.write(f"  - {sk}: {row[f'skill_{sk}']}\n")
            f.write("\nTop positive quote(s):\n")
            if row["top_positive_quote_1"]:
                f.write("  1) " + row["top_positive_quote_1"] + "\n")
            if row["top_positive_quote_2"]:
                f.write("  2) " + row["top_positive_quote_2"] + "\n")
            f.write("\nTop constructive / negative quote(s):\n")
            if row["top_negative_quote_1"]:
                f.write("  1) " + row["top_negative_quote_1"] + "\n")
            if row["top_negative_quote_2"]:
                f.write("  2) " + row["top_negative_quote_2"] + "\n")
            f.write("\nAction items:\n")
            if row["action_items"]:
                for ai in row["action_items"].split(" || "):
                    f.write("  - " + ai + "\n")
            else:
                f.write("  - None detected\n")
            f.write("\nRecommendation: " + row["recommendation"] + "\n")
            f.write("\nGlobal topics snapshot:\n")
            f.write(row["global_topics"] + "\n")
        print(f"Wrote: {fn}")

    print("\nDone. Check the reports directory for CSV and individual text reports.")

# ---------- Sample CSV writer (for testing) ----------
def generate_sample_csv(path="intern_feedback.csv"):
    sample = [
        {"intern_id":"I001","intern_name":"Asha Singh","reviewer":"Rohit","date":"2025-08-20",
         "feedback_text":"Asha shows excellent communication and explains her approach clearly. She took initiative to propose optimizations. Needs to improve unit testing skills."},
        {"intern_id":"I001","intern_name":"Asha Singh","reviewer":"Priya","date":"2025-08-25",
         "feedback_text":"Very proactive and a pleasure to work with. Could improve understanding of system design; recommend reading more architecture patterns."},
        {"intern_id":"I002","intern_name":"Rahul Verma","reviewer":"Sonia","date":"2025-08-22",
         "feedback_text":"Rahul's code quality is okay but often late on deliverables. Needs to improve time management. Works well with the team."},
        {"intern_id":"I002","intern_name":"Rahul Verma","reviewer":"Anil","date":"2025-08-28",
         "feedback_text":"He is helpful and collaborates with peers. Recommend focusing on debugging and following code standards."},
        {"intern_id":"I003","intern_name":"Neha Rao","reviewer":"Maya","date":"2025-08-21",
         "feedback_text":"Neha picks up tasks quickly and demonstrates strong technical skills. Could be more confident when presenting to stakeholders."},
    ]
    pd.DataFrame(sample).to_csv(path, index=False)
    print(f"Sample CSV written to {path}")

# ---------- Run ----------
if __name__ == "__main__":
    # For quick test: generate sample CSV if not present
    if not os.path.exists("intern_feedback.csv"):
        generate_sample_csv("intern_feedback.csv")
    main("intern_feedback.csv", output_dir="reports")
