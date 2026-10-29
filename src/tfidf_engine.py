"""
CodeAlpha - Task 2: FAQ Chatbot (core engine)
-------------------------------------------------
The "brain" of the chatbot. It:
  1. Loads question/answer pairs from faqs.json
  2. Pre-processes text with NLP (tokenize, lowercase, remove stopwords)
  3. Builds TF-IDF vectors for every FAQ question
  4. Matches a user's question to the most similar FAQ using cosine similarity

NLTK is used for preprocessing. If the required NLTK data is not available,
the code gracefully falls back to a lightweight regex tokenizer, so the
chatbot always works.

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import os
import re
import json
import math
from collections import Counter

# ---------------------------------------------------------------------------
# NLP preprocessing (uses NLTK if its data is already present, else a built-in
# fallback). We never trigger a network download here - that can fail on a
# fresh machine and isn't needed: the fallback below works just as well.
# To get the full NLTK word lists, run once:  python -m nltk.downloader stopwords punkt punkt_tab
# ---------------------------------------------------------------------------
# A solid built-in English stopword list (used when NLTK data isn't installed).
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "our",
    "their", "this", "that", "these", "those", "to", "for", "of", "and", "or",
    "but", "if", "so", "as", "at", "by", "from", "in", "into", "on", "onto",
    "with", "about", "against", "between", "through", "during", "how", "what",
    "which", "who", "whom", "when", "where", "why", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "not", "no", "yes",
    "please", "plz", "pls", "u", "ur", "me", "give", "get", "want", "need",
}
_USE_NLTK = False
word_tokenize = None

try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords
    from nltk.tokenize import word_tokenize as _nltk_word_tokenize

    # Only use NLTK data that is ALREADY downloaded - never fetch it here.
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt")
    _STOPWORDS = set(_nltk_stopwords.words("english")) | _STOPWORDS
    word_tokenize = _nltk_word_tokenize
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False


def preprocess(text):
    """Lowercase, tokenize, and remove stopwords/punctuation -> list of tokens."""
    text = text.lower()
    if _USE_NLTK:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = re.findall(r"[a-z0-9]+", text)
    else:
        tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t.isalnum() and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# TF-IDF + cosine similarity (pure Python - no heavy dependencies)
# ---------------------------------------------------------------------------
class FAQChatbot:
    def __init__(self, faqs_path=None, threshold=0.15):
        if faqs_path is None:
            # Data lives in the project's top-level data/ folder.
            faqs_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "faqs.json")
        with open(faqs_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.threshold = threshold          # min similarity to accept a match
        # Index each FAQ on its QUESTION (weighted x2 so it matters most) PLUS
        # its ANSWER text. Searching the answer too greatly improves recall -
        # e.g. a query about a "link" or "website" can match an answer that
        # mentions them even if the question doesn't.
        self.docs = [preprocess(f["question"]) * 2 + preprocess(f["answer"])
                     for f in self.faqs]
        self._build_index()

    def _build_index(self):
        """Compute IDF for the vocabulary and a TF-IDF vector per FAQ question."""
        n_docs = len(self.docs)
        df = Counter()
        for doc in self.docs:
            for term in set(doc):
                df[term] += 1
        # Smoothed inverse document frequency.
        self.idf = {t: math.log((1 + n_docs) / (1 + d)) + 1 for t, d in df.items()}
        self.doc_vectors = [self._vectorize(doc) for doc in self.docs]

    def _vectorize(self, tokens):
        """Turn a token list into a {term: tf-idf weight} vector."""
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {t: (count / total) * self.idf.get(t, 0.0)
                for t, count in tf.items()}

    @staticmethod
    def _cosine(v1, v2):
        """Cosine similarity between two sparse vectors (dicts)."""
        # Iterate over the smaller vector for the dot product.
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        dot = sum(w * v2.get(t, 0.0) for t, w in v1.items())
        n1 = math.sqrt(sum(w * w for w in v1.values()))
        n2 = math.sqrt(sum(w * w for w in v2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def get_response(self, user_question):
        """Return (answer, confidence) for the best-matching FAQ."""
        query_vec = self._vectorize(preprocess(user_question))
        if not query_vec:
            return ("Could you please rephrase your question?", 0.0)

        best_idx, best_score = -1, 0.0
        for i, vec in enumerate(self.doc_vectors):
            score = self._cosine(query_vec, vec)
            if score > best_score:
                best_idx, best_score = i, score

        if best_idx == -1 or best_score < self.threshold:
            return ("Sorry, I don't have an answer for that yet. "
                    "Please try rephrasing, or contact services@codealpha.tech.",
                    best_score)
        return (self.faqs[best_idx]["answer"], best_score)


# Quick command-line test:  python chatbot.py
if __name__ == "__main__":
    bot = FAQChatbot()
    print("FAQ Chatbot ready! Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye! 👋")
            break
        answer, score = bot.get_response(q)
        print(f"Bot: {answer}   (match: {score:.2f})\n")
