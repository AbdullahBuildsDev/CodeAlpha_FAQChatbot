"""
CodeAlpha - Smart FAQ Chatbot (semantic engine)
-------------------------------------------------
A retrieval chatbot that understands the MEANING of a question, not just
keywords. It works like a small RAG (Retrieval-Augmented) system:

  1. A knowledge base (knowledge.md + faqs.json) is split into passages.
  2. Each passage is turned into an embedding vector with a free, local
     sentence-transformers model (all-MiniLM-L6-v2).
  3. A user's question is embedded the same way and matched to the most
     semantically similar passage using cosine similarity.

Everything is free and runs offline after the ~90 MB model downloads once.
If sentence-transformers is not installed, it automatically falls back to the
pure-Python TF-IDF engine in chatbot.py, so the bot always works.

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import os
import re
import json


def _extract_pdf_text(path):
    """Extract plain text from a PDF (returns '' if it can't be read)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _chunk(text, size=180, overlap=40):
    """Split long text into overlapping word chunks for precise retrieval."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size]).strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        i += size - overlap
    return chunks


def _load_passages():
    """Build (text, answer, source) passages from knowledge.md, faqs.json, and PDFs."""
    import glob
    # Data lives in the project's top-level data/ folder (one level up from src/).
    here = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    passages = []

    # 1) knowledge.md -> one passage per section (## heading).
    kb_path = os.path.join(here, "knowledge.md")
    if os.path.exists(kb_path):
        with open(kb_path, encoding="utf-8") as f:
            raw = f.read()
        for block in re.split(r"\n##\s+", raw):
            block = block.strip().lstrip("#").strip()
            if len(block) > 40:
                title = block.splitlines()[0].strip()
                passages.append((block, None, f"Knowledge: {title}"))

    # 2) faqs.json -> one passage per Q&A (answer returned directly).
    faq_path = os.path.join(here, "faqs.json")
    if os.path.exists(faq_path):
        with open(faq_path, encoding="utf-8") as f:
            for item in json.load(f):
                passages.append((f"{item['question']} {item['answer']}",
                                 item["answer"], "FAQ"))

    # 3) Any PDFs dropped in data/documents/ -> chunked passages (RAG).
    for pdf in sorted(glob.glob(os.path.join(here, "documents", "*.pdf"))):
        name = os.path.basename(pdf)
        for chunk in _chunk(_extract_pdf_text(pdf)):
            passages.append((chunk, None, f"Document: {name}"))

    return passages


class SmartAssistant:
    """Retrieval over ALL knowledge (knowledge.md + FAQs + PDFs).

    Uses semantic embeddings when available, otherwise a pure-Python TF-IDF
    index - both built over the same passages, so PDFs work in either mode.
    """

    def __init__(self):
        self.passages = _load_passages()
        self.texts = [p[0] for p in self.passages]
        self.mode = "semantic"
        try:
            from sentence_transformers import SentenceTransformer, util
            self._util = util
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(self.texts, convert_to_tensor=True,
                                                normalize_embeddings=True)
            # Tuned on the course corpus: real in-scope questions score >=0.28,
            # off-topic ones (weather, cooking, ...) stay well below, so this
            # cleanly separates "answer it" from "ask to rephrase".
            self.threshold = 0.28
        except Exception:
            self.mode = "tf-idf (fallback)"
            self.threshold = 0.12
            self._build_tfidf()

    # ---- TF-IDF index over the passages (fallback) ----
    def _build_tfidf(self):
        import math
        from collections import Counter
        from src.tfidf_engine import preprocess
        self._preprocess = preprocess
        docs = [preprocess(t) for t in self.texts]
        n = len(docs)
        df = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = [self._tfidf_vec(d) for d in docs]

    def _tfidf_vec(self, tokens):
        from collections import Counter
        if not tokens:
            return {}
        tf = Counter(tokens)
        total = len(tokens)
        return {t: (c / total) * self._idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _cosine(a, b):
        import math
        if len(a) > len(b):
            a, b = b, a
        dot = sum(w * b.get(t, 0.0) for t, w in a.items())
        na = math.sqrt(sum(w * w for w in a.values()))
        nb = math.sqrt(sum(w * w for w in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    # ---- public ----
    def answer(self, question):
        """Return (answer_text, confidence, source)."""
        question = (question or "").strip()
        if not question:
            return ("Please type a question.", 0.0, "")

        if self.mode.startswith("semantic"):
            q_emb = self.model.encode(question, convert_to_tensor=True,
                                      normalize_embeddings=True)
            scores = self._util.cos_sim(q_emb, self.embeddings)[0]
            best = int(scores.argmax())
            score = float(scores[best])
        else:
            qv = self._tfidf_vec(self._preprocess(question))
            best, score = -1, 0.0
            for i, v in enumerate(self._vecs):
                s = self._cosine(qv, v)
                if s > score:
                    best, score = i, s

        if best < 0 or score < self.threshold:
            return ("I'm not sure about that one. Try rephrasing, or contact "
                    "services@codealpha.tech.", score, "")
        text, ans, source = self.passages[best]
        return ((ans or text).strip(), score, source)


# Quick CLI test:  python engine.py
if __name__ == "__main__":
    bot = SmartAssistant()
    print(f"SmartAssistant ready (mode: {bot.mode}). Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "bye"}:
            break
        ans, score, src = bot.answer(q)
        print(f"Bot: {ans}  (conf {score:.2f})\n")
