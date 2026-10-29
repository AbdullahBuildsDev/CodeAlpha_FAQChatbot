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


def _load_passages():
    """Build a list of (text, source) passages from the knowledge base + FAQs."""
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
                passages.append((block, f"Knowledge: {title}"))

    # 2) faqs.json -> one passage per Q&A (question + answer together).
    faq_path = os.path.join(here, "faqs.json")
    if os.path.exists(faq_path):
        with open(faq_path, encoding="utf-8") as f:
            for item in json.load(f):
                passages.append((f"{item['question']} {item['answer']}",
                                 item["answer"]))
    return passages


class SmartAssistant:
    """Semantic retrieval with an automatic TF-IDF fallback."""

    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.passages = _load_passages()
        self.mode = "semantic"
        try:
            from sentence_transformers import SentenceTransformer, util
            self._util = util
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [p[0] for p in self.passages]
            self.embeddings = self.model.encode(texts, convert_to_tensor=True,
                                                normalize_embeddings=True)
        except Exception:
            # Fallback: reuse the pure-Python TF-IDF chatbot.
            from src.tfidf_engine import FAQChatbot
            self.mode = "tf-idf (fallback)"
            self._fallback = FAQChatbot()

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
            if score < self.threshold:
                return ("I'm not sure about that one. Try rephrasing, or contact "
                        "services@codealpha.tech.", score, "")
            # Return the answer part (text after the question, if it was a FAQ).
            text, source = self.passages[best]
            answer = source if source and not source.startswith("Knowledge:") else text
            return (answer.strip(), score, source)

        # TF-IDF fallback path.
        answer, score = self._fallback.get_response(question)
        return (answer, score, "")


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
