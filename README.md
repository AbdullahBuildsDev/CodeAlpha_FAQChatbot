# CodeAlpha — Smart FAQ Chatbot (Task 2)

A **semantic AI chatbot** that understands the *meaning* of a question, not just
keywords. It works like a small **RAG (Retrieval-Augmented)** system: the
knowledge base is embedded with a free, local **sentence-transformers** model,
and each user question is matched to the most semantically similar passage using
**cosine similarity**. If the embedding model isn't installed, it automatically
falls back to a pure-Python **TF-IDF** engine, so it always works.

## ✨ Features
- Loads Q&A pairs from an easy-to-edit `faqs.json`
- NLP preprocessing with **NLTK** (tokenizing + stopword removal)
- **TF-IDF + cosine similarity** matching (implemented in pure Python)
- Confidence score for every answer
- A friendly **chat UI** built with Tkinter
- Also runs from the command line (`python chatbot.py`)
- Graceful fallback if NLTK data isn't downloaded — it always works

## 🛠️ Tech Stack
- **Python 3**
- **NLTK** — text preprocessing
- **Tkinter** — chat interface (built into Python)
- Pure-Python TF-IDF & cosine similarity (no heavy ML dependency)

## 🚀 How to Run

```bash
# 1. (Optional) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the chat UI
python app.py
```

## 📖 How It Works
1. **Load** — questions and answers are read from `faqs.json`.
2. **Preprocess** — each question is lowercased, tokenized, and stripped of
   stopwords/punctuation.
3. **Vectorize** — a TF-IDF weight is computed for every word.
4. **Match** — the user's question is vectorized the same way, and
   **cosine similarity** picks the most similar FAQ.
5. **Respond** — if the best score passes a threshold, that FAQ's answer is
   returned; otherwise the bot asks the user to rephrase.

## 🧪 Try asking
- "How do I apply for the internship?"
- "Will I get a certificate?"
- "Can I submit on Google Drive?"
- "How do I contact support?"

## 📂 Project Structure
```
CodeAlpha_FAQChatbot/
├── app.py              # Gradio chat UI (entry point)
├── src/
│   ├── engine.py       # Semantic engine (sentence-transformers / RAG)
│   └── tfidf_engine.py # Pure-Python TF-IDF fallback engine
├── data/
│   ├── knowledge.md    # Knowledge base passages
│   └── faqs.json       # Questions & answers (edit to add your own)
├── requirements.txt    # Dependencies
├── LICENSE
└── README.md
```

## 🔧 Customize
Add your own FAQs by editing `faqs.json` — just add more
`{"question": "...", "answer": "..."}` objects. No code changes needed.

---
**CodeAlpha AI Internship — Task 2** · Built by Abdullah Awan
