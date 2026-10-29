"""
CodeAlpha - Smart FAQ Chatbot (web app)
-------------------------------------------------
A polished Gradio chat interface over the semantic SmartAssistant engine.
It understands the meaning of a question (AI embeddings) and answers from the
CodeAlpha knowledge base, showing a confidence score for every reply.

Run:  python app.py

Author: Abdullah Awan  |  CodeAlpha AI Internship
"""

import gradio as gr
from src.engine import SmartAssistant

BOT = SmartAssistant()

GREETING = ("Hi! 👋 I'm the CodeAlpha smart assistant. Ask me anything about the "
            "internship — applying, tasks, certificates, submission, and more.")

EXAMPLES = [
    "What is this internship about?",
    "How do I apply?",
    "Is it paid or free?",
    "How many tasks do I need to complete?",
    "Can I submit my project on Google Drive?",
    "What do I get after completing it?",
]


def respond(message, history):
    answer, score, _ = BOT.answer(message)
    # Add a subtle confidence note.
    bar = "🟢" if score >= 0.6 else ("🟡" if score >= 0.4 else "🟠")
    return f"{answer}\n\n<sub>{bar} confidence {score:.0%}</sub>"


CSS = """
.gradio-container {max-width: 860px !important; margin:auto;}
#hero {text-align:center; padding:26px 18px; border-radius:18px; margin-bottom:14px;
  background:linear-gradient(120deg,#4361ee 0%,#7209b7 60%,#f72585 100%); color:#fff;
  box-shadow:0 10px 30px rgba(67,97,238,.35);}
#hero h1 {font-size:1.9rem; margin:0; font-weight:800;}
#hero p {opacity:.92; margin:6px 0 0;}
.mode-badge {display:inline-block; margin-top:10px; padding:4px 12px; border-radius:20px;
  background:rgba(255,255,255,.2); font-size:.8rem; font-weight:600;}
footer {display:none !important;}
"""

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple",
                       radius_size="lg", font=["Inter", "system-ui", "sans-serif"])

with gr.Blocks(theme=theme, css=CSS, title="CodeAlpha · Smart Chatbot") as demo:
    gr.HTML(f"""
      <div id='hero'>
        <h1>🧠 CodeAlpha Smart Assistant</h1>
        <p>Semantic AI chatbot · understands meaning, not just keywords</p>
        <div class='mode-badge'>engine: {BOT.mode}</div>
      </div>""")

    gr.ChatInterface(
        fn=respond,
        type="messages",
        chatbot=gr.Chatbot(value=[{"role": "assistant", "content": GREETING}],
                           type="messages", height=440, show_label=False),
        examples=EXAMPLES,
        submit_btn="Send",
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
