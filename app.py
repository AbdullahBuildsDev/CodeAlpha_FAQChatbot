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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
.gradio-container{
  --body-background-fill:#09090f; --background-fill-primary:#09090f;
  --background-fill-secondary:rgba(255,255,255,.03);
  --block-background-fill:rgba(255,255,255,.04);
  --block-border-color:rgba(255,255,255,.07); --border-color-primary:rgba(255,255,255,.07);
  --body-text-color:#ededf3; --body-text-color-subdued:#9a9aa8;
  --button-primary-background-fill:linear-gradient(92deg,#8b5cf6,#ec4899);
  --button-primary-background-fill-hover:linear-gradient(92deg,#a78bfa,#f472b6);
  --button-primary-text-color:#fff; --button-primary-border-color:transparent;
  --input-background-fill:rgba(255,255,255,.045); --block-radius:18px;
  font-family:'Inter',system-ui,sans-serif !important; max-width:880px !important; margin:auto !important;}
body{background:
  radial-gradient(900px 500px at 82% -12%, rgba(236,72,153,.13), transparent 60%),
  radial-gradient(900px 520px at 8% -4%, rgba(139,92,246,.15), transparent 56%),
  #09090f !important;}
#hero{padding:34px 8px 12px; text-align:center;}
#hero h1{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.2rem; margin:0;
  letter-spacing:-.5px; background:linear-gradient(90deg,#c4b5fd,#f9a8d4);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;}
#hero p{color:#9a9aa8; margin:9px 0 0; font-size:.97rem;}
.mode-badge{display:inline-block; margin-top:12px; padding:5px 14px; border-radius:20px;
  background:rgba(139,92,246,.18); border:1px solid rgba(139,92,246,.3);
  color:#c4b5fd; font-size:.78rem; font-weight:600;}
.block,.form{backdrop-filter:blur(14px); box-shadow:0 10px 40px rgba(0,0,0,.32) !important;}
.block:has(.options),.form:has(.options){overflow:visible !important; backdrop-filter:none !important;}
button.primary,button[variant=primary]{font-weight:600 !important; letter-spacing:.3px;
  box-shadow:0 10px 26px rgba(236,72,153,.26) !important; transition:transform .15s ease;}
button.primary:hover{transform:translateY(-2px);}
label span{color:#b9b9c6 !important; font-weight:500 !important;}
ul.options,.options,div[role=listbox]{background:#15151e !important;
  border:1px solid rgba(255,255,255,.12) !important; border-radius:12px !important;
  box-shadow:0 14px 40px rgba(0,0,0,.55) !important; z-index:1000 !important; backdrop-filter:none !important;}
ul.options li,.options .item,li[role=option],[role=option]{color:#ededf3 !important; background:transparent !important;}
ul.options li:hover,.options .item:hover,li[role=option]:hover{background:rgba(139,92,246,.30) !important; color:#fff !important;}
footer{display:none !important;}
"""

theme = gr.themes.Base(primary_hue="violet", neutral_hue="slate")

with gr.Blocks(title="CodeAlpha · Smart Chatbot") as demo:
    gr.HTML(f"""
      <div id='hero'>
        <h1>CodeAlpha Smart Assistant</h1>
        <p>Semantic AI chatbot — understands meaning, not just keywords</p>
        <div class='mode-badge'>engine: {BOT.mode}</div>
      </div>""")

    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(value=[{"role": "assistant", "content": GREETING}],
                           height=440, show_label=False),
        examples=EXAMPLES,
        submit_btn="Send",
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme, css=CSS)
