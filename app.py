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
  --body-background-fill:#06100c; --background-fill-primary:#06100c;
  --background-fill-secondary:rgba(255,255,255,.03);
  --block-background-fill:rgba(255,255,255,.035);
  --block-border-color:rgba(16,185,129,.16); --border-color-primary:rgba(16,185,129,.16);
  --body-text-color:#e6f4ee; --body-text-color-subdued:#8ba89b;
  --button-primary-background-fill:linear-gradient(92deg,#10b981,#14b8a6);
  --button-primary-background-fill-hover:linear-gradient(92deg,#34d399,#2dd4bf);
  --button-primary-text-color:#04140d; --button-primary-border-color:transparent;
  --input-background-fill:rgba(255,255,255,.04); --block-radius:16px;
  font-family:'Inter',system-ui,sans-serif !important; max-width:880px !important; margin:auto !important;}
body{background:
  radial-gradient(900px 500px at 82% -12%, rgba(20,184,166,.14), transparent 60%),
  radial-gradient(900px 520px at 8% -4%, rgba(16,185,129,.15), transparent 56%),
  #06100c !important;}
#hero{padding:34px 8px 12px; text-align:center;}
#hero h1{font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.2rem; margin:0;
  letter-spacing:-.5px; background:linear-gradient(90deg,#6ee7b7,#5eead4);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;}
#hero p{color:#8ba89b; margin:9px 0 0; font-size:.97rem;}
.mode-badge{display:inline-block; margin-top:12px; padding:5px 14px; border-radius:20px;
  background:rgba(16,185,129,.16); border:1px solid rgba(16,185,129,.32);
  color:#6ee7b7; font-size:.78rem; font-weight:600;}
.block,.form{backdrop-filter:blur(14px); box-shadow:0 10px 40px rgba(0,12,8,.38) !important;}
.block:has(.options),.form:has(.options){overflow:visible !important; backdrop-filter:none !important;}
button.primary,button[variant=primary]{font-weight:700 !important; letter-spacing:.3px;
  box-shadow:0 10px 26px rgba(16,185,129,.28) !important; transition:transform .15s ease;}
button.primary:hover{transform:translateY(-2px);}
label span{color:#a3c4b6 !important; font-weight:500 !important;}
ul.options,.options,div[role=listbox]{background:#0a1a14 !important;
  border:1px solid rgba(16,185,129,.24) !important; border-radius:12px !important;
  box-shadow:0 14px 40px rgba(0,12,8,.6) !important; z-index:1000 !important; backdrop-filter:none !important;}
ul.options li,.options .item,li[role=option],[role=option]{color:#e6f4ee !important; background:transparent !important;}
ul.options li:hover,.options .item:hover,li[role=option]:hover{background:rgba(16,185,129,.24) !important; color:#fff !important;}
footer{display:none !important;}
"""

theme = gr.themes.Base(primary_hue="emerald", neutral_hue="slate")

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
