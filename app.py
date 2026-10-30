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

GREETING = ("Hi! 👋 I'm the CodeAlpha assistant. I know the tasks for **all** "
            "CodeAlpha courses — AI, Machine Learning, Python, Full Stack, DevOps, "
            "Data Analytics, Cyber Security & Cloud Computing — plus general "
            "internship questions. Ask me anything about your course!")

EXAMPLES = [
    "What are the Machine Learning tasks?",
    "Tell me about the Python Programming tasks",
    "What should I build for Cyber Security?",
    "Data Analytics project ideas",
    "How do I apply?",
    "How many tasks do I need to complete?",
]


def respond(message, history):
    answer, score, source = BOT.answer(message)
    # Add a subtle confidence note + the source document when the answer
    # came from one of the uploaded course PDFs.
    bar = "🟢" if score >= 0.5 else ("🟡" if score >= 0.25 else "🟠")
    src = f" · 📄 {source.split(':', 1)[1].strip()}" \
        if source.startswith("Document:") else ""
    return f"{answer}\n\n<sub>{bar} confidence {score:.0%}{src}</sub>"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
.gradio-container, .gradio-container.dark{
  --body-background-fill:#06100c !important; --background-fill-primary:#06100c !important;
  --background-fill-secondary:rgba(255,255,255,.03) !important;
  --block-background-fill:rgba(255,255,255,.035) !important;
  --block-border-color:rgba(16,185,129,.16) !important; --border-color-primary:rgba(16,185,129,.16) !important;
  --body-text-color:#e6f4ee !important; --body-text-color-subdued:#8ba89b !important;
  --button-primary-background-fill:linear-gradient(92deg,#10b981,#14b8a6) !important;
  --button-primary-background-fill-hover:linear-gradient(92deg,#34d399,#2dd4bf) !important;
  --button-primary-text-color:#04140d !important; --button-primary-border-color:transparent !important;
  --input-background-fill:rgba(255,255,255,.04) !important; --input-border-color:rgba(16,185,129,.16) !important;
  --block-radius:16px !important; --block-label-text-color:#a3c4b6 !important;
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

# Force dark mode regardless of the visitor's system light/dark setting
# (Gradio reads ?__theme=dark; this redirect sets it before the page renders).
HEAD = """
<script>
(function(){
  var u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme','dark');
    window.location.replace(u.toString());
  }
})();
</script>
"""

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
    demo.launch(inbrowser=True, theme=theme, css=CSS, head=HEAD)
