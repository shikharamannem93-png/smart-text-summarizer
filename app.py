# app.py
import os
import streamlit as st
from dotenv import load_dotenv

import gradio as gr
from transformers import pipeline


st.set_page_config(page_title="Smart Summarizer", page_icon="🧠")
st.title("🧠 Smart Text Summarizer")

# Load environment variables from .env (if present)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")

def summarize_text(text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """
    Tries the new OpenAI SDK first (from openai import OpenAI),
    falls back to the legacy SDK (import openai) if needed.
    """
    # --- Try new SDK ---
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You summarize text clearly and concisely."},
                {"role": "user", "content": f"Summarize the following:\n\n{text}"}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # --- Fallback to legacy SDK ---
        try:
            import openai  # type: ignore
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You summarize text clearly and concisely."},
                    {"role": "user", "content": f"Summarize the following:\n\n{text}"}
                ],
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e2:
            # Re-raise with a clearer message so Streamlit can show it
            raise RuntimeError(
                "Failed calling OpenAI API. Check your openai package version and API key."
            ) from e2

# --- UI ---
input_text = st.text_area("Paste your text here:", height=220, placeholder="Drop an article, email, or notes…")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Summarize")
with col2:
    with st.expander("⚙️ Debug"):
        # Don’t print secrets — just show whether it's set.
        st.write("OPENAI_API_KEY set:", bool(API_KEY))
        try:
            import openai as _legacy
            st.write("openai package version:", getattr(_legacy, "__version__", "unknown"))
        except Exception as e:
            st.write("openai import error:", str(e))

if run_btn:
    if not input_text.strip():
        st.warning("Please paste some text first.")
        st.stop()
    if not API_KEY:
        st.error("Missing OPENAI_API_KEY. Add it to a .env file or export it in your shell.")
        st.info("Example .env line:\n\nOPENAI_API_KEY=sk-...")
        st.stop()

    with st.spinner("Summarizing…"):
        try:
            summary = summarize_text(input_text, API_KEY)
            st.subheader("✨ Summary")
            st.write(summary)
        except Exception as e:
            st.exception(e)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text, max_len, min_len):
    if not text.strip():
        return ""
    out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return out[0]["summary_text"]

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Smart Text Summarizer")

    # Input section
    inp = gr.Textbox(label="Input Text", lines=12, placeholder="Paste text here...")

    # Summarize button directly after input
    btn = gr.Button("✨ Summarize")

    # Sliders for parameters
    with gr.Row():
        max_len = gr.Slider(32, 256, value=120, step=1, label="Max length")
        min_len = gr.Slider(8, 128, value=30, step=1, label="Min length")

    # Button logic
    btn.click(summarize, [inp, max_len, min_len], out)

    # Output section
    out = gr.Textbox(label="Summary", lines=8)


if __name__ == "__main__":
    demo.launch()