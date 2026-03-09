import os, warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline

st.set_page_config(page_title="Hebrew Sentiment Analysis", layout="wide")

# Force RTL for Hebrew text throughout the app
st.markdown("""
<style>
    .stTextInput input, .stTextArea textarea { direction: rtl; text-align: right; }
    .stMarkdown, .stTable { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.title("Hebrew Aspect-Based Sentiment Analysis")

# --- Model options ---
ASPECT_MODELS = {
    "mDeBERTa-v3-base (~280M) — recommended": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "XLM-RoBERTa-Large (~560M) — may exceed free cloud RAM": "joeddav/xlm-roberta-large-xnli",
}
SENTIMENT_MODELS = {
    "DictaBERT sentiment (~130M)": {
        "model": "dicta-il/dictabert-sentiment",
        "task": "text-classification",
        "labels": {"positive": "pos", "negative": "neg", "neutral": "neu"},
    },
    "HeBERT sentiment (~110M)": {
        "model": "avichr/heBERT_sentiment_analysis",
        "task": "sentiment-analysis",
        "labels": {"positive": "pos", "negative": "neg", "neutral": "neu"},
    },
}

# --- Sidebar: model selection ---
with st.sidebar:
    st.header("Model Selection")
    aspect_choice = st.selectbox("Aspect (zero-shot) model", list(ASPECT_MODELS.keys()))
    sentiment_choice = st.selectbox("Sentiment model", list(SENTIMENT_MODELS.keys()))
    if "XLM-RoBERTa" in aspect_choice:
        st.warning("⚠️ This model is ~2× larger if it crashed try move to less text or different model.")


@st.cache_resource
def load_aspect_model(model_name: str):
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device="cpu",
    )


@st.cache_resource
def load_sentiment_model(model_name: str, task: str):
    return pipeline(
        task,
        model=model_name,
        top_k=None,
        device="cpu",
    )


aspect_model_id = ASPECT_MODELS[aspect_choice]
sent_cfg = SENTIMENT_MODELS[sentiment_choice]
aspect_clf = load_aspect_model(aspect_model_id)
sent_clf = load_sentiment_model(sent_cfg["model"], sent_cfg["task"])
sent_label_map = sent_cfg["labels"]

# --- Inputs ---
aspects_input = st.text_input("Aspects (comma-separated)", value="כאב, שינה, אוכל")
text = st.text_area("Hebrew text to analyze", height=150)

if st.button("Analyze") and text.strip():
    aspects = [a.strip() for a in aspects_input.split(",") if a.strip()]
    if not aspects:
        st.error("Please enter at least one aspect.")
        st.stop()

    # --- helpers ---
    def split_sentences(t: str):
        parts = re.split(r"[.!?。\n]+", t)
        return [p.strip() for p in parts if p.strip()]

    def aspect_scores(t: str):
        out = aspect_clf(t, candidate_labels=aspects, multi_label=True)
        return dict(zip(out["labels"], out["scores"]))

    def sentiment_probs(t: str):
        scores = sent_clf(t)[0]
        d = {x["label"].lower(): float(x["score"]) for x in scores}
        return {
            "pos": d.get("positive", d.get("pos", 0.0)),
            "neg": d.get("negative", d.get("neg", 0.0)),
            "neu": d.get("neutral", d.get("neu", 0.0)),
        }

    # --- analysis ---
    with st.spinner("Analyzing..."):
        sents = split_sentences(text) or [text]
        per_sent = [(aspect_scores(s), sentiment_probs(s)) for s in sents]

        relevance = {asp: sum(a.get(asp, 0.0) for a, _ in per_sent) / len(per_sent) for asp in aspects}
        overall = {k: sum(p[k] for _, p in per_sent) / len(per_sent) for k in ["pos", "neg", "neu"]}

        eps = 1e-9
        by_asp = {}
        for asp in aspects:
            wsum = sum(a.get(asp, 0.0) for a, _ in per_sent)
            by_asp[asp] = {k: sum(a.get(asp, 0.0) * p[k] for a, p in per_sent) / (wsum + eps) for k in ["pos", "neg", "neu"]}

    # --- Display scores ---
    st.subheader("Overall Sentiment")
    cols = st.columns(3)
    labels = {"pos": "Positive", "neg": "Negative", "neu": "Neutral"}
    for col, k in zip(cols, ["pos", "neg", "neu"]):
        col.metric(labels[k], f"{overall[k]:.2%}")

    # --- Overall sentiment pie ---
    fig_pie = go.Figure(go.Pie(
        labels=[labels[k] for k in ["pos", "neg", "neu"]],
        values=[overall[k] for k in ["pos", "neg", "neu"]],
        marker_colors=["#2ecc71", "#e74c3c", "#95a5a6"],
    ))
    fig_pie.update_layout(title="Overall Sentiment Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Aspect relevance bar ---
    st.subheader("Aspect Relevance")
    fig_rel = go.Figure(go.Bar(
        x=list(relevance.keys()),
        y=list(relevance.values()),
        marker_color="#3498db",
    ))
    fig_rel.update_layout(yaxis_title="Relevance Score", yaxis_range=[0, 1])
    st.plotly_chart(fig_rel, use_container_width=True)

    # --- Sentiment per aspect grouped bar ---
    st.subheader("Sentiment by Aspect")
    df = pd.DataFrame(by_asp).T.rename(columns={"pos": "Positive", "neg": "Negative", "neu": "Neutral"})
    st.markdown(df.style.format("{:.2%}").to_html(), unsafe_allow_html=True)

    fig_asp = go.Figure()
    colors = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
    for sentiment in ["Positive", "Negative", "Neutral"]:
        fig_asp.add_trace(go.Bar(name=sentiment, x=df.index, y=df[sentiment], marker_color=colors[sentiment]))
    fig_asp.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1])
    st.plotly_chart(fig_asp, use_container_width=True)
