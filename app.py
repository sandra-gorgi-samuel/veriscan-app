import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import numpy as np
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download
import requests
from bs4 import BeautifulSoup
import re

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
# TOKENIZER (cached so it loads only once)
# ─────────────────────────────────────────
@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

tokenizer = load_tokenizer()

# ─────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────
class BertLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_output.last_hidden_state
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

# ─────────────────────────────────────────
# LOAD MODEL (cached so it loads only once)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="sandra-gorgi-samuel/veriscan-bert-lstm",
        filename="bert_lstm_model.pth"
    )
    m = BertLSTMClassifier()
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    return m

model = load_model()

# ─────────────────────────────────────────
# LIME PREDICT FUNCTION
# LIME passes a list of text strings and
# expects an array of shape (n_texts, n_classes)
# ─────────────────────────────────────────
def predict_proba(texts):
    """Returns probability array for a batch of text strings."""
    all_probs = []
    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # shape: (2,)
        all_probs.append(probs)

    return np.array(all_probs)  # shape: (n_texts, 2)

# ─────────────────────────────────────────
# LIME EXPLAINER (index 0 = REAL, 1 = FAKE)
# ─────────────────────────────────────────
@st.cache_resource
def get_explainer():
    return LimeTextExplainer(class_names=["REAL", "FAKE"])

lime_explainer = get_explainer()

# ─────────────────────────────────────────
# WORD LIMIT CONSTANTS
# ─────────────────────────────────────────
WORD_LIMIT = 400  # roughly matches BERT's 512-token cap

def count_words(text):
    return len(text.split())

# ─────────────────────────────────────────
# URL ARTICLE EXTRACTION
# ─────────────────────────────────────────
def is_url(text):
    """Checks whether the pasted input looks like a URL rather than article text."""
    return bool(re.match(r"^https?://", text.strip()))

def extract_article_from_url(url):
    """Fetches a webpage and extracts the main article text using BeautifulSoup."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    # Remove non-article elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # Prefer <article> tag if present, else fall back to all <p> tags
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────
# PAGE CONFIG & HEADER
# ─────────────────────────────────────────
st.set_page_config(
    page_title="VERISCAN | Fake News Detection",
    page_icon="🛡️",
    layout="centered"
)

# ── Open Graph tags for link previews (WhatsApp, etc.) ──
components.html(
    """
    <script>
        const ogTags = [
            {property: "og:title", content: "VERISCAN — Fake News Detection System"},
            {property: "og:description", content: "Powered by BERT + LSTM · Explained by LIME"},
            {property: "og:image", content: "https://raw.githubusercontent.com/sandra-gorgi-samuel/veriscan-app/main/assets/og-image.png"},
            {property: "og:type", content: "website"}
        ];
        ogTags.forEach(tag => {
            const meta = document.createElement('meta');
            meta.setAttribute('property', tag.property);
            meta.setAttribute('content', tag.content);
            window.parent.document.head.appendChild(meta);
        });
    </script>
    """,
    height=0
)

st.markdown(
    "<h1 style='text-align:center; letter-spacing:2px; "
    "color:#21e6c1; text-shadow:0 0 12px rgba(33,230,193,0.5);'>"
    "🛡️ VERISCAN</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px; margin-top:-10px;'>"
    "Fake News Detection System</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:grey;'>Powered by BERT + LSTM · Explained by LIME</p>",
    unsafe_allow_html=True
)
st.write("---")

# ─────────────────────────────────────────
# INPUT AREA
# ─────────────────────────────────────────
if "text" not in st.session_state:
    st.session_state.text = ""

input_mode = st.radio(
    "Choose input method",
    ["📝 Paste Text", "🔗 Paste URL"],
    horizontal=True,
    label_visibility="collapsed"
)

if input_mode == "🔗 Paste URL":
    user_input = st.text_input(
        "🔗 Paste a News Article URL",
        key="text",
        placeholder="https://www.example.com/news/article-title..."
    )
else:
    user_input = st.text_area(
        "📝 Paste News Article Below",
        height=200,
        key="text",
        placeholder="Paste a news article here and click Analyse News..."
    )

# ── Live word count feedback (text mode only) ───────────
if input_mode == "📝 Paste Text" and user_input.strip() != "":
    wc = count_words(user_input)
    if wc > WORD_LIMIT:
        st.warning(
            f"⚠️ Your article is **{wc} words**, which exceeds the model's "
            f"~{WORD_LIMIT}-word effective limit (512 BERT tokens). "
            f"Only the first ~{WORD_LIMIT} words will actually be analysed — "
            f"the rest will be truncated."
        )
    else:
        st.caption(f"📝 Word count: {wc} / {WORD_LIMIT}")

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("🔍 Analyse News", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Clear", use_container_width=True)

if clear_btn:
    st.session_state.clear()
    st.rerun()

# ─────────────────────────────────────────
# PREDICTION + LIME EXPLANATION
# ─────────────────────────────────────────
if predict_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text or a URL before analysing.")
    elif input_mode == "🔗 Paste URL" and not is_url(user_input):
        st.error("⚠️ That doesn't look like a valid URL. It should start with http:// or https://")
    else:
        # ── Step 0: Resolve input — URL or raw text ─────────
        if input_mode == "🔗 Paste URL":
            with st.spinner("🔗 Fetching article from URL..."):
                try:
                    article_text = extract_article_from_url(user_input.strip())
                except Exception as e:
                    article_text = ""
                    st.error(f"❌ Couldn't fetch that URL: {e}")

            if not article_text or count_words(article_text) < 30:
                st.error(
                    "⚠️ Couldn't extract readable article text from that link. "
                    "Some sites block automated fetching, or the page may not be a "
                    "plain article. Try pasting the article text directly instead."
                )
                st.stop()
            else:
                fetched_wc = count_words(article_text)
                st.success(f"✅ Article fetched — {fetched_wc} words extracted.")
                if fetched_wc > WORD_LIMIT:
                    st.warning(
                        f"⚠️ The fetched article is **{fetched_wc} words**, which exceeds "
                        f"the model's ~{WORD_LIMIT}-word effective limit (512 BERT tokens). "
                        f"Only the first ~{WORD_LIMIT} words will actually be analysed — "
                        f"the rest will be truncated."
                    )
                with st.expander("📄 View fetched article text"):
                    st.write(article_text)
                analysis_text = article_text
        else:
            analysis_text = user_input
            wc = count_words(analysis_text)
            if wc > WORD_LIMIT:
                st.warning(
                    f"⚠️ Your article is **{wc} words**, which exceeds "
                    f"the model's ~{WORD_LIMIT}-word effective limit (512 BERT tokens). "
                    f"Only the first ~{WORD_LIMIT} words will actually be analysed — "
                    f"the rest will be truncated."
                )

        # ── Step 1: Run prediction ──────────────────────────
        with st.spinner("Running prediction... 🧠"):
            enc = tokenizer(
                analysis_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                confidence = torch.max(probs).item()
                prediction = torch.argmax(probs, dim=1).item()
                real_prob = probs[0][0].item()
                fake_prob = probs[0][1].item()

        st.write("---")

        # ── Verdict banner ──────────────────────────────────
        if prediction == 1:
            st.error("🚨 This news is likely **FAKE**")
        else:
            st.success("✅ This news appears to be **REAL**")

        # ── Confidence bar ──────────────────────────────────
        st.markdown("#### 📊 Prediction Probabilities")
        col_r, col_f = st.columns(2)
        col_r.metric("✅ REAL", f"{real_prob * 100:.1f}%")
        col_f.metric("🚨 FAKE", f"{fake_prob * 100:.1f}%")

        st.progress(fake_prob, text=f"Fake probability: {fake_prob*100:.1f}%")

        st.write("---")

        # ── Step 2: Run LIME ────────────────────────────────
        st.markdown("### 🧠 LIME Explainability")
        st.caption(
            "LIME perturbs your text by removing words one at a time and watches "
            "how the model's prediction changes — revealing which words matter most."
        )

        with st.spinner("Generating LIME explanation — this may take ~30 seconds... ⏳"):
            explanation = lime_explainer.explain_instance(
                analysis_text,
                predict_proba,
                num_features=12,   # top 12 influential words
                num_samples=500    # higher = more accurate, but slower
            )

        # ── Word importance bar chart ───────────────────────
        st.markdown("#### 🔑 Top Words & Their Influence on the Prediction")
        st.caption(
            "🔴 **Red** = pushes prediction toward **FAKE** | "
            "🟢 **Green** = pushes prediction toward **REAL**"
        )

        word_weights = explanation.as_list()  # list of (word, weight) tuples

        for word, weight in sorted(word_weights, key=lambda x: abs(x[1]), reverse=True):
            is_fake_signal = weight > 0  # positive weight → toward FAKE (class 1)
            colour = "#ff4b4b" if is_fake_signal else "#21c354"
            label = "→ FAKE" if is_fake_signal else "→ REAL"
            bar_pct = min(abs(weight) * 600, 100)  # scale for visual display

            st.markdown(
                f"""
                <div style='display:flex; align-items:center; margin-bottom:8px; font-size:15px;'>
                    <span style='width:180px; font-weight:600; font-family:monospace;'>{word}</span>
                    <div style='
                        background:{colour};
                        width:{bar_pct:.1f}%;
                        height:20px;
                        border-radius:5px;
                        margin-left:10px;
                        min-width:4px;
                    '></div>
                    <span style='margin-left:12px; color:{colour}; font-weight:600;'>
                        {label} &nbsp;<span style='color:grey; font-weight:400;'>({weight:+.4f})</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("---")

        # ── LIME highlighted HTML view ──────────────────────
        st.markdown("#### 📄 LIME Highlighted Text View")
        st.caption(
            "The highlighted version of your article shows which words LIME "
            "identified as most influential, coloured by direction."
        )
        lime_html = explanation.as_html()
        components.html(lime_html, height=450, scrolling=True)

        st.write("---")
        st.markdown(
            "<p style='text-align:center; color:grey; font-size:13px;'>"
            "⚠️ This tool is for research/educational purposes. "
            "Always verify news through trusted sources."
            "</p>",
            unsafe_allow_html=True
        )