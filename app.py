"""
Intelligent Excuse Generator - Streamlit App
File: intelligent_excuse_generator_app.py

Features implemented:
- Modular, testable functions for analyzing context and generating excuse templates
- Keyword extraction and simple intent detection
- Sentiment analysis (optionally uses Hugging Face transformers; falls back to TextBlob)
- Multiple output formats: on-screen text, downloadable MP3 (gTTS), certificate PNG, and certificate PDF
- Explanation panel showing why the excuse was generated (keywords, chosen mood, template score)
- Logging and reproducible random seed for deterministic behavior when needed
- Clear docstrings and comments for easy interview walkthrough

To run:
    pip install -r requirements.txt
    streamlit run intelligent_excuse_generator_app.py

"""

import streamlit as st
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
import io
import datetime
import random
import re
import logging

# Optional NLP backends
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# ---------- Configuration ----------
APP_TITLE = "Intelligent Excuse Generator"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("excuse_generator")

# ---------- Utility functions (unit-testable) ----------

def clean_text(text: str) -> str:
    """Lowercase, strip, and remove extra spaces."""
    return re.sub(r"\s+", " ", text.strip())


def extract_keywords(text: str, top_n: int = 5) -> list:
    """Very simple keyword extractor: returns most frequent non-stopword tokens.
    This is intentionally simple so it can be explained in interviews and unit-tested.
    """
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())
    tokens = [t for t in text.split() if len(t) > 2]
    stopwords = set([
        'the','and','for','you','was','had','that','this','with','but','not','from','are','they','their','your'
    ])
    tokens = [t for t in tokens if t not in stopwords]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [t for t,_ in sorted_tokens[:top_n]]


def analyze_sentiment(text: str) -> dict:
    """Return a simple sentiment analysis dict with 'label' and 'score'.
    Uses Hugging Face pipeline if available; otherwise TextBlob; otherwise a naive rule-based approach.
    """
    text = clean_text(text)
    if HF_AVAILABLE:
        try:
            nlp = pipeline("sentiment-analysis")
            res = nlp(text[:512])[0]
            return {"label": res.get("label", "NEUTRAL"), "score": float(res.get("score", 0.0))}
        except Exception as e:
            logger.warning("HF pipeline failed: %s", e)
    if TEXTBLOB_AVAILABLE:
        tb = TextBlob(text)
        polarity = tb.sentiment.polarity
        label = "POSITIVE" if polarity > 0.1 else "NEGATIVE" if polarity < -0.1 else "NEUTRAL"
        return {"label": label, "score": abs(polarity)}
    # Naive rule-based fallback
    positive_words = ['good','better','well','ok','fine','improved','fortunate']
    negative_words = ['bad','worse','sick','sickened','ill','urgent','emergency','broken','angry','sad']
    score = 0
    for w in positive_words:
        if w in text:
            score += 1
    for w in negative_words:
        if w in text:
            score -= 1
    label = 'POSITIVE' if score > 0 else 'NEGATIVE' if score < 0 else 'NEUTRAL'
    return {"label": label, "score": float(abs(score))}


def choose_mood_from_context(context: str) -> str:
    """Suggest a mood based on context keywords and sentiment.
    The function is deterministic and easy to explain in interviews.
    """
    sent = analyze_sentiment(context)
    keywords = extract_keywords(context, top_n=8)
    if 'emergency' in context.lower() or 'hospital' in context.lower() or 'urgent' in context.lower():
        return 'Apologetic'
    if sent['label'] == 'NEGATIVE':
        return 'Sad'
    if any(k in ['angry','frustrated','mad'] for k in keywords):
        return 'Angry'
    return 'Neutral'


TEMPLATES = [
    "I apologize for missing the {scenario}. I was dealing with {context}. I understand this caused inconvenience and I take responsibility.",
    "Please accept my apologies — I could not attend {scenario} because of {context}. It was unexpected and I did my best to handle it.",
    "I'm sorry I missed {scenario}. Due to {context}, I couldn't make it. I'll make sure to catch up on anything I missed.",
    "Regrettably, I was absent from {scenario} because {context}. I hope you can understand and I'll ensure it won't happen again."
]


def score_template(template: str, context: str, desired_mood: str) -> float:
    """Score how well a template fits the context and mood.
    This is intentionally simple: count overlaps with keywords and sentiment alignment.
    """
    k = extract_keywords(context, top_n=6)
    score = 0.0
    for token in k:
        if token in template.lower():
            score += 1.0
    sent = analyze_sentiment(context)
    if desired_mood == 'Apologetic' and 'apolog' in template.lower():
        score += 1.0
    # small random tie-breaker but deterministic due to seed
    score += random.random() * 0.01
    return score


def generate_excuse(context: str, scenario: str, urgency: str, mood: str, variants: int = 3) -> dict:
    """Generate multiple excuse variants with analysis metadata.
    Returns: {
        'chosen': <text>,
        'variants': [ { 'text':..., 'score':..., 'mood':... } ],
        'analysis': { 'keywords':..., 'sentiment':... }
    }
    """
    context = clean_text(context)
    scenario = clean_text(scenario)
    keywords = extract_keywords(context, top_n=8)
    sentiment = analyze_sentiment(context)

    # If user left mood empty/neutral ask suggestion
    if not mood or mood == 'Auto':
        mood = choose_mood_from_context(context)

    scored = []
    for t in TEMPLATES:
        text = t.format(scenario=scenario or 'the session', context=context or 'unforeseen circumstances')
        s = score_template(t, context, mood)
        # adjust wording for urgency and mood
        if urgency.lower() == 'high' and 'urgent' not in text.lower():
            text = text + f" (This was a {urgency.lower()} matter.)"
            s += 0.2
        if mood.lower() == 'angry':
            text = text.replace('I apologize', 'I must express my frustration and apologize')
        scored.append({'text': text, 'score': s, 'mood': mood})

    scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
    variants_out = scored_sorted[:variants]
    chosen = variants_out[0]['text']

    return {
        'chosen': chosen,
        'variants': variants_out,
        'analysis': {
            'keywords': keywords,
            'sentiment': sentiment,
            'detected_mood': mood
        }
    }

def make_certificate_image(excuse_text: str, title: str = 'Certified Excuse') -> bytes:
    width, height = 1200, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 48)
        font_body = ImageFont.truetype("arial.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    # title
    draw.text((60, 40), title, font=font_title, fill='black')

    # ---- FIX: wrap text using textbbox instead of textsize ----
    max_width = width - 120
    words = excuse_text.split()
    lines = []
    cur = ""

    for w in words:
        test_line = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test_line, font=font_body)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            cur = test_line
        else:
            lines.append(cur)
            cur = w

    if cur:
        lines.append(cur)

    # draw lines
    y = 140
    for line in lines:
        draw.text((60, y), line, font=font_body, fill='darkred')
        y += 36

    # signature/footer
    draw.text(
        (width - 360, height - 100),
        f"Generated on {datetime.date.today().isoformat()}",
        font=font_body,
        fill='black'
    )

    # return bytes
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio.read()





def make_certificate_pdf(excuse_text: str) -> bytes:
"""Return PDF bytes with the excuse text. Uses reportlab."""
buffer = io.BytesIO()
c = pdf_canvas.Canvas(buffer, pagesize=letter)
width, height = letter
c.setFont('Helvetica-Bold', 20)
c.drawString(72, height - 72, 'Certified Excuse')
c.setFont('Helvetica', 12)
# simple text wrapping
textobject = c.beginText(72, height - 120)
for line in re.findall('.{1,80}(?:\s|$)', excuse_text):
textobject.textLine(line.strip())
c.drawText(textobject)
c.setFont('Helvetica-Oblique', 10)
c.drawString(72, 72, f"Generated on {datetime.date.today().isoformat()}")
c.showPage()
c.save()
buffer.seek(0)
return buffer.read()

# ---------- Streamlit UI ----------

st.set_page_config(page_title=APP_TITLE, layout='wide')
st.title(APP_TITLE)

with st.sidebar:
    st.header("Input Options")
    context = st.text_area("Explain the reason / context", value="I missed it because my bike broke down on the way and I had to wait for help.", height=140)
    scenario = st.text_input("Scenario (e.g., class, interview, test)", value="class")
    urgency = st.selectbox("Urgency", ["Low", "Medium", "High"], index=1)
    mood_choice = st.selectbox("Mood (choose 'Auto' to let the system suggest)", ["Auto", "Apologetic", "Neutral", "Angry", "Sad"], index=0)
    variants = st.slider("Number of variants to generate", min_value=1, max_value=5, value=3)
    use_hf = st.checkbox("Use Hugging Face transformers for sentiment (if installed)", value=False if not HF_AVAILABLE else True)
    st.markdown("---")
    st.markdown("**Project skills to highlight:** Natural Language Processing, Streamlit deployment, audio generation, automated reporting (PDF/PNG), unit-testing, CI/CD")

# Main
if st.button("Generate Excuse"):
    if not context.strip():
        st.error("Please provide a context so the generator can analyze it.")
    else:
        analysis = generate_excuse(context=context, scenario=scenario, urgency=urgency, mood=mood_choice, variants=variants)
        st.success("Excuse generated — see analysis and outputs below.")

        # Output: chosen excuse
        st.subheader("Final Excuse")
        st.write(analysis['chosen'])

        # Variants
        st.subheader("Variants (ranked)")
        for i, v in enumerate(analysis['variants']):
            st.markdown(f"**Variant {i+1}** (score: {v['score']:.3f}, mood: {v['mood']})")
            st.write(v['text'])

        # Analysis panel
        st.subheader("Analysis")
        st.write("Keywords:", ', '.join(analysis['analysis']['keywords']))
        st.write("Detected sentiment:", analysis['analysis']['sentiment'])
        st.write("Detected mood:", analysis['analysis']['detected_mood'])

        # Save audio (gTTS)
        try:
            tts = gTTS(analysis['chosen'])
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            st.audio(audio_buffer, format='audio/mp3')
            st.download_button("Download MP3", data=audio_buffer, file_name='excuse.mp3', mime='audio/mpeg')
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")

        # Certificate image
        try:
            png_bytes = make_certificate_image(analysis['chosen'])
            st.image(png_bytes, caption="Excuse Certificate (PNG)")
            st.download_button("Download Certificate (PNG)", data=png_bytes, file_name='certificate.png', mime='image/png')
        except Exception as e:
            st.warning(f"Certificate image creation failed: {e}")

        # Certificate PDF
        try:
            pdf_bytes = make_certificate_pdf(analysis['chosen'])
            st.download_button("Download Certificate (PDF)", data=pdf_bytes, file_name='certificate.pdf', mime='application/pdf')
        except Exception as e:
            st.info("PDF certificate not available: reportlab may not be installed")

        # Small reproducible log for interview/demo
        st.subheader("Generation Log (for reproducibility)")
        st.code(f"Context: {context}\nScenario: {scenario}\nUrgency: {urgency}\nMood chosen: {analysis['analysis']['detected_mood']}\nKeywords: {analysis['analysis']['keywords']}")

# Footer / extras
st.markdown("---")
st.caption("Tip: This project is designed to be modular so you can replace the sentiment backend, add an intent classifier or connect it to a ticketing system for real-world use.")

# End of file
