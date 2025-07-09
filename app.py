import streamlit as st
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import datetime

st.title("🧠 Intelligent Excuse Generator")

context = st.text_input("Context (why you missed it)")
scenario = st.text_input("Scenario (e.g., class, test)")
urgency = st.selectbox("Urgency", ["Low", "Medium", "High"])
mood = st.selectbox("Mood", ["Apologetic", "Neutral", "Angry", "Sad"])

if st.button("Generate Excuse"):
    excuse = f"I am sorry I missed the {scenario} due to {context}. It was a {urgency.lower()} urgency and I was feeling very {mood.lower()} about it."
    st.success(excuse)

    # Save audio
    tts = gTTS(excuse)
    tts.save("excuse.mp3")
    st.audio("excuse.mp3")

    # Save certificate
    img = Image.new('RGB', (1000, 400), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Certified Excuse", fill='black')
    draw.text((50, 150), excuse, fill='darkred')
    img.save("certificate.png")
    st.image("certificate.png", caption="Excuse Certificate")
