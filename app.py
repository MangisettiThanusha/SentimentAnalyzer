import streamlit as st
from transformers import pipeline
import time
import nltk
from nltk.corpus import words
import re
import base64  # Add this import

# Download word list if not already present
nltk.download('words')

english_vocab = set(words.words())

st.set_page_config(page_title="Dark Red Sentiment Analyzer", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# Function to get base64 string of your local image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_of_bin_file("i1.jpg")  # Your local image path here

st.markdown(
    f"""
    <style>
    /* Overall app background with image and dark overlay */
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        position: relative;
        color: #f0e6d2;  /* Light beige text */
        font-family: 'Georgia', serif;
        padding: 1rem 3rem;
        min-height: 100vh;
        overflow: hidden;
    }}

    /* Dark overlay */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 0;
    }}

    /* Make sure children are above overlay */
    .main > div {{
        position: relative;
        z-index: 1;
    }}

    /* Title styling */
    .title {{
        font-size: 2.5rem;
        font-weight: 700;
        color: #f7c59f;  /* Soft warm beige */
        margin-bottom: 1.5rem;
        text-align: center;
        text-shadow: 1px 1px 4px #440000;
    }}

    /* Text area */
    textarea {{
        background-color: #2a0000; /* Dark red */
        color: #f0e6d2;
        border-radius: 8px;
        padding: 14px;
        font-size: 1.1rem;
        border: 2px solid #661111;
        resize: vertical;
        min-height: 120px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
    }}

    /* Textarea hover effect: glowing border + scale */
    textarea:hover {{
        border-color: #aa2222;
        box-shadow: 0 0 12px 4px #aa2222;
        transform: scale(1.03);
        outline: none;
    }}

    /* Button */
    div.stButton > button {{
        background-color: #3b0000;  /* Dark red */
        color: #000000;  /* Black text */
        font-size: 1.15rem;
        padding: 12px 30px;
        border-radius: 10px;
        border: 2px solid #661111;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
        margin-top: 1.5rem;
        width: 100%;
        box-shadow: 0 5px 10px rgba(59, 0, 0, 0.7);
        font-weight: 700;
    }}

    div.stButton > button:hover {{
        background-color: #661111;
        box-shadow: 0 0 20px 6px #a33333;
        color: #000000;
        border-color: #a33333;
        transform: scale(1.05);
    }}

    /* Result box */
    .result {{
        margin-top: 1.8rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f7c59f;
        text-align: center;
        background: #2a0000;
        padding: 18px 25px;
        border-radius: 12px;
        animation: fadeIn 1s ease-in-out;
        border: 2px solid #661111;
        box-shadow: 0 4px 10px rgba(102, 17, 17, 0.9);
    }}

    /* Warning and error messages styled */
    .stWarning {{
        color: #f1c232 !important;
        font-weight: 600;
    }}
    .stError {{
        color: #ff4b4b !important;
        font-weight: 700;
    }}

    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(15px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Sentiment Analyzer</div>', unsafe_allow_html=True)

user_input = st.text_area("Enter your sentence here:")


def is_valid_sentence(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    if len(tokens) < 2:
        return False
    valid_words = [w for w in tokens if w in english_vocab]
    if len(valid_words) / len(tokens) < 0.6:
        return False
    return True


if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to analyze.")
    elif not is_valid_sentence(user_input):
        st.error("❌ Invalid sentence. Please enter a meaningful sentence.")
    else:
        with st.spinner("Analyzing sentiment... 🔥"):
            time.sleep(1.5)
            result = sentiment_pipeline(user_input)[0]
            label = result['label']
            score = result['score']
        st.success("✅ Done analyzing!")
        st.markdown(
            f'<div class="result">Predicted Sentiment: <strong>{label}</strong><br>Confidence: <strong>{score:.3f}</strong></div>',
            unsafe_allow_html=True,
        )
