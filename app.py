import streamlit as st
import pickle
import re

# ----------------------------
# Load model (and vectorizer if separate)
# ----------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Try to detect if model has vectorizer
vectorizer = None
if not hasattr(model, "predict"):
    st.error("The loaded file is not a valid model.")
    st.stop()

# If the model is NOT a pipeline (i.e., needs vectorizer)
if not hasattr(model, "steps"):  # not a pipeline
    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except:
        vectorizer = None

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.markdown("""
    <style>
        .stTextInput > div > div > input {
            border-radius: 30px;
            padding: 0.8em 1.5em;
            font-size: 1.1em;
            border: 2px solid #4a90e2;
            width: 100%;
        }
        .stButton>button {
            border-radius: 25px;
            background-color: #4a90e2;
            color: white;
            padding: 0.6em 1.5em;
            font-size: 1em;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2e6dcf;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Sentiment Analyzer")
st.markdown("Type your sentence below and press Enter to predict sentiment!")

# ----------------------------
# Input box (search bar style)
# ----------------------------
user_input = st.text_input("", placeholder="Type a sentence...")

# ----------------------------
# Text preprocessing
# ----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------
# Prediction logic
# ----------------------------
if user_input:
    clean_text = preprocess_text(user_input)
    try:
        if hasattr(model, "steps"):  # If model is a pipeline (already has vectorizer)
            prediction = model.predict([clean_text])[0]
        else:
            if vectorizer is None:
                st.error("Missing vectorizer.pkl file. Please include your TF-IDF or CountVectorizer model.")
                st.stop()
            text_vector = vectorizer.transform([clean_text])
            prediction = model.predict(text_vector)[0]

        st.markdown("---")
        if prediction == 1:
            st.success("🙂 Positive Sentiment Detected!")
        else:
            st.error("☹️ Negative Sentiment Detected!")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
