import re
import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# CONFIGURATION
# ==========================================

MAX_SEQUENCE_LENGTH = 30


# ==========================================
# LOAD MODEL AND TOKENIZER
# ==========================================

@st.cache_resource
def load_model_and_tokenizer():

    model = load_model(
        "models/hate_speech_rnn.keras"
    )

    with open("models/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove numbers and special characters
    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(
    page_title="Hate Speech Classifier",
    page_icon="🛡️",
    layout="centered"
)


st.title("🛡️ Hate Speech Classification")
st.write(
    "Simple RNN-based binary classifier for detecting "
    "hate speech in tweets."
)


st.subheader("Enter a Tweet")

tweet = st.text_area(
    "Tweet",
    placeholder="Enter a tweet here...",
    height=120
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Classify Tweet"):

    if not tweet.strip():

        st.warning("Please enter a tweet.")

    else:

        # Clean text
        cleaned_tweet = clean_text(tweet)

        if not cleaned_tweet:

            st.warning(
                "The tweet contains no usable text "
                "after preprocessing."
            )

        else:

            # Convert text to sequence
            sequence = tokenizer.texts_to_sequences(
                [cleaned_tweet]
            )

            # Pad sequence
            padded_sequence = pad_sequences(
                sequence,
                maxlen=MAX_SEQUENCE_LENGTH,
                padding="post",
                truncating="post"
            )

            # Prediction probability
            probability = model.predict(
                padded_sequence,
                verbose=0
            )[0][0]

            # Classification
            if probability >= 0.5:

                st.error("⚠️ Predicted Class: Hate Speech")

            else:

                st.success("✅ Predicted Class: Non-Hate")

            st.write(
                f"Prediction probability: "
                f"{probability:.4f}"
            )

            st.write(
                f"Cleaned tweet: `{cleaned_tweet}`"
            )