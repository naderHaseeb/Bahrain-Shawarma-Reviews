import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Bahrain Restaurant Review Analyzer",
    page_icon="🍽️"
)

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

@st.cache_resource
def load_zero_shot_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

sentiment_model = load_sentiment_model()
zero_shot_model = load_zero_shot_model()

candidate_labels = [
    "Food Quality",
    "Service",
    "Price",
    "Cleanliness",
    "Atmosphere",
    "Location",
    "Waiting Time"
]

st.title("Bahrain Restaurant Review Analyzer")

st.write(
    "Enter a restaurant review to analyze its sentiment and main topic."
)

review = st.text_area(
    "Restaurant review",
    placeholder="Example: The food was great but the service was very slow."
)

if st.button("Analyze"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        sentiment_result = sentiment_model(review)[0]

        sentiment_output = {
            "label": sentiment_result["label"],
            "score": sentiment_result["score"],
            "metadata": "huggingface_AI_model"
        }

        topic_result = zero_shot_model(
            review,
            candidate_labels
        )

        topic_output = {
            "label": topic_result["labels"][0],
            "score": topic_result["scores"][0],
            "metadata": "huggingface_AI_model"
        }

        st.subheader("Sentiment")

        st.write(
            sentiment_output["label"].capitalize()
        )

        st.write(
            f"Confidence: {sentiment_output['score']:.2%}"
        )

        st.subheader("Main Topic")

        st.write(
            topic_output["label"]
        )

        st.write(
            f"Confidence: {topic_output['score']:.2%}"
        )

        with st.expander("Model output"):
            st.write("Sentiment:", sentiment_output)
            st.write("Topic:", topic_output)
