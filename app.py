import streamlit as st
import joblib
import logging

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Fake News Detector 🕵️‍♂️")
st.write(
    "Enter a news article below and click **Check News** to see "
    "if it’s real or fake using our pre-trained logistic regression model."
)

# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        logger.info("✅ Successfully loaded vectorizer and model.")
        return vectorizer, model
    except FileNotFoundError as fnf_err:
        logger.error(f"❌ FileNotFoundError: {fnf_err}")
        st.error(
            "Could not find `vectorizer.jb` or `lr_model.jb`. "
            "Make sure both files are in your app directory."
        )
        st.stop()
    except Exception as err:
        logger.exception("❌ Unexpected error loading resources.")
        st.error("An unexpected error occurred while loading model files.")
        st.stop()

vectorizer, model = load_resources()

# ──────────────────────────────────────────────────────────────────────────────
news_text = st.text_area("News Article", height=200)

if st.button("Check News"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze...")
    else:
        with st.spinner("Analyzing the news..."):
            try:
                X = vectorizer.transform([news_text])
                logger.debug(f"Transformed input shape: {X.shape}")


                pred = model.predict(X)[0]
                confidence = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    confidence = probs.max()
                    logger.debug(f"Predicted probabilities: {probs}")

            
                if pred == 1:
                    st.success("🟢 This news appears to be REAL.")
                else:
                    st.error("🔴 This news appears to be FAKE.")

                if confidence is not None:
                    st.write(f"**Confidence:** {confidence:.2f}")

                
                logger.info(f"Input length={len(news_text)} chars; Prediction={pred}; Confidence={confidence}")
            except Exception as pred_err:
                logger.exception("❌ Error during prediction.")
                st.error(f"An error occurred during analysis: {pred_err}")
