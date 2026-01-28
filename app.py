import streamlit as st
import pickle
from preprocessing import preprocess_text

# -------------------
# Page configuration
# -------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🛒",
    layout="centered"
)

# -------------------
# Load trained model + vectorizer
# -------------------
@st.cache_resource
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        saved_objects = pickle.load(f)
    return saved_objects['model'], saved_objects['vectorizer']

model, cv = load_model()

# -------------------
# Prediction function
# -------------------
def predict_rating(review):
    preprocessed_review = preprocess_text([review])
    review_vectorized = cv.transform(preprocessed_review)
    prediction = model.predict(review_vectorized)
    return prediction[0]

# -------------------
# Sidebar
# -------------------
st.sidebar.title("📊 Project Info")
st.sidebar.markdown("""
**Project:** Sentiment Analysis  
**Model:** Decision Tree Classifier  
**Vectorizer:** TF-IDF  
**Use Case:** Product Reviews  
""")

st.sidebar.markdown("---")
st.sidebar.info("Enter a review and click **Analyze Sentiment**")

# -------------------
# Main UI
# -------------------
st.markdown(
    "<h1 style='text-align: center;'>🛒 Sentiment Analysis</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: grey;'>Analyze whether a product review is Positive or Negative</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Input box
review_input = st.text_area(
    "✍️ Enter your review below:",
    height=150,
    placeholder="Example: This product is amazing and worth the price!"
)

# Button
analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

# -------------------
# Output section
# -------------------
if analyze:
    if review_input.strip():
        with st.spinner("Analyzing sentiment..."):
            prediction = predict_rating(review_input)

        st.markdown("---")

        if prediction == 1:
            st.success("✅ **Positive Review**  \n⭐ Rating: **5 or above**")
            st.balloons()
        else:
            st.error("❌ **Negative Review**  \n⭐ Rating: **Below 5**")

    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# -------------------
# Footer
# -------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: grey;'>"
    "Developed by lokesh </p>",
    unsafe_allow_html=True
)
