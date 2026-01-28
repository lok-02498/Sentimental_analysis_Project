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
st.markdown('<div class="title-text">🛒  Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Analyze customer reviews using Machine Learning & TF-IDF</div>', unsafe_allow_html=True)

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

st.markdown("""
<style>
            /* Hide Streamlit default header */
header[data-testid="stHeader"] {
    display: none;
}

/* Remove top padding added by Streamlit */
.stApp {
    margin-top: -80px;
}

/* Button */
div.stButton > button {
    background-color: #00c6ff;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #0096c7;
    color: white;
}
            
/* Title */
.title-text {
    font-size: 40px;
    font-weight: bold;
    color: #0f2a44;
    text-align: center;
}

/* Subtitle */
.subtitle-text {
    font-size: 18px;
    text-align: center;
    color: #0f2a44;
    margin-bottom: 30px;
}
            /* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #88bdf2,
        #b6d7f5,
        #e6f1fb
    );
    color: #1f2d3d;
            
} 
            /* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #88bdf2, #e6f1fb);
    padding: 20px;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: #0f2a44;
    font-size: 26px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
}

/* Sidebar text */
section[data-testid="stSidebar"] p {
    color: #1f2d3d;
    font-size: 15px;
    line-height: 1.6;
}

/* Sidebar bold labels */
section[data-testid="stSidebar"] strong {
    color: #0b3c5d;
}
         </style>
""", unsafe_allow_html=True)
