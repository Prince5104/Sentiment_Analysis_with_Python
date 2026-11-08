import streamlit as st
from transformers import pipeline

# Page styling
st.set_page_config(page_title="Sentiment Analysis AI", page_icon="📊", layout="centered")

st.title(" Sentiment Analysis AI Model")
st.write("Live inference pipeline powered by Transformers and Streamlit.")

@st.cache_resource
def load_model():
    print("Loading sentiment analysis model...")
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = load_model()

# User input text box
user_text = st.text_area("Type your sentence or review here:", placeholder="The software update was installed at 4:00 PM.")

if st.button("Analyze Sentiment"):
    if not user_text or not user_text.strip():
        st.warning("Please enter valid text for analysis.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                result = classifier(user_text)[0]
                raw_label = result['label'].lower()
                
                # Dynamic mapping for standard cardiffnlp labels
                if '0' in raw_label or 'neg' in raw_label:
                    label = "Negative 📉"
                elif '1' in raw_label or 'neu' in raw_label:
                    label = "Neutral ➖"
                elif '2' in raw_label or 'pos' in raw_label:
                    label = "Positive 😊"
                else:
                    label = raw_label.capitalize()
                    
                score = round(result['score'] * 100, 2)
                
                # Display results nicely
                st.success(f"**Prediction:** {label}")
                st.info(f"**Confidence:** {score}%")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
