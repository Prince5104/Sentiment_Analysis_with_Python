import gradio as gr
from transformers import pipeline

print("Loading 3-class sentiment analysis model (Positive, Neutral, Negative)...")
# Using a model that natively supports Neutral sentiment classification
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter valid text for analysis."
    try:
        result = classifier(text)[0]
        label = result['label'].capitalize() # Positive, Neutral, or Negative
        score = round(result['score'] * 100, 2)
        return f"Prediction: {label} ({score}% confidence)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your sentence here..."),
    outputs="text",
    title="📊 3-Way Sentiment Analysis AI Model",
    description="Accurate sentiment classification supporting Positive, Neutral, and Negative outputs."
)

if __name__ == "__main__":
    demo.launch(share=True)
