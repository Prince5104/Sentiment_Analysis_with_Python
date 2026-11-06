import gradio as gr
from transformers import pipeline

print("Loading twitter-roberta sentiment model...")
# Load classifier with explicit return_all_scores=False
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Twitter-roberta-base-sentiment-latest mapping:
# LABEL_0 -> Negative
# LABEL_1 -> Neutral
# LABEL_2 -> Positive
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter valid text for analysis."
    try:
        raw_result = classifier(text)[0]
        raw_label = raw_result['label']
        label = label_mapping.get(raw_label, raw_label.capitalize())
        score = round(raw_result['score'] * 100, 2)
        return f"Prediction: {label} ({score}% confidence)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your sentence here..."),
    outputs="text",
    title="📊 Multi-Class Sentiment Analysis Model",
    description="Accurate sentiment classification (Positive, Neutral, Negative)."
)

if __name__ == "__main__":
    demo.launch(share=True)
