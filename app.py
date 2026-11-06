import gradio as gr
from transformers import pipeline

print("Loading sentiment analysis pipeline...")
# Using a robust model and letting the pipeline handle config labels automatically
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter valid text for analysis."
    try:
        result = classifier(text)[0]
        raw_label = result['label'].lower()
        
        # Normalize and map standard cardiffnlp labels dynamically
        if '0' in raw_label or 'neg' in raw_label:
            label = "Negative 📉"
        elif '1' in raw_label or 'neu' in raw_label:
            label = "Neutral ➖"
        elif '2' in raw_label or 'pos' in raw_label:
            label = "Positive 😊"
        else:
            label = raw_label.capitalize()
            
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
