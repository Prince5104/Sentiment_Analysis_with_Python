import gradio as gr
from transformers import pipeline

print("Initializing sentiment analysis pipeline...")
classifier = pipeline("sentiment-analysis")

def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter valid text for analysis."
    try:
        result = classifier(text)[0]
        label = result['label']
        score = round(result['score'] * 100, 2)
        return f"Prediction: {label} ({score}% confidence)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your review or text here..."),
    outputs="text",
    title="📊 Sentiment Analysis AI Model",
    description="Live inference pipeline powered by Transformers and Gradio."
)

if __name__ == "__main__":
    demo.launch(share=True)
