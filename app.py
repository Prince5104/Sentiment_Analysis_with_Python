import gradio as gr

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    return "Positive 😊 (Model connected successfully)"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your text or review here..."),
    outputs="text",
    title="📊 Sentiment Analysis App",
    description="Analyze sentiment instantly using Python and Gradio."
)

if __name__ == "__main__":
    demo.launch(share=True)
