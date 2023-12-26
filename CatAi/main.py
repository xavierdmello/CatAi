from fastai.vision.all import *
import gradio as gr

# Load  model
learn = load_learner('model.pkl')
labels = learn.dls.vocab

# Define prediction function
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Create Gradio interface using components
interface = gr.Interface(
    fn=predict, 
    inputs=gr.components.Image(),  # Updated this line
    outputs=gr.components.Label()
)

# Launch interface
interface.launch(share=True)
