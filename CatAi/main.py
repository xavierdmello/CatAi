from fastai.vision.all import *
import gradio as gr

# Load your model
learn = load_learner('model.pkl')
labels = learn.dls.vocab

# Define your prediction function
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

# Launch the interface
interface.launch(share=True)
