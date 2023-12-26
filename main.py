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

# Example images (replace with your actual file paths or URLs)
example_images = ["./example_images"]

# Create Gradio interface using components
interface = gr.Interface(
    fn=predict, 
    inputs=gr.components.Image(), 
    outputs=gr.components.Label(num_top_classes=3),
    examples=example_images  # Added example images
)

# Launch the interface
interface.launch(share=True)
