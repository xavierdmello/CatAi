from fastai.vision.all import *
import gradio as gr
import os

# Load your model
learn = load_learner('model.pkl')
labels = learn.dls.vocab

# Define your prediction function
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Directory containing example images
example_images_dir = "./example_images/"

# Generate list of file paths for example images
example_images = [os.path.join(example_images_dir, file) for file in os.listdir(example_images_dir) if file.endswith(('jpg', 'jpeg', 'png', 'JPG'))]

# Create Gradio interface using components
interface = gr.Interface(
    fn=predict, 
    inputs=gr.components.Image(), 
    outputs=gr.components.Label(),
    examples=example_images 
)

# Launch the interface
interface.launch(share=True)
