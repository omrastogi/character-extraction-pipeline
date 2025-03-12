import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP-2 VQA model from LAVIS
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip_vqa", model_type="vqav2", is_eval=True, device=device
)

def extract_attributes(image_path):
    """Extract character attributes using BLIP-2 VQA"""
    
    # Load and preprocess image
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # Define structured questions for attribute extraction
    questions = {
        "Age": "What is the character's age group?",
        "Gender": "What is the character's gender?",
        "Hair Color": "What is the character's hair color?",
        "Eye Color": "What is the character's eye color?",
        "Outfit": "Describe the character's outfit."
    }

    # Process and get answers
    attributes = {}
    for key, question in questions.items():
        processed_question = txt_processors["eval"](question)
        answer = model.predict_answers(
            samples={"image": image, "text_input": processed_question},
            inference_method="generate"
        )[0]
        attributes[key] = answer

    return attributes

# Example Usage
image_path = "data/image.png"


attributes = extract_attributes(image_path)
print(attributes)
