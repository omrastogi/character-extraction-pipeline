import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor for attribute extraction
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def preprocess_image(image_path):
    """Load and preprocess the image for model input."""
    image = Image.open(image_path).convert("RGB")
    transform_pipeline = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor()
    ])
    return transform_pipeline(image).unsqueeze(0)

def extract_character_attributes(image_path):
    """Generate a description from an image and extract structured attributes."""
    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        generated_caption = model.generate(**model_inputs)
    
    description = processor.decode(generated_caption[0], skip_special_tokens=True)
    
    attributes = {
        "Age": "Unknown",
        "Gender": "Unknown",
        "Ethnicity": "Unknown",
        "Hair Style": "Unknown",
        "Hair Color": "Unknown",
        "Hair Length": "Unknown",
        "Eye Color": "Unknown",
        "Body Type": "Unknown",
        "Dress": "Unknown"
    }
    
    keyword_to_attribute = {
        "child": "Child", "teen": "Teen", "young": "Young Adult", "middle-aged": "Middle-aged", "elderly": "Elderly",
        "male": "Male", "female": "Female", "non-binary": "Non-binary",
        "asian": "Asian", "african": "African", "caucasian": "Caucasian",
        "ponytail": "Ponytail", "curly": "Curly", "straight": "Straight", "bun": "Bun",
        "black hair": "Black", "blonde hair": "Blonde", "red hair": "Red", "blue hair": "Blue",
        "short hair": "Short", "medium hair": "Medium", "long hair": "Long",
        "brown eyes": "Brown", "blue eyes": "Blue", "green eyes": "Green", "black eyes": "Black",
        "slim": "Slim", "muscular": "Muscular", "curvy": "Curvy",
        "casual": "Casual", "formal": "Formal", "traditional": "Traditional"
    }
    
    for keyword, attribute_value in keyword_to_attribute.items():
        if keyword in description.lower():
            for attr in attributes:
                if attributes[attr] == "Unknown":
                    attributes[attr] = attribute_value
                    break
    
    return attributes

# Example Usage
if __name__ == "__main__":
    image_path = "data/image.png"  # Replace with actual image path
    character_attributes = extract_character_attributes(image_path)
    print(json.dumps(character_attributes, indent=4))