import gradio as gr
import json
from src.char_detection import PersonCropper
from src.vlm_blip3 import CharacterAttributeExtractor

def extract_character_attributes_pipeline(image):
    """
    Pipeline that extracts characters from an image and then extracts their attributes.
    
    Args:
        image (numpy array): Image uploaded via Gradio
    
    Returns:
        dict: Dictionary mapping cropped character paths to their attributes
    """
    # Save input image temporarily
    input_path = "temp_input.jpg"
    image.save(input_path)
    
    # Step 1: Extract characters from the image
    cropper = PersonCropper()
    cropped_characters = cropper.crop_persons(input_path, "cropped_persons")
    
    if not cropped_characters:
        return "No valid characters detected."
    
    # Step 2: Extract attributes for each character
    extractor = CharacterAttributeExtractor(
        model_name="blip2_t5",
        model_type="pretrain_flant5xl"
    )
    
    results = {}
    for char_path in cropped_characters:
        attributes = extractor.extract_attributes(char_path)
        results[char_path] = attributes
    
    return json.dumps(results, indent=4)

# Define Gradio interface
iface = gr.Interface(
    fn=extract_character_attributes_pipeline,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="Who's That Character?",
    description="Upload an image, detect characters, and extract their attributes using BLIP-2."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
