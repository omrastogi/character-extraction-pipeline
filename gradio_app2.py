import gradio as gr
import json
from PIL import Image

# Placeholder imports for your pipeline
from src.char_detection import PersonCropper
from src.vlm_blip3 import CharacterAttributeExtractor

def pipeline(image):
    if not image:
        return None, "No image provided."

    input_path = "temp_input.jpg"
    image.save(input_path)

    # 1) Crop persons
    cropper = PersonCropper()
    cropped_paths = cropper.crop_persons(input_path, "cropped_persons")

    if not cropped_paths:
        return None, "No valid characters detected."

    # 2) Extract attributes
    extractor = CharacterAttributeExtractor(
        model_name="blip2_t5",
        model_type="pretrain_flant5xl"
    )

    cropped_images = []
    results = {}
    for path in cropped_paths:
        # Load the cropped image for the gallery
        cropped_img = Image.open(path).convert("RGB")
        cropped_images.append(cropped_img)

        # Run attribute extraction
        attributes = extractor.extract_attributes(path)
        results[path] = attributes

    # Return a list of PIL images + JSON string
    return cropped_images, json.dumps(results, indent=4)


# Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("## Who’s That Character?\nUpload an image, detect characters, and extract attributes.")
    
    # Input component
    img_in = gr.Image(type="pil", label="Upload an Image")
    run_button = gr.Button("Run")

    # Gallery: specify columns=3, no .style() usage
    gallery_out = gr.Gallery(label="Cropped Characters", columns=3)
    json_out = gr.JSON(label="Attributes")

    # Link the button to our pipeline
    run_button.click(fn=pipeline, inputs=img_in, outputs=[gallery_out, json_out])

if __name__ == "__main__":
    demo.launch()
