import gradio as gr
import json
from PIL import Image
import time

# Placeholder imports for your pipeline
from src.char_detection import PersonCropper
from src.vlm_blip3 import CharacterAttributeExtractor
from deepdanbooru_tagger import DanbooruTagger

cropper = PersonCropper()
tagger = DanbooruTagger()
extractor = CharacterAttributeExtractor(
    model_name="blip2_opt",
    model_type="pretrain_opt2.7b",
)


def process_character_attributes(char_path, tagger, extractor, threshold=0.4):
    """
    Process character attributes using DanbooruTagger and CharacterAttributeExtractor.
    
    Args:
        char_path (str): Path to the cropped character image.
        tagger (DanbooruTagger): Instance of DanbooruTagger for predicting tags.
        extractor (CharacterAttributeExtractor): Instance of CharacterAttributeExtractor.
        threshold (float): Score threshold for filtering tags.
        
    Returns:
        dict: Extracted character attributes.
    """
    start_time = time.time()
    danbooru_output = tagger.predict_all(char_path, threshold=threshold)
    formatted_string = ", ".join(
        [f"{category}: {', '.join(tag['tag'] for tag in tags)}" 
         for category, tags in danbooru_output["categorized_tags"].items()]
    )
    low_score_keys = [
        key for key, value in danbooru_output["best_candidates"].items() 
        if value["score"] < threshold
    ]
    low_score_keys.append("Ethnicity") # As this is not being estimated by DanbooruTagger
    danbooru_attributes = {key: ", ".join(item["tag"] for item in value) for key, value in danbooru_output["categorized_tags"].items()}
    character_attributes = extractor.extract_attributes(char_path, topics=low_score_keys, context=formatted_string)
    character_attributes.update(danbooru_attributes)
    end_time = time.time()
    print(f"Time taken to process character attributes: {end_time - start_time} seconds")
    return character_attributes


def pipeline(image):
    if not image:
        return None, "No image provided."

    input_path = "temp_input.jpg"
    image.save(input_path)

    # 1) Crop persons
    cropped_paths = cropper.crop_persons(input_path, "cropped_persons")

    if not cropped_paths:
        return None, "No valid characters detected."

    # 2) Extract attributes
    cropped_images = []
    results = {}
    for path in cropped_paths:
        # Load the cropped image for the gallery
        cropped_img = Image.open(path).convert("RGB")
        cropped_images.append(cropped_img)

        # Run attribute extraction
        results[path] = process_character_attributes(
                            path, 
                            tagger, 
                            extractor
        )

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
