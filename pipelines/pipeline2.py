from src.char_detection import PersonCropper
from src.vlm_blip3 import CharacterAttributeExtractor
from deepdanbooru_tagger import DanbooruTagger

cropper = PersonCropper()
tagger = DanbooruTagger()
extractor = CharacterAttributeExtractor(
        model_name="blip2_t5",
        model_type="pretrain_flant5xl"
    )
# extractor = CharacterAttributeExtractor(
#         model_name="blip2_opt",
#         model_type="pretrain_opt2.7b"
#     )


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
    return character_attributes

def extract_character_attributes_pipeline(image_path, output_dir="cropped_persons"):
    """
    Pipeline that extracts characters from an image and then extracts their attributes.
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save cropped character images
        
    Returns:
        dict: Dictionary mapping cropped character paths to their attributes
    """
    # Step 1: Extract characters from the image
    cropped_characters = cropper.crop_persons(image_path, output_dir)
    
    if not cropped_characters:
        return {}
    
    # Step 2: Extract attributes for each character  
    character_attributes = {}
    for cropped_character_path in cropped_characters:
        character_attributes[cropped_character_path] = process_character_attributes(
            cropped_character_path, tagger, extractor
        )

    return character_attributes

if __name__ == "__main__":
    import json
    import time
    import os
    input_image_path = "cropped_characters/cropped_character_0.jpg"
    output_directory = "cropped_persons"
    
    results = extract_character_attributes_pipeline(input_image_path, output_directory)
    
    if results:
        print(f"Processed {len(results)} characters.")
        for character_path, attributes in results.items():
            print(f"\nCharacter at {character_path}:")
            print(json.dumps(attributes, indent=4))
    else:
        print("No valid characters detected.")
    # ---------------
    input_dir = "test_images"
    output_file = "testing_results.txt"

    with open(output_file, "w") as f:
        for filename in sorted(os.listdir(input_dir)):  # Process in sorted order
            image_path = os.path.join(input_dir, filename)

            # Skip non-image files
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            # Start timer
            start_time = time.time()

            # Extract attributes
            results = extract_character_attributes_pipeline(image_path)

            # Measure time taken
            elapsed_time = round(time.time() - start_time, 2)

            # Write to file
            f.write(f"input - {filename}\n")
            f.write("output - \n")
            f.write(json.dumps(results, indent=4))
            f.write(f"\ntime - {elapsed_time} seconds\n\n")

            print(f"Processed {filename} in {elapsed_time} seconds")

    print(f"Processing completed. Results saved to {output_file}.")
