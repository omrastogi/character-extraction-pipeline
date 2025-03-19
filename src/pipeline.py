from src.char_detection import PersonCropper
from src.vlm_blip3 import CharacterAttributeExtractor

cropper = PersonCropper()
extractor = CharacterAttributeExtractor(
        model_name="blip2_t5",
        model_type="pretrain_flant5xl"
    )
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
    results = {}
    for char_path in cropped_characters:
        attributes = extractor.extract_attributes(char_path)
        results[char_path] = attributes
    
    return results

if __name__ == "__main__":
    import json
    import os
    import time

    input_image_path = "data/continued/sensitive/danbooru_1370513_e8f30add09fdad6eb332b284f4a408bd.jpg"
    output_directory = "cropped_persons"
    
    results = extract_character_attributes_pipeline(input_image_path, output_directory)
    
    if results:
        print(f"Processed {len(results)} characters.")
        for character_path, attributes in results.items():
            print(f"\nCharacter at {character_path}:")
            print(json.dumps(attributes, indent=4))
    else:
        print("No valid characters detected.")

# -------------------------------------------------------------
    input_dir = "test_images"  # Folder containing test images
    output_file = "pipeline_test_results.txt"

    with open(output_file, "w") as f:
        for filename in sorted(os.listdir(input_dir)):  # Process files in order
            image_path = os.path.join(input_dir, filename)

            # Skip non-image files
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            # Start timer
            start_time = time.time()

            # Run the pipeline
            results = extract_character_attributes_pipeline(image_path, output_dir="cropped_persons")

            # Measure time taken
            elapsed_time = round(time.time() - start_time, 2)

            # Write results to file
            f.write(f"input - {filename}\n")
            f.write("output - \n")
            f.write(json.dumps(results, indent=4))
            f.write(f"\ntime - {elapsed_time} seconds\n\n")

            print(f"Processed {filename} in {elapsed_time} seconds")

    print(f"Processing completed. Results saved to {output_file}.")

