import json

# Path to your dataset and bucket definitions
DATASET_PATH = "tags/meta_lat_sensitive.json"
BUCKETS_PATH = "tags/tag_buckets.json"

# Multi-person tags that cause removal
MULTI_PERSON_TAGS = {
    "2boys", "2girls", "multiple boys", "multiple girls",
    "3boys", "3girls", "6+boys", "6+girls"
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def annotate_attributes(dataset, buckets):
    """
    dataset: dict of the form:
      {
        "/path/to/img.jpg": {
          "tags": "1girl, angry, bikini, blue eyes, ...",
          "train_resolution": [...]
        },
        ...
      }

    buckets: dict of the form:
      {
        "Hair Style": [...],
        "Hair Color": [...],
        "Eyes": [...],
        ...
      }

    Returns: dict with annotated attributes for each image.
    """
    annotated_data = {}

    for image_path, info in dataset.items():
        # 1. Parse tags
        tags_str = info["tags"]
        # Convert to a set for efficient membership checks
        tags = set(t.strip() for t in tags_str.split(",") if t.strip())

        # Skip 1boy, 1girl
        if "1boy, 1girl" in tags_str:
            continue

        # 2. Check for multi-person tags -> skip
        if tags.intersection(MULTI_PERSON_TAGS):
            # If any multi-person tag found, we ignore this image
            continue

        # 3. For each bucket, find matching tags
        image_attributes = {}
        for bucket_name, bucket_tags in buckets.items():
            # Get all matching tags in this bucket
            matched_tags = tags.intersection(bucket_tags)
            if matched_tags:
                # You could store them as a list, or just pick one if you prefer
                # We'll store them as a list for completeness
                image_attributes[bucket_name] = list(matched_tags)
            else:
                # If no match, store "None"
                image_attributes[bucket_name] = None

        # 4. Save to annotated_data
        annotated_data[image_path] = {
            "train_resolution": info["train_resolution"],
            "attributes": image_attributes
        }

    return annotated_data

if __name__ == "__main__":
    # Load dataset and buckets
    dataset = load_json(DATASET_PATH)
    buckets = load_json(BUCKETS_PATH)

    # Annotate
    annotated = annotate_attributes(dataset, buckets)

    # Save results
    with open("tags/annotated_output.json", "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=4)
    
    print(f"Annotated {len(annotated)} images. Results in annotated_output.json")
