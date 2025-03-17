#!/usr/bin/env python

import json
import os
import pathlib
import numpy as np
import PIL.Image
import tensorflow as tf
import deepdanbooru as dd
import huggingface_hub

class DanbooruTagger:
    def __init__(self, model_repo="public-data/DeepDanbooru", json_path="danbooru_bucket.json"):
        """
        Initializes the Danbooru tagger model and loads tag buckets.

        Args:
            model_repo (str): Hugging Face repo where the model is stored.
            json_path (str): Path to the JSON file containing tag buckets.
        """
        self.model = self._load_model(model_repo)
        self.labels = self._load_labels(model_repo)
        self.tag_buckets = self._load_tag_buckets(json_path)

    def _load_model(self, repo: str) -> tf.keras.Model:
        """Loads the DeepDanbooru model from Hugging Face Hub."""
        path = huggingface_hub.hf_hub_download(repo, "model-resnet_custom_v3.h5")
        return tf.keras.models.load_model(path)

    def _load_labels(self, repo: str) -> list[str]:
        """Loads the tag labels from Hugging Face Hub."""
        path = huggingface_hub.hf_hub_download(repo, "tags.txt")
        with pathlib.Path(path).open() as f:
            return [line.strip() for line in f]

    def _load_tag_buckets(self, json_path: str) -> dict:
        """Loads tag buckets from a JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)

    def predict_tags(self, image: PIL.Image.Image, score_threshold: float = 0.5) -> dict:
        """
        Predicts tags for an input image using DeepDanbooru.

        Args:
            image (PIL.Image.Image): Input image.
            score_threshold (float): Minimum score threshold for returned tags.

        Returns:
            dict: Dictionary of predicted tags with confidence scores.
        """
        _, height, width, _ = self.model.input_shape
        image = np.asarray(image)
        image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
        image = image.numpy()
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.0

        probs = self.model.predict(image[None, ...])[0].astype(float)
        indices = np.argsort(probs)[::-1]

        result = {self.labels[i]: probs[i] for i in indices if probs[i] >= score_threshold}
        return result

    def find_tags_in_buckets(self, tags: dict) -> dict:
        """
        Finds the tags in their respective attribute buckets.

        Args:
            tags (dict): Dictionary of predicted tags with confidence scores.

        Returns:
            dict: Tags categorized into their attribute buckets.
        """
        categorized_tags = {bucket: [] for bucket in self.tag_buckets}
        for tag in tags:
            for bucket, attributes in self.tag_buckets.items():
                if tag in attributes:
                    categorized_tags[bucket].append(tag)
        return categorized_tags

# Example Usage
if __name__ == "__main__":
    tagger = DanbooruTagger(json_path="danbooru_bucket.json")
    
    # Load a sample image
    sample_image_path = "data/image.png"  # Replace with the actual image path
    image = PIL.Image.open(sample_image_path)

    # Get tags
    tags = tagger.predict_tags(image, score_threshold=0.1)

    # Categorize tags into buckets
    categorized_tags = tagger.find_tags_in_buckets(tags)
    
    # Print results
    print("Categorized Tags:", categorized_tags)
