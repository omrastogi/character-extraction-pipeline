#!/usr/bin/env python

import json
import os
import pathlib
import numpy as np
import PIL.Image
import tensorflow as tf
import deepdanbooru as dd
import huggingface_hub
from typing import Union, List 

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
        """Loads tag buckets from a JSON file safely."""
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Using empty tag buckets.")
            return {}

        with open(json_path, 'r') as f:
            return json.load(f)

    def predict_tags(self, image: PIL.Image.Image, score_threshold: float = 0.5) -> dict:
        """
        Predicts tags for an input image using DeepDanbooru.

        Args:
            image (PIL.Image.Image): Input image.
            score_threshold (float): Minimum score threshold for returned tags.

        Returns:
            dict: Dictionary of predicted tags -> confidence scores.
        """
        _, height, width, _ = self.model.input_shape

        # Convert image to a NumPy array
        image = np.asarray(image)
        image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
        image = image.numpy()
        # Transform and pad for model input
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.0  # Normalize

        # Predict
        probs = self.model.predict(image[None, ...])[0].astype(float)
        indices = np.argsort(probs)[::-1]

        # Filter out tags below threshold
        filtered_tags = {}
        for i in indices:
            tag = self.labels[i]
            score = probs[i]
            if score >= score_threshold:
                filtered_tags[tag] = score
        
        return filtered_tags, probs

    def find_tags_in_buckets(self, tags: dict) -> dict:
        """
        Finds the tags (and their scores) in their respective attribute buckets.

        Args:
            tags (dict): Dictionary of predicted tags -> confidence scores.

        Returns:
            dict:
              {
                "Hair": [ {"tag": "long_hair", "score": 0.78}, ... ],
                "Eyes": [ {"tag": "blue_eyes", "score": 0.91}, ... ],
                ...
              }
        """
        # For each bucket, we'll store a list of {"tag": <str>, "score": <float>} objects
        categorized_tags = {}

        for tag, score in tags.items():
            for bucket_name, bucket_tags in self.tag_buckets.items():
                if tag in bucket_tags:
                    if bucket_name not in categorized_tags:
                        categorized_tags[bucket_name] = []
                    categorized_tags[bucket_name].append({"tag": tag, "score": score})

        return categorized_tags

    def find_best_candidates(self, probs: np.ndarray) -> dict:
        """
        For each attribute bucket, pick the single highest scoring tag, ignoring score threshold.

        Args:
            probs (numpy.ndarray): The raw probability array from the model.predict output (before thresholding).
        
        Returns:
            dict: 
              {
                "Hair": {"tag": "long_hair", "score": 0.78},
                "Eyes": {"tag": "blue_eyes", "score": 0.91},
                ...
              }
            If no tags from that bucket are in the model's label set, that bucket is omitted.
        """
        best_candidates = {}
        # We need to check each bucket, find which tag in that bucket has the highest score.

        # Convert prob array -> label->prob
        label_probs = dict(zip(self.labels, probs))

        for bucket_name, bucket_tags in self.tag_buckets.items():
            # Among bucket_tags, find the one with highest label_probs
            candidate = None
            candidate_score = -1.0

            for t in bucket_tags:
                if t in label_probs and label_probs[t] > candidate_score:
                    candidate = t
                    candidate_score = label_probs[t]

            if candidate is not None and candidate_score >= 0.0:
                best_candidates[bucket_name] = {"tag": candidate, "score": float(candidate_score)}

        return best_candidates

    def predict_all(
        self, image: Union[str, PIL.Image.Image], threshold: float = 0.5
    ) -> dict:
        """
        Convenience method to:
          1) Predict tags above threshold
          2) Categorize them into buckets
          3) Also pick best candidate for each bucket ignoring threshold

        Args:
            image (Union[str, PIL.Image.Image]): Input image or image path.
            threshold (float): Minimum score threshold for returned tags.

        Returns:
          {
            "categorized_tags": { <bucket>: [{"tag": <str>, "score": <float>}, ...], ... },
            "best_candidates": { <bucket>: {"tag": <str>, "score": <float>}, ... }
          }
        """
        _, height, width, _ = self.model.input_shape

        # Load the image if a path is provided
        if isinstance(image, str):
            image = PIL.Image.open(image)

        # Convert image to a NumPy array
        image = np.asarray(image)
        image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
        image = image.numpy()
        # Transform and pad for model input
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.0  # Normalize

        # Run inference
        probs = self.model.predict(image[None, ...])[0].astype(float)
        # 1) thresholded tags
        thresholded_tags = {}
        indices = np.argsort(probs)[::-1]
        for i in indices:
            if probs[i] < threshold:
                break
            thresholded_tags[self.labels[i]] = probs[i]

        # 2) categorize thresholded tags
        cat_tags = self.find_tags_in_buckets(thresholded_tags)

        # 3) best candidate ignoring threshold
        best_cands = self.find_best_candidates(probs)

        return {
            "categorized_tags": cat_tags,
            "best_candidates": best_cands
        }
        image = np.asarray(image)
        image = tf.image.resize(image, size=(height, width), method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
        image = image.numpy()
        # Transform and pad for model input
        image = dd.image.transform_and_pad_image(image, width, height)
        image = image / 255.0  # Normalize

        # Run inference
        probs = self.model.predict(image[None, ...])[0].astype(float)
        # 1) thresholded tags
        thresholded_tags = {}
        indices = np.argsort(probs)[::-1]
        for i in indices:
            if probs[i] < threshold:
                break
            thresholded_tags[self.labels[i]] = probs[i]

        # 2) categorize thresholded tags
        cat_tags = self.find_tags_in_buckets(thresholded_tags)

        # 3) best candidate ignoring threshold
        best_cands = self.find_best_candidates(probs)

        return {
            "categorized_tags": cat_tags,
            "best_candidates": best_cands
        }

# Example usage
if __name__ == "__main__":
    tagger = DanbooruTagger(json_path="danbooru_bucket.json")
    
    # Load a sample image
    sample_image_path = "images/07.jpg"  # Replace with the actual image path
    image = PIL.Image.open(sample_image_path)

    # 1) Basic thresholded predictions
    thresholded, probs = tagger.predict_tags(image, score_threshold=0.4)
    # categorized = tagger.find_tags_in_buckets(thresholded)

    print("Thresholded Tags:\n", json.dumps(thresholded, indent=4))

    # print("Thresholded, Categorized Tags:\n", json.dumps(categorized, indent=4))

    # 2) Single best candidate ignoring threshold
    #    (We'll re-run the model so we get raw probabilities)
    #    Or use the convenience method 'predict_all'
    result = tagger.predict_all(image, threshold=0.4)
    print("\n\nCategorized result:\n", json.dumps(result["categorized_tags"], indent=4))
    print("\n\nBest candidates result:\n", json.dumps(result["best_candidates"], indent=4))

