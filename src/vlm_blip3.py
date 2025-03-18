import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import re
import json
from typing import List, Dict, Optional

class CharacterAttributeExtractor:
    def __init__(self, model_name="blip2_opt", model_type="pretrain_opt2.7b", device=None):
        """
        Initializes a BLIP-2 model (OPT variant) for question-based attribute extraction.
        model_name (str): e.g. "blip2_opt" 
        model_type (str): e.g. "pretrain_opt2.7b"
        """
        # Use GPU if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model and preprocess tools from LAVIS
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=self.device
        )

    def _ask_vlm(self, image: Image.Image, question: str) -> str:
        """
        Ask BLIP-2 (OPT) a question via `generate()`.
        Returns the model's text response.
        """
        # Preprocess image
        processed_image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

        # Optionally prepend "Question: ... Answer:" to encourage short, direct answers
        prompt_text = f"Question: {question} Answer:"
        
        with torch.no_grad():
            # For BLIP-2 OPT, we call `generate()` with {"image": ..., "prompt": ...}
            answer_list = self.model.generate({
                "image": processed_image,
                "prompt": prompt_text
            })

        # The returned `answer_list` is typically a list of strings
        return answer_list[0]

    # def extract_attributes(self, image_path: str, topics: List[str] =None, context: str = None) -> dict:
    #     """
    #     Extract structured attributes from an image by asking five broad questions.
    #     Returns a dict of the final parsed attributes.
    #     """
    #     # Load and convert the image
    #     image = Image.open(image_path).convert("RGB")

    #     # Define questions
    #     questions = {
    #         "Art Style": (
    #             "What is the art style? Choose one: anime, cartoon, semi-realistic, realistic, 3D-rendered."
    #         ),
    #         "Age": (
    #             "What is the character’s age group? Choose from: child, teen, young adult, middle-aged, elderly."
    #         ),
    #         "Gender": (
    #             "What is the character’s gender? Choose from: male or female."
    #         ),
    #         "Ethnicity": (
    #             "What is the character’s ethnicity? Choose from: Asian, African, Caucasian, Hispanic, other."
    #         ),
    #         "Hair Color": (
    #             "What is the character’s hair color? Choose from: black, blonde, red, blue, green, brown, white, gray."
    #         ),
    #         "Hair Style": (
    #             "What is the character’s hair style? Choose from: ponytail, curly, straight, bun, braided, short bob."
    #         ),
    #         "Hair Length": (
    #             "What is the character’s hair length? Choose from: short, medium, long."
    #         ),
    #         "Eye Color": (
    #             "What is the character’s eye color? Choose from: black, brown, blue, green, gray, hazel, red."
    #         ),
    #         "Body Type": (
    #             "What is the character’s body type? Choose from: slim, muscular, curvy, average."
    #         ),
    #         "Dress": (
    #             "What is the character’s outfit style? Choose from: casual, formal, traditional, futuristic, uniform."
    #         ),
    #         "Dress Description": (
    #             "Describe the character’s outfit"
    #         ),
    #         "Facial Expression": (
    #             "What is the character’s facial expression? Choose from: neutral, smiling, serious, surprised, sad."
    #         ),
    #         "Accessories & Unique Traits": (
    #             "Does the character have any unique traits? Choose from: scars, tattoos, glasses, hat, jewelry, none."
    #         )
    #     }
        
    #     if topics:
    #         questions = {k: v for k, v in questions.items() if k in topics}
    #     # Ask each question
    #     answers = {}
    #     for key, q in questions.items():
    #         answers[key] = self._ask_vlm(image, q)
    #     print(answers)
    #     # Parse the final answers into structured fields
    #     parsed = {}
    #     for attr, q in questions.items():
    #         parsed[attr] = answers[attr] if answers[attr] in q else "Unknown"
    #     return parsed


    def extract_attributes(self, image_path: str, topics: Optional[List[str]] = None, context: str = None) -> Dict[str, str]:
        """
        Extract structured attributes from an image by asking broad questions.
        Supports optional filtering via 'topics' and integrates previous 'context'.
        Returns a dictionary of extracted attributes.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Define attribute-based questions
        questions = {
            "Art Style": "What is the art style? Choose one: anime, cartoon, semi-realistic, realistic, 3D-rendered.",
            "Age": "What is the character’s age group? Choose from: child, teen, young adult, middle-aged, elderly.",
            "Gender": "What is the character’s gender? Choose from: male or female.",
            "Ethnicity": "What is the character’s ethnicity? Choose from: Asian, African, Caucasian, Hispanic, other.",
            "Hair Color": "What is the character’s hair color? Choose from: black, blonde, red, blue, green, brown, white, gray.",
            "Hair Style": "What is the character’s hair style? Choose from: ponytail, curly, straight, bun, braided, short bob.",
            "Hair Length": "What is the character’s hair length? Choose from: short, medium, long.",
            "Eye Color": "What is the character’s eye color? Choose from: black, brown, blue, green, gray, hazel, red.",
            "Body Type": "What is the character’s body type? Choose from: slim, muscular, curvy, average.",
            "Dress & Clothing": "What is the character’s outfit style? Choose from: casual, formal, traditional, futuristic, uniform.",
            "Facial Expression": "What is the character’s facial expression? Choose from: neutral, smiling, serious, surprised, sad.",
            "Accessories & Unique Traits": "Does the character have any unique traits? Choose from: scars, tattoos, glasses, hat, jewelry, none."
        }

        # Filter questions based on selected topics
        if topics:
            questions = {k: v for k, v in questions.items() if k in topics}

        # Store raw answers
        answers = {}

        # Ask BLIP-2 model each question
        for key, question in questions.items():
            if context:
                prompt = f"{context} Question: {question}"
            else:
                prompt = f"Question: {question}"

            # Call BLIP-2 or Vision-Language Model
            answers[key] = self._ask_vlm(image, prompt)

        # Parse the final answers into structured fields
        parsed = {}
        for attr, question in questions.items():
            # Ensure the answer is within the predefined choices
            if any(option in answers[attr].lower() for option in question.lower()):
                parsed[attr] = answers[attr]
            else:
                parsed[attr] = "Unknown"

        return parsed

if __name__ == "__main__":
    # Replace with your image path
    image_path = "data/continued/sensitive/danbooru_1380747_fcc57517b1c5073ea341c5d0cc0c1797.jpg"

    # Initialize BLIP-2 (OPT) model
    extractor = CharacterAttributeExtractor(
        model_name="blip2_t5",
        model_type="pretrain_flant5xl"
    )

    # Extract attributes
    attributes = extractor.extract_attributes(image_path)

    # Print output in JSON
    print(json.dumps(attributes, indent=4))
