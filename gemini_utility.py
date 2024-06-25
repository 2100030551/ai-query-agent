import os
import json
from PIL import Image

import google.generativeai as genai

# Get the current working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load configuration from config.json file
config_file_path = os.path.join(working_dir, "config.json")
with open(config_file_path, "r") as config_file:
    config_data = json.load(config_file)

# Retrieve the GOOGLE_API_KEY from the loaded configuration
GOOGLE_API_KEY = config_data.get("GOOGLE_API_KEY")

# Configure google.generativeai with the API key
genai.configure(api_key=GOOGLE_API_KEY)


def load_gemini_pro_model():
    """Load and return the Gemini-Pro generative model."""
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model


def gemini_pro_vision_response(prompt, image):
    """
    Get response from Gemini-Pro-Vision model - image/text to text.

    Args:
    - prompt (str): The text prompt for generation.
    - image (str): Path to the image file.

    Returns:
    - str: Generated text response from the model.
    """
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result


def embeddings_model_response(input_text):
    """
    Get response from embeddings model - text to embeddings.

    Args:
    - input_text (str): The text to embed.

    Returns:
    - list: List of embeddings generated from the model.
    """
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document"
    )
    embedding_list = embedding["embedding"]
    return embedding_list


def gemini_pro_response(user_prompt):
    """
    Get response from Gemini-Pro model - text to text.

    Args:
    - user_prompt (str): The prompt for text generation.

    Returns:
    - str: Generated text response from the model.
    """
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result
