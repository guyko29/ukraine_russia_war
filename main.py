from lib.generate_features import GenerateFeatures
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

if __name__ == "__main__":
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    print("Generating features for Ukraine...")
    ukraine_features = GenerateFeatures("Ukraine", huggingface_token).generate_features()

    print("Generating features for Russia...")
    russia_features = GenerateFeatures("Russia", huggingface_token).generate_features()

