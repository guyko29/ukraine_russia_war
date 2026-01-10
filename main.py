from lib.generate_features import GenerateFeatures
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

if __name__ == "__main__":
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    print("Generating features for Ukraine...")
    ukraine_helper_dict = GenerateFeatures("Ukraine", huggingface_token).generate_features()

    print("Generating features for Russia...")
    russia_helper_dict = GenerateFeatures("Russia", huggingface_token, ukraine_helper_dict).generate_features()

