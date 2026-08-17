import os
# pyrefly: ignore [missing-import]
from transformers import pipeline
import torch

HF_TOKEN = os.getenv("HF_TOKEN")

def load_model():
    model = pipeline(
        "image-text-to-text",
        model="google/medgemma-1.5-4b-it",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model