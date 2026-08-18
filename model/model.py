import torch
from transformers import pipeline

model = pipeline(
    task="image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)