from PIL import Image


def run_medgemma(model, image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
                "role": "system",
                "content": "You are a professional medical AI assistant specialized in dermatology benchmarking. Analyze the image and                 provide clinical differential diagnoses as requested."
            },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    output = model(
        text=messages,
        max_new_tokens=2000,
    )
    print("output-->>",output)

    return output[0]["generated_text"][-1]["content"]