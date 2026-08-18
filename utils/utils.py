import re
from PIL import Image
import os
import shutil
import json
from src.config.config import ABNORMALITY_MAP_DERMA, NEGATION_PATTERNS

def run_medgemma(model, image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
                "role": "system",
                "content": "You are a professional medical AI assistant specialized in dermatology benchmarking. Analyze the image and provide clinical differential diagnoses as requested."
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

    return output[0]["generated_text"][-1]["content"]

def is_negated(sentence):
    sentence = sentence.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, sentence):
            return True
    return False

def extract_abnormalities(text, abnormalities_map):
    abnormalities = []
    # Split text by periods, commas, or newlines to isolate context
    sentences = re.split(r"[.;\n,]+", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check if the specific chunk contains a negation word
        negated = is_negated(sentence)
        
        for label, pattern in abnormalities_map.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                # Only add if the sentence was NOT negated
                if not negated:
                    abnormalities.append(label)
    print("abnormalities-->>",list(set(abnormalities)))                
    return sorted(list(set(abnormalities)))

def is_definitively_normal(text):
    """
    Checks if the AI explicitly declared 'Normal' as the primary diagnosis,
    allowing us to safely ignore its differential diagnosis list.
    """
    normal_overrides = [
        r"(?i)most likely condition is\s*\*?\*?\s*normal",
        r"(?i)most likely diagnosis\s*(?:is|:)?\s*\*?\*?\s*normal",
        r"(?i)diagnosis:\s*\*?\*?\s*normal",
        r"(?i)condition:\s*\*?\*?\s*normal",
        r"(?i)without any obvious signs of pathology",
        r"(?i)no signs of pathology"
    ]
    for pattern in normal_overrides:
        if re.search(pattern, text):
            return True
    return False

def derma_report_to_json(report_text):
    # 1. SMART OVERRIDE: If the model explicitly says it's normal, force normal True
    if is_definitively_normal(report_text):
        return {
            "normal": True,
            "abnormality": [],
            "findings": report_text.strip()
        }
    # Extract abnormalities using the robust negated regex map
    abnormalities = extract_abnormalities(report_text, ABNORMALITY_MAP_DERMA)
    
    # Determine normality
    is_normal = False
    if len(abnormalities) == 0:
        # If no abnormalities found, check if model explicitly stated normal
        if re.search(r"(?i)\b(?:normal|healthy|clear)\b", report_text):
            is_normal = True
            
    # Assemble the final payload, retaining the entire AI response for the backend
    return {
        "normal": is_normal,
        "abnormality": abnormalities,
        "findings": report_text.strip()
    }



def parse_derma_output(response_text):
    """
    Attempts to parse direct JSON from the model. 
    Falls back to regex extraction if the model returns plain text.
    """
    # 1. Try to extract and parse JSON
    try:
        # Strip potential markdown fences just in case the model ignores the "No markdown" rule
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_string = clean_text[start_idx:end_idx+1]
            parsed_json = json.loads(json_string)
            
            # Ensure the required keys are present before trusting the JSON
            if "normal" in parsed_json and "abnormality" in parsed_json and "findings" in parsed_json:
                print("Successfully parsed direct JSON from model.")
                # Ensure abnormality is a list
                if not isinstance(parsed_json["abnormality"], list):
                    parsed_json["abnormality"] = [parsed_json["abnormality"]]
                return parsed_json
                
    except json.JSONDecodeError:
        print("JSON parse failed. Falling back to text regex extraction.")
        pass

    # 2. Fallback to the robust regex parser (from our previous step)
    print("Executing fallback text parser...by derma_report_to_json")
    return derma_report_to_json(response_text)


def cleanup_temp_dir(path: str):
    """Deletes the temporary directory after the response is successfully sent."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"🧹 Cleanup successful: Deleted {path}")
    except Exception as e:
        print(f"❌ Cleanup failed for {path}: {str(e)}")