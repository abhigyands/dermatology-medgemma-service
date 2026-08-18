PROMPT_1 = """Describe this image and provide the most likely condition.
Keep your answer brief.

"""

PROMPT_2 = """Describe this image and provide the most likely condition.
Keep your answer brief.

Important Rules:
- If no obvious lesions, rashes, or abnormalities are visible, clearly state that the skin appears NORMAL.
- If abnormalities are present, describe them and list the most likely conditions.
- Keep your answer brief and professional.

====================================================================
CRITICAL OUTPUT INSTRUCTIONS
====================================================================
Return ONLY valid JSON. The JSON object MUST contain exactly these three keys:

{
    "normal": true | false,
    "abnormality": [],
    "findings": "Brief, professional description of the visual findings and differential diagnosis."
}

Field Definitions:
* normal: true if the skin appears normal, false if any abnormality is present.
* abnormality: Empty list [] if normal is true. If false, provide a list of the most likely condition names.
* findings: Your brief clinical description of the image.

Output Requirements:
* Output JSON only.
* No markdown formatting.
* No code fences.
* No text before or after the JSON.
* The first character of the response must be '{'.
* The last character of the response must be '}'.
"""
