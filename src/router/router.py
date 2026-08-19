import os
import json
import shutil
import tempfile
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException,BackgroundTasks
from fastapi.responses import JSONResponse

from model.model import model
from utils.utils import run_medgemma, cleanup_temp_dir,parse_derma_output
from utils.prompt import PROMPT_1,PROMPT_2
from src.config.config import IMAGE_TEMP_PATH

os.makedirs(IMAGE_TEMP_PATH, exist_ok=True)


router = APIRouter(tags=["Skin Image Analysis"])

@router.post("/predict-derma")
async def predict(file: UploadFile = File(...)):
    
    # ── 1. Check filename extension ────────────────────────────
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid file type: {ext}. Only png, jpg, and jpeg files are allowed."}
        )

    temp_dir = tempfile.mkdtemp()

    try:
        # ── 2. Save uploaded image ─────────────────────────────
        image_path = os.path.join(temp_dir, filename)
        with open(image_path, "wb") as f:
            f.write(await file.read())
            
        # ── 3. Run MedGemma inference ──────────────────────────
        raw_ai_text = run_medgemma(
            model=model,
            image_path=image_path,
            prompt=PROMPT_1
        )
        # print("raw ai text-->>",raw_ai_text)
        # ── 4. Parse output to robust JSON ─────────────────────
        # Try JSON extraction first, fallback to regex mapping if needed
        predictions = parse_derma_output(raw_ai_text)
        print("Backend output:- ", predictions)

        # ── 5. Create file_id and Save JSON ────────────────────
        file_id = str(uuid.uuid4())
        result_dir = os.path.join(IMAGE_TEMP_PATH, file_id)
        os.makedirs(result_dir, exist_ok=True)
        
        json_path = os.path.join(result_dir, "predictions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4)

        return JSONResponse(content={"file_id": file_id})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/json/{file_id}")
async def get_json_object(file_id: str, background_tasks: BackgroundTasks):
    json_path = os.path.join(IMAGE_TEMP_PATH, file_id, "predictions.json")
    result_dir = os.path.join(IMAGE_TEMP_PATH, file_id)

    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="File not found")

    # 1. Read the JSON completely into memory
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 2. Schedule the cleanup task to run AFTER the response is sent
    background_tasks.add_task(cleanup_temp_dir, result_dir)

    # 3. Return the data dictionary
    return data