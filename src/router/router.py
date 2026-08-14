import os
import glob
import json
import shutil
import tempfile
import uuid

from zipfile import ZipFile

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from model.model import load_model
from utils.utils import run_medgemma
from utils.prompt import PROMPT_1
from src.config.config import IMAGE_TEMP_PATH


os.makedirs(IMAGE_TEMP_PATH, exist_ok=True)


model = load_model()


router = APIRouter(
    tags=["Skin Image Analysis"]
)


@router.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # ── 1. Check filename extension ────────────────────────────
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid file type: {ext}. Only PNG, JPG, and JPEG files are allowed."
            }
        )

    temp_dir = tempfile.mkdtemp()

    try:

        # ── 2. Save uploaded image ─────────────────────────────
        image_path = os.path.join(
            temp_dir,
            filename
        )
        with open(image_path, "wb") as f:
            f.write(await file.read())
            
        # ── 3. Run MedGemma inference ──────────────────────────
        result_1 = run_medgemma(
            model=model,
            image_path=image_path,
            prompt=PROMPT_1
        )

        predictions = [
            {
                "image": filename,
                "prediction": result_1,
            }
        ]

        # ── 4. Create file_id ──────────────────────────────────
        file_id = str(uuid.uuid4())

        result_dir = os.path.join(
            IMAGE_TEMP_PATH,
            file_id
        )

        os.makedirs(
            result_dir,
            exist_ok=True
        )

        # ── 5. Save predictions.json ───────────────────────────
        json_path = os.path.join(
            result_dir,
            "predictions.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "predictions": predictions
                },
                f,
                indent=4
            )

        # ── 6. Return file_id ──────────────────────────────────
        return JSONResponse(
            content={
                "file_id": file_id
            }
        )


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


    finally:

        # remove uploaded image
        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )



@router.get("/json/{file_id}")
async def get_json_object(
    file_id: str
):

    json_path = os.path.join(
        IMAGE_TEMP_PATH,
        file_id,
        "predictions.json"
    )


    result_dir = os.path.join(
        IMAGE_TEMP_PATH,
        file_id
    )


    try:

        if not os.path.exists(json_path):

            raise HTTPException(
                status_code=404,
                detail="File not found"
            )


        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)


    finally:

        # cleanup after retrieval
        if os.path.exists(result_dir):

            shutil.rmtree(
                result_dir,
                ignore_errors=True
            )