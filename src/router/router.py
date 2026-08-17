import os
import glob
import json
import re
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


@router.post("/predict-derma")
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

        # 1. Determine Normal status from the generated text
        normal = False
        has_negative_normal = re.search(r"(?i)\b(?:not|never|isn't|is not|aren't|no longer)\s+(?:normal|healthy|clear)\b", result_1)
        
        # Check if any abnormal keyword is present AND is NOT negated
        abnormal_keywords = ["abnormal", "unhealthy", "lesion", "rash", "disease", "infection", "dermatitis", "melasma", "nigricans", "eczema", "psoriasis", "erythema"]
        found_abnormal = False
        for kw in abnormal_keywords:
            kw_matches = list(re.finditer(r"(?i)\b" + re.escape(kw) + r"(?:s|es|ies)?\b", result_1))
            for match in kw_matches:
                # Find the start of the clause/sentence containing this match
                boundary_match = list(re.finditer(r"[.;!]", result_1[:match.start()]))
                clause_start = boundary_match[-1].end() if boundary_match else 0
                clause_text = result_1[clause_start:match.start()].lower()
                
                # Check for negation words in the clause before the keyword
                if not re.search(r"\b(?:no|without|free\s+of|negative\s+for|absence\s+of|clear\s+of|no\s+obvious|no\s+clear|no\s+active|no\s+signs\s+of|deny|denies|not)\b", clause_text):
                    found_abnormal = True
                    break
            if found_abnormal:
                break
        
        if not (has_negative_normal or found_abnormal):
            if re.search(r"(?i)\b(?:normal|healthy|clear)\b", result_1):
                normal = True

        # 2. Extract abnormality list from the generated text
        abnormality = []
        condition_text = ""
        
        m1 = re.search(r"(?i)most likely condition(?:\s+is)?\s*[:*]*\s*([^*.\n]+)", result_1)
        if m1:
            condition_text = m1.group(1).strip()
        else:
            m2 = re.search(r"(?i)\bprediction\s*:\s*([^\n]+)", result_1)
            if m2:
                condition_text = m2.group(1).strip()
            else:
                m3 = re.search(r"(?i)\bdiagnosis\s*:\s*([^\n]+)", result_1)
                if m3:
                    condition_text = m3.group(1).strip()

        if condition_text:
            condition_text = condition_text.replace("**", "").replace("*", "").strip()
            condition_text = re.sub(r"(?i).*?\bsuggesting\b\s*", "", condition_text)
            condition_text = re.sub(r"(?i).*?\bsuggestive of\b\s*", "", condition_text)
            condition_text = re.sub(r"(?i).*?\bsuggests\b\s*", "", condition_text)
            
            parts = re.split(r',|;|\bor\b|\band\b', condition_text, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                # Skip if the part itself is a negation or contains negation words
                if re.search(r"(?i)\b(?:no|none|without|negative|absence|clear|free|not)\b", part):
                    continue
                part = re.sub(r"(?i)^\s*(?:a|an|the|possibly|likely|suggests a|suggesting a)\b\s*", "", part)
                part = re.sub(r"(?i)\s+(?:lesion|lesions|condition|disease|skin condition)\s*$", "", part)
                part = part.strip()
                if part and part.lower() not in ("none", "normal", "healthy", "n/a", "a", "an", "the", "mark"):
                    abnormality.append(part)

        if not abnormality:
            known_conditions = {
                # =========================
                # FUNGAL INFECTIONS
                # =========================
                "tinea": [
                    "tinea",
                    "ringworm",
                    "dermatophytosis",
                    "dermatophyte infection",
                    "fungal infection",
                    "superficial fungal infection",
                    "dermatophycosis"
                ],

                "tinea corporis": [
                    "tinea corporis",
                    "body ringworm",
                    "ringworm of the body",
                    "corporis"
                ],

                "tinea cruris": [
                    "tinea cruris",
                    "jock itch",
                    "groin ringworm",
                    "groin fungal infection",
                    "cruris"
                ],

                "tinea pedis": [
                    "tinea pedis",
                    "athlete's foot",
                    "athletes foot",
                    "foot fungus",
                    "foot fungal infection"
                ],

                "tinea faciei": [
                    "tinea faciei",
                    "facial ringworm",
                    "face ringworm",
                    "facial fungal infection"
                ],

                "tinea capitis": [
                    "tinea capitis",
                    "scalp ringworm",
                    "ringworm of scalp",
                    "scalp fungal infection"
                ],

                "tinea versicolor": [
                    "tinea versicolor",
                    "pityriasis versicolor",
                    "pityriasis versicolor",
                    "versicolor",
                    "malassezia infection"
                ],

                "onychomycosis": [
                    "onychomycosis",
                    "fungal nail infection",
                    "nail fungus",
                    "tinea unguium"
                ],

                "candidiasis": [
                    "candidiasis",
                    "cutaneous candidiasis",
                    "cutaneous candida",
                    "candida infection",
                    "yeast infection"
                ],

                # =========================
                # PARASITIC INFECTIONS
                # =========================
                "scabies": [
                    "scabies",
                    "scabies infestation",
                    "sarcoptic mange",
                    "sarcoptes scabiei",
                    "sarcoptes infestation"
                ],

                "pediculosis": [
                    "pediculosis",
                    "head lice",
                    "lice infestation",
                    "pediculosis capitis",
                    "pediculosis corporis",
                    "pediculosis pubis"
                ],

                # =========================
                # BACTERIAL INFECTIONS
                # =========================
                "impetigo": [
                    "impetigo",
                    "bullous impetigo",
                    "nonbullous impetigo"
                ],

                "folliculitis": [
                    "folliculitis",
                    "bacterial folliculitis",
                    "inflamed hair follicles",
                    "follicular infection"
                ],

                "furuncle": [
                    "furuncle",
                    "boil",
                    "furunculosis",
                    "skin boil"
                ],

                "carbuncle": [
                    "carbuncle",
                    "carbunculosis"
                ],

                "cellulitis": [
                    "cellulitis",
                    "skin cellulitis",
                    "cutaneous cellulitis"
                ],

                "leprosy": [
                    "leprosy",
                    "Hansen's disease",
                    "Hansen disease",
                    "Hansen's leprosy",
                    "Mycobacterium leprae infection"
                ],

                # =========================
                # ACNE / FOLLICULAR
                # =========================
                "acne": [
                    "acne",
                    "acne vulgaris",
                    "vulgar acne",
                    "pimples",
                    "pimple",
                    "comedonal acne",
                    "inflammatory acne",
                    "cystic acne",
                    "nodulocystic acne"
                ],

                "acne keloidalis nuchae": [
                    "acne keloidalis nuchae",
                    "akn",
                    "folliculitis keloidalis",
                    "keloidal folliculitis",
                    "nuchal keloid"
                ],

                "hidradenitis suppurativa": [
                    "hidradenitis suppurativa",
                    "HS",
                    "acne inversa",
                    "inverse acne"
                ],

                # =========================
                # DERMATITIS / ECZEMA
                # =========================
                "dermatitis": [
                    "dermatitis",
                    "eczema",
                    "eczematous dermatitis",
                    "eczematous eruption"
                ],

                "atopic dermatitis": [
                    "atopic dermatitis",
                    "atopic eczema",
                    "AD",
                    "infantile eczema"
                ],

                "contact dermatitis": [
                    "contact dermatitis",
                    "contact eczema",
                    "allergic contact dermatitis",
                    "irritant contact dermatitis",
                    "ACD",
                    "ICD"
                ],

                "seborrheic dermatitis": [
                    "seborrheic dermatitis",
                    "seborrhoeic dermatitis",
                    "seborrheic eczema",
                    "seborrhoeic eczema",
                    "dandruff",
                    "pityriasis capitis"
                ],

                "nummular eczema": [
                    "nummular eczema",
                    "nummular dermatitis",
                    "discoid eczema",
                    "discoid dermatitis"
                ],

                "dyshidrotic eczema": [
                    "dyshidrotic eczema",
                    "dyshidrosis",
                    "pompholyx",
                    "vesicular hand eczema"
                ],

                "lichen simplex chronicus": [
                    "lichen simplex chronicus",
                    "LSC",
                    "neurodermatitis",
                    "circumscribed neurodermatitis"
                ],

                # =========================
                # PSORIASIS
                # =========================
                "psoriasis": [
                    "psoriasis",
                    "psoriasis vulgaris",
                    "chronic plaque psoriasis",
                    "plaque psoriasis",
                    "psoriatic disease"
                ],

                "guttate psoriasis": [
                    "guttate psoriasis",
                    "guttate psoriatic eruption"
                ],

                "pustular psoriasis": [
                    "pustular psoriasis",
                    "generalized pustular psoriasis",
                    "GPP"
                ],

                # =========================
                # PIGMENTARY DISORDERS
                # =========================
                "melasma": [
                    "melasma",
                    "chloasma",
                    "mask of pregnancy",
                    "facial melasma",
                    "malar melasma"
                ],

                "post inflammatory hyperpigmentation": [
                    "post inflammatory hyperpigmentation",
                    "post-inflammatory hyperpigmentation",
                    "PIH",
                    "postinflammatory pigmentation"
                ],

                "hyperpigmentation": [
                    "hyperpigmentation",
                    "hyperpigmented skin",
                    "increased pigmentation",
                    "dark pigmentation",
                    "skin darkening"
                ],

                "hypopigmentation": [
                    "hypopigmentation",
                    "hypopigmented lesion",
                    "hypopigmented patch",
                    "loss of pigmentation"
                ],

                "vitiligo": [
                    "vitiligo",
                    "vitiligo vulgaris",
                    "leukoderma",
                    "acquired leukoderma",
                    "depigmentation"
                ],

                "pityriasis alba": [
                    "pityriasis alba",
                    "pityriasis alba faciei",
                    "white patches",
                    "hypopigmented facial patches"
                ],

                "acanthosis nigricans": [
                    "acanthosis nigricans",
                    "AN",
                    "acanthosis"
                ],

                # =========================
                # URTICARIA / ALLERGIC
                # =========================
                "urticaria": [
                    "urticaria",
                    "hives",
                    "wheals",
                    "nettle rash",
                    "urticarial eruption"
                ],

                "angioedema": [
                    "angioedema",
                    "angioneurotic edema",
                    "angioneurotic oedema"
                ],

                # =========================
                # HAIR DISORDERS
                # =========================
                "alopecia areata": [
                    "alopecia areata",
                    "AA",
                    "patchy hair loss",
                    "localized hair loss",
                    "spot baldness"
                ],

                "androgenetic alopecia": [
                    "androgenetic alopecia",
                    "androgenic alopecia",
                    "male pattern baldness",
                    "male pattern hair loss",
                    "female pattern hair loss",
                    "female pattern alopecia",
                    "pattern hair loss"
                ],

                "telogen effluvium": [
                    "telogen effluvium",
                    "diffuse hair shedding",
                    "diffuse hair loss",
                    "hair shedding"
                ],

                "alopecia": [
                    "alopecia",
                    "hair loss",
                    "hair fall",
                    "baldness"
                ],

                # =========================
                # VIRAL
                # =========================
                "warts": [
                    "wart",
                    "warts",
                    "verruca",
                    "verrucae",
                    "viral wart",
                    "viral verruca"
                ],

                "molluscum contagiosum": [
                    "molluscum contagiosum",
                    "molluscum",
                    "molluscum lesions"
                ],

                "herpes zoster": [
                    "herpes zoster",
                    "shingles",
                    "zoster",
                    "varicella zoster",
                    "varicella-zoster virus infection"
                ],

                "herpes simplex": [
                    "herpes simplex",
                    "HSV",
                    "oral herpes",
                    "genital herpes",
                    "herpetic infection"
                ],

                "chickenpox": [
                    "chickenpox",
                    "varicella",
                    "varicella zoster infection",
                    "varicella infection"
                ],

                # =========================
                # PAPULOSQUAMOUS
                # =========================
                "lichen planus": [
                    "lichen planus",
                    "LP",
                    "cutaneous lichen planus"
                ],

                "pityriasis rosea": [
                    "pityriasis rosea",
                    "roseola annulata",
                    "herald patch"
                ],

                # =========================
                # ROSACEA
                # =========================
                "rosacea": [
                    "rosacea",
                    "acne rosacea",
                    "adult acne rosacea",
                    "erythematotelangiectatic rosacea"
                ],

                # =========================
                # KERATINIZATION
                # =========================
                "keratosis pilaris": [
                    "keratosis pilaris",
                    "KP",
                    "follicular keratosis",
                    "chicken skin"
                ],

                "ichthyosis vulgaris": [
                    "ichthyosis vulgaris",
                    "ichthyosis",
                    "fish scale skin",
                    "fish skin disease"
                ],

                # =========================
                # CYSTS / BENIGN LESIONS
                # =========================
                "pilar cyst": [
                    "pilar cyst",
                    "trichilemmal cyst",
                    "wen",
                    "scalp cyst"
                ],

                "epidermoid cyst": [
                    "epidermoid cyst",
                    "epidermal inclusion cyst",
                    "sebaceous cyst",
                    "epidermal cyst"
                ],

                "lipoma": [
                    "lipoma",
                    "fatty tumor",
                    "benign fatty tumor"
                ],

                "keloid": [
                    "keloid",
                    "keloidal scar",
                    "keloid scar"
                ],

                # =========================
                # PRE-CANCEROUS / CANCER
                # =========================
                "basal cell carcinoma": [
                    "basal cell carcinoma",
                    "BCC",
                    "basal cell cancer",
                    "basal cell skin cancer"
                ],

                "squamous cell carcinoma": [
                    "squamous cell carcinoma",
                    "SCC",
                    "cutaneous squamous cell carcinoma",
                    "squamous cell skin cancer"
                ],

                "melanoma": [
                    "melanoma",
                    "malignant melanoma",
                    "cutaneous melanoma"
                ],

                "actinic keratosis": [
                    "actinic keratosis",
                    "solar keratosis",
                    "senile keratosis",
                    "AK"
                ],

                # =========================
                # INFLAMMATORY / VASCULAR
                # =========================
                "erythema nodosum": [
                    "erythema nodosum",
                    "EN",
                    "nodular panniculitis"
                ],

                "erythema ab igne": [
                    "erythema ab igne",
                    "toasted skin syndrome",
                    "heat-induced reticulate erythema"
                ],

                "vasculitis": [
                    "vasculitis",
                    "cutaneous vasculitis",
                    "leukocytoclastic vasculitis",
                    "cutaneous small vessel vasculitis"
                ],

                # =========================
                # COMMON OTHER CONDITIONS
                # =========================
                "miliaria": [
                    "miliaria",
                    "prickly heat",
                    "heat rash",
                    "sweat rash",
                    "miliaria rubra"
                ],

                "pruritus": [
                    "pruritus",
                    "itching",
                    "itch",
                    "skin itching"
                ],

                "nevus": [
                    "nevus",
                    "naevus",
                    "mole",
                    "melanocytic nevus",
                    "benign mole"
                ],

                "skin lesion": [
                    "skin lesion",
                    "cutaneous lesion",
                    "lesion",
                    "skin growth",
                    "cutaneous abnormality"
                ],

                "rash": [
                    "rash",
                    "skin rash",
                    "cutaneous rash",
                    "eruption",
                    "skin eruption"
                ],

                "erythema": [
                    "erythema",
                    "redness",
                    "skin redness",
                    "erythematous lesion"
                ]
            }
            result_lower = result_1.lower()
            for canonical, synonyms in known_conditions.items():
                for syn in synonyms:
                    # Find all occurrences of the synonym
                    syn_matches = list(re.finditer(r"(?i)\b" + re.escape(syn) + r"(?:s|es|ies)?\b", result_1))
                    non_negated_match_found = False
                    for match in syn_matches:
                        # Find the clause containing this match
                        boundary_match = list(re.finditer(r"[.;!]", result_1[:match.start()]))
                        clause_start = boundary_match[-1].end() if boundary_match else 0
                        clause_text = result_1[clause_start:match.start()].lower()
                        
                        # If the synonym is NOT negated, we count it
                        if not re.search(r"\b(?:no|without|free\s+of|negative\s+for|absence\s+of|clear\s+of|no\s+obvious|no\s+clear|no\s+active|no\s+signs\s+of|deny|denies|not)\b", clause_text):
                            non_negated_match_found = True
                            break
                    if non_negated_match_found:
                        abnormality.append(canonical)
                        break

        seen = set()
        abnormality = [x for x in abnormality if not (x.lower() in seen or seen.add(x.lower()))]

        # 3. Clean and extract only the findings/description block for "ai predictions"
        ai_predictions = result_1
        findings_match = re.search(r"((?:The|This) (?:image|skin) shows[\s\S]*?)(?=\n\n|\r\n\r\n|\n[A-Za-z ]+:|\Z)", result_1, re.IGNORECASE)
        if findings_match:
            ai_predictions = findings_match.group(1).strip()
        else:
            ai_predictions = re.sub(r"(?i)\bJSON output:\s*$", "", ai_predictions).strip()

        predictions = {
                "normal": normal,
                "abnormality": abnormality,
                "findings": ai_predictions,
                }
        

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
                
                predictions
                ,
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