import os

IMAGE_TEMP_PATH = '/home/ubuntu/Documents/dermatology-medgemma-service/tmp'
MODEL_NAME = "google/medgemma-1.5-4b-it"

# Negation patterns to prevent false positives on normal images
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bno evidence of\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bnot seen\b",
    r"\bfree of\b",
    r"\bno obvious\b",
    r"\blacks?\b",              # Catches "the image lacks these features"
    r"\bnot definitively\b",    # Catches "not definitively visible"
    r"\bunlikely\b",            # Catches "is unlikely"
    r"\bnot apparent\b"         # Catches "no scaling is apparent"
]

# Comprehensive Dermatology Abnormality Regex Map
ABNORMALITY_MAP_DERMA = {
    # =========================
    # FUNGAL INFECTIONS
    # =========================
    "tinea": r"\b(?:tinea|ringworm|dermatophytosis|dermatophyte infection|fungal infection|superficial fungal infection|dermatophycosis)\b",
    "tinea corporis": r"\b(?:tinea corporis|body ringworm|ringworm of the body|corporis)\b",
    "tinea cruris": r"\b(?:tinea cruris|jock itch|groin ringworm|groin fungal infection|cruris)\b",
    "tinea pedis": r"\b(?:tinea pedis|athlete's foot|athletes foot|foot fungus|foot fungal infection)\b",
    "tinea faciei": r"\b(?:tinea faciei|facial ringworm|face ringworm|facial fungal infection)\b",
    "tinea capitis": r"\b(?:tinea capitis|scalp ringworm|ringworm of scalp|scalp fungal infection)\b",
    "tinea versicolor": r"\b(?:tinea versicolor|pityriasis versicolor|versicolor|malassezia infection)\b",
    "onychomycosis": r"\b(?:onychomycosis|fungal nail infection|nail fungus|tinea unguium)\b",
    "candidiasis": r"\b(?:candidiasis|cutaneous candidiasis|cutaneous candida|candida infection|yeast infection)\b",

    # =========================
    # PARASITIC INFECTIONS
    # =========================
    "scabies": r"\b(?:scabies|scabies infestation|sarcoptic mange|sarcoptes scabiei|sarcoptes infestation)\b",
    "pediculosis": r"\b(?:pediculosis|head lice|lice infestation|pediculosis capitis|pediculosis corporis|pediculosis pubis)\b",

    # =========================
    # BACTERIAL INFECTIONS
    # =========================
    "impetigo": r"\b(?:impetigo|bullous impetigo|nonbullous impetigo)\b",
    "folliculitis": r"\b(?:folliculitis|bacterial folliculitis|inflamed hair follicles|follicular infection)\b",
    "furuncle": r"\b(?:furuncle|boil|furunculosis|skin boil)\b",
    "carbuncle": r"\b(?:carbuncle|carbunculosis)\b",
    "cellulitis": r"\b(?:cellulitis|skin cellulitis|cutaneous cellulitis)\b",
    "leprosy": r"\b(?:leprosy|Hansen's disease|Hansen disease|Hansen's leprosy|Mycobacterium leprae infection)\b",

    # =========================
    # ACNE / FOLLICULAR
    # =========================
    "acne": r"\b(?:acne|acne vulgaris|vulgar acne|pimples|pimple|comedonal acne|inflammatory acne|cystic acne|nodulocystic acne)\b",
    "acne keloidalis nuchae": r"\b(?:acne keloidalis nuchae|akn|folliculitis keloidalis|keloidal folliculitis|nuchal keloid)\b",
    "hidradenitis suppurativa": r"\b(?:hidradenitis suppurativa|HS|acne inversa|inverse acne)\b",

    # =========================
    # DERMATITIS / ECZEMA
    # =========================
    "dermatitis": r"\b(?:dermatitis|eczema|eczematous dermatitis|eczematous eruption)\b",
    "atopic dermatitis": r"\b(?:atopic dermatitis|atopic eczema|infantile eczema)\b",
    "contact dermatitis": r"\b(?:contact dermatitis|contact eczema|allergic contact dermatitis|irritant contact dermatitis|ACD|ICD)\b",
    "seborrheic dermatitis": r"\b(?:seborrheic dermatitis|seborrhoeic dermatitis|seborrheic eczema|seborrhoeic eczema|dandruff|pityriasis capitis)\b",
    "nummular eczema": r"\b(?:nummular eczema|nummular dermatitis|discoid eczema|discoid dermatitis)\b",
    "dyshidrotic eczema": r"\b(?:dyshidrotic eczema|dyshidrosis|pompholyx|vesicular hand eczema)\b",
    "lichen simplex chronicus": r"\b(?:lichen simplex chronicus|LSC|neurodermatitis|circumscribed neurodermatitis)\b",

    # =========================
    # PSORIASIS
    # =========================
    "psoriasis": r"\b(?:psoriasis|psoriasis vulgaris|chronic plaque psoriasis|plaque psoriasis|psoriatic disease)\b",
    "guttate psoriasis": r"\b(?:guttate psoriasis|guttate psoriatic eruption)\b",
    "pustular psoriasis": r"\b(?:pustular psoriasis|generalized pustular psoriasis|GPP)\b",

    # =========================
    # PIGMENTARY DISORDERS
    # =========================
    "melasma": r"\b(?:melasma|chloasma|mask of pregnancy|facial melasma|malar melasma)\b",
    "post inflammatory hyperpigmentation": r"\b(?:post inflammatory hyperpigmentation|post-inflammatory hyperpigmentation|PIH|postinflammatory pigmentation)\b",
    "hyperpigmentation": r"\b(?:hyperpigmentation|hyperpigmented skin|increased pigmentation|dark pigmentation|skin darkening)\b",
    "hypopigmentation": r"\b(?:hypopigmentation|hypopigmented lesion|hypopigmented patch|loss of pigmentation)\b",
    "vitiligo": r"\b(?:vitiligo|vitiligo vulgaris|leukoderma|acquired leukoderma|depigmentation)\b",
    "pityriasis alba": r"\b(?:pityriasis alba|pityriasis alba faciei|white patches|hypopigmented facial patches)\b",
    "acanthosis nigricans": r"\b(?:acanthosis nigricans|acanthosis)\b",

    # =========================
    # URTICARIA / ALLERGIC
    # =========================
    "urticaria": r"\b(?:urticaria|hives|wheals|nettle rash|urticarial eruption)\b",
    "angioedema": r"\b(?:angioedema|angioneurotic edema|angioneurotic oedema)\b",

    # =========================
    # HAIR DISORDERS
    # =========================
    "alopecia areata": r"\b(?:alopecia areata|AA|patchy hair loss|localized hair loss|spot baldness)\b",
    "androgenetic alopecia": r"\b(?:androgenetic alopecia|androgenic alopecia|male pattern baldness|male pattern hair loss|female pattern hair loss|female pattern alopecia|pattern hair loss)\b",
    "telogen effluvium": r"\b(?:telogen effluvium|diffuse hair shedding|diffuse hair loss|hair shedding)\b",
    "alopecia": r"\b(?:alopecia|hair loss|hair fall|baldness)\b",

    # =========================
    # VIRAL
    # =========================
    "warts": r"\b(?:wart|warts|verruca|verrucae|viral wart|viral verruca)\b",
    "molluscum contagiosum": r"\b(?:molluscum contagiosum|molluscum|molluscum lesions)\b",
    "herpes zoster": r"\b(?:herpes zoster|shingles|zoster|varicella zoster|varicella-zoster virus infection)\b",
    "herpes simplex": r"\b(?:herpes simplex|HSV|oral herpes|genital herpes|herpetic infection)\b",
    "chickenpox": r"\b(?:chickenpox|varicella|varicella zoster infection|varicella infection)\b",

    # =========================
    # PAPULOSQUAMOUS
    # =========================
    "lichen planus": r"\b(?:lichen planus|LP|cutaneous lichen planus)\b",
    "pityriasis rosea": r"\b(?:pityriasis rosea|roseola annulata|herald patch)\b",

    # =========================
    # ROSACEA
    # =========================
    "rosacea": r"\b(?:rosacea|acne rosacea|adult acne rosacea|erythematotelangiectatic rosacea)\b",

    # =========================
    # KERATINIZATION
    # =========================
    "keratosis pilaris": r"\b(?:keratosis pilaris|KP|follicular keratosis|chicken skin)\b",
    "ichthyosis vulgaris": r"\b(?:ichthyosis vulgaris|ichthyosis|fish scale skin|fish skin disease)\b",

    # =========================
    # CYSTS / BENIGN LESIONS
    # =========================
    "pilar cyst": r"\b(?:pilar cyst|trichilemmal cyst|wen|scalp cyst)\b",
    "epidermoid cyst": r"\b(?:epidermoid cyst|epidermal inclusion cyst|sebaceous cyst|epidermal cyst)\b",
    "lipoma": r"\b(?:lipoma|fatty tumor|benign fatty tumor)\b",
    "keloid": r"\b(?:keloid|keloidal scar|keloid scar)\b",

    # =========================
    # PRE-CANCEROUS / CANCER
    # =========================
    "basal cell carcinoma": r"\b(?:basal cell carcinoma|BCC|basal cell cancer|basal cell skin cancer)\b",
    "squamous cell carcinoma": r"\b(?:squamous cell carcinoma|SCC|cutaneous squamous cell carcinoma|squamous cell skin cancer)\b",
    "melanoma": r"\b(?:melanoma|malignant melanoma|cutaneous melanoma)\b",
    "actinic keratosis": r"\b(?:actinic keratosis|solar keratosis|senile keratosis|AK)\b",

    # =========================
    # INFLAMMATORY / VASCULAR
    # =========================
    "erythema nodosum": r"\b(?:erythema nodosum|nodular panniculitis)\b",
    "erythema ab igne": r"\b(?:erythema ab igne|toasted skin syndrome|heat-induced reticulate erythema)\b",
    "vasculitis": r"\b(?:vasculitis|cutaneous vasculitis|leukocytoclastic vasculitis|cutaneous small vessel vasculitis)\b",

    # =========================
    # COMMON OTHER CONDITIONS
    # =========================
    "miliaria": r"\b(?:miliaria|prickly heat|heat rash|sweat rash|miliaria rubra)\b",
    "pruritus": r"\b(?:pruritus|itching|itch|skin itching)\b",
    "nevus": r"\b(?:nevus|naevus|mole|melanocytic nevus|benign mole)\b",
    "skin lesion": r"\b(?:skin lesion|cutaneous lesion|lesion|skin growth|cutaneous abnormality)\b",
    "rash": r"\b(?:rash|skin rash|cutaneous rash|eruption|skin eruption)\b",
    "erythema": r"\b(?:erythema|redness|skin redness|erythematous lesion)\b"
}