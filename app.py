from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# ===============================
# Load trained ML models
# ===============================
BASE_DIR = Path(__file__).resolve().parent


def load_first_existing_model(candidates: list[str], model_label: str):
    """Load the first existing model file from candidate paths."""
    for candidate in candidates:
        candidate_path = BASE_DIR / candidate
        if candidate_path.exists():
            print(f"Loaded {model_label} model from: {candidate}")
            return joblib.load(candidate_path)
    raise FileNotFoundError(
        f"No {model_label} model found. Tried: {', '.join(candidates)}"
    )


models = {
    "xgboost": load_first_existing_model(
        [
            "final_model_xgboost_all_features.pkl",
            "model/final_model_xgboost_all_features.pkl",
            "models/final_model_xgboost_all_features.pkl",
        ],
        "xgboost",
    ),
    "random_forest": load_first_existing_model(
        [
            "final_model__all_features.pkl",
            "final_model_instagram_all_features.pkl",
            "model/final_model_instagram_all_features.pkl",
            "models/final_model_instagram_all_features.pkl",
        ],
        "random_forest",
    ),
}

FEATURE_COLUMNS = [
    "media_type",
    "likes",
    "comments",
    "shares",
    "saves",
    "reach",
    "impressions",
    "caption_length",
    "hashtags_count",
    "followers_gained",
    "traffic_source",
    "content_category",
    "share_rate",
    "save_rate",
    "engagement_score",
    "manual_sum",
    "diff",
]

# ===============================
# Create FastAPI app
# ===============================
app = FastAPI(title="Instagram Engagement Predictor")

# ===============================
# Mount frontend
# ===============================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")


def build_model_input(d: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [[
            d["media_type"],
            d["likes"],
            d["comments"],
            d["shares"],
            d["saves"],
            d["reach"],
            d["impressions"],
            d["caption_length"],
            d["hashtags_count"],
            d["followers_gained"],
            d["traffic_source"],
            d["content_category"],
            d["share_rate"],
            d["save_rate"],
            d["engagement_score"],
            0,
            0,
        ]],
        columns=FEATURE_COLUMNS,
    )


# ===============================
# Input schema
# ===============================
class InstaInput(BaseModel):
    media_type: str
    likes: float
    comments: float
    shares: float
    saves: float
    reach: float
    impressions: float
    caption_length: float
    hashtags_count: float
    followers_gained: float
    traffic_source: str
    content_category: str
    share_rate: float
    save_rate: float
    engagement_score: float


# ===============================
# Prediction API
# ===============================
@app.post("/predict")
def predict_engagement(data: InstaInput):
    # Default endpoint keeps backward compatibility and uses xgboost.
    return predict_engagement_by_model("xgboost", data)


@app.post("/predict/{model_name}")
def predict_engagement_by_model(model_name: str, data: InstaInput):
    try:
        d = data.dict()
        normalized_model_name = model_name.strip().lower().replace("-", "_")

        if normalized_model_name not in models:
            return {
                "error": (
                    "Unsupported model. Use one of: "
                    + ", ".join(models.keys())
                )
            }

        input_data = build_model_input(d)

        # -------------------------------
        # Debug print
        # -------------------------------
        print(f"Features sent to {normalized_model_name} model:")
        print(input_data)

        # -------------------------------
        # Model prediction
        # -------------------------------
        prediction = float(models[normalized_model_name].predict(input_data)[0])

        print("Raw model prediction:", prediction)

        return {
            "model": normalized_model_name,
            "predicted_engagement": float(round(prediction, 4))
        }

    except Exception as e:

        import traceback

        print("Prediction error:")
        print(traceback.format_exc())

        return {
            "error": f"Prediction failed: {str(e)}"
        }