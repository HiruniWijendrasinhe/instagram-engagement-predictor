from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("instagram_engagement_model.pkl")

# Define FastAPI app
app = FastAPI(title="Instagram Engagement Predictor")

# Input schema

# Only require the minimal fields needed to compute all features
class InstaInput(BaseModel):
    channel_info: str
    influence_score: float
    posts: float
    followers: float
    avg_likes: float
    new_post_avg_like: float
    total_likes: float
    country: str
    rank: float = 0  # Optional, default to 0 if not provided

@app.post("/predict")
def predict_engagement(data: InstaInput):
    # Convert input to dict
    d = data.dict()

    # Compute derived features
    # Avoid division by zero
    followers = d["followers"] if d["followers"] != 0 else 1
    posts = d["posts"] if d["posts"] != 0 else 1
    total_likes = d["total_likes"]
    avg_likes = d["avg_likes"]
    new_post_avg_like = d["new_post_avg_like"]

    d["likes_per_follower"] = avg_likes / followers
    d["followers_per_post"] = followers / posts
    d["recent_like_ratio"] = new_post_avg_like / avg_likes if avg_likes != 0 else 0
    d["total_likes_per_follower"] = total_likes / followers

    # Ensure all required columns are present
    required_cols = [
        'rank', 'channel_info', 'influence_score', 'posts', 'followers', 'avg_likes',
        'new_post_avg_like', 'total_likes', 'country', 'likes_per_follower',
        'followers_per_post', 'recent_like_ratio', 'total_likes_per_follower'
    ]
    input_df = pd.DataFrame([{col: d.get(col, 0) for col in required_cols}])

    # Predict using the full pipeline
    prediction = model.predict(input_df)
    return {"predicted_60_day_eng_rate": float(prediction[0])}