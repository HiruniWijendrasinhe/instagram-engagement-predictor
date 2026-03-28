# Instagram Engagement Predictor

FastAPI web app for predicting Instagram engagement from post-level features using two trained models:

- XGBoost
- Random Forest

The UI lets you submit one feature set and run prediction with either model using separate buttons.

## Current Features

- Dual-model inference from the same form input
- FastAPI endpoints for model-specific prediction
- Dropdowns for categorical features (`media_type`, `traffic_source`, `content_category`)
- Raw feature input flow (no derived feature calculations in the app)
- Internal model compatibility fields: `manual_sum=0` and `diff=0`

## Training Approach

When training on the dataset, use a validation-first workflow:

- Use cross-validation ( K-Fold) to estimate model stability.
- Use `GridSearchCV` to tune hyperparameters for both Random Forest and XGBoost.
- Save the best estimator from grid search as the final `.pkl` model artifact used by this app.

## Model Files

The backend loads these files:

- XGBoost: `final_model_xgboost_all_features.pkl`
- Random Forest: tries `final_model__all_features.pkl`, falls back to `final_model_instagram_all_features.pkl`

## Input Schema

Request body for prediction endpoints:

```json
{
  "media_type": "Carousel",
  "likes": 12000,
  "comments": 2000,
  "shares": 1000,
  "saves": 3000,
  "reach": 200000,
  "impressions": 250000,
  "caption_length": 120,
  "hashtags_count": 15,
  "followers_gained": 50,
  "traffic_source": "Explore",
  "content_category": "Fitness",
  "share_rate": 0.4,
  "save_rate": 1.2,
  "engagement_score": 56000
}
```

### Categorical Options

- `media_type`: `Carousel`, `Video`, `Reel`, `Photo`
- `traffic_source`: `Home Feed`, `Hashtags`, `Reels Feed`, `External`, `Profile`, `Explore`
- `content_category`: `Photography`, `Fashion`, `Technology`, `Lifestyle`, `Food`, `Fitness`, `Music`, `Travel`, `Beauty`, `Comedy`

## API Endpoints

- `POST /predict`
  - Default model: `xgboost`
- `POST /predict/xgboost`
- `POST /predict/random_forest`

Example success response:

```json
{
  "model": "xgboost",
  "predicted_engagement": 10.2461
}
```

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
uvicorn app:app --reload
```

3. Open:

- App UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Tech Stack

- Python
- FastAPI
- pandas
- scikit-learn
- XGBoost model artifact (loaded via joblib)
