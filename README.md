# 📸 Instagram Engagement Predictor

Predict Instagram **60-day engagement rate** using a 🌲 **Random Forest Regressor** with automated preprocessing, feature engineering, and a ⚡ **FastAPI API** for real-time predictions.  

---

## 🛠 Features

- ✅ Handles missing values, duplicates, and extreme/outlier values  
- 📊 Creates normalized features like `eng_per_follower`  
- 🔄 Applies **log transformation** to reduce skewness in target  
- 🏗 Performs **encoding** for categorical variables (`channel_info`, `country`)  
- 🌲 Uses **Random Forest** with **GridSearchCV** for hyperparameter tuning  
- ⚡ Provides a **FastAPI endpoint** for real-time prediction  

---

## 🧩 Input Schema

```json
{
  "channel_info": "string",
  "influence_score": 0,
  "posts": 0,
  "followers": 0,
  "avg_likes": 0,
  "new_post_avg_like": 0,
  "total_likes": 0,
  "country": "string"
}
---
##🚀 Usage
### 1.Clone the repository:
git clone https://github.com/HiruniWijendrasinhe/instagram-engagement-predictor.git
cd instagram-engagement-predictor
###2.Install dependencies:
pip install -r requirements.txt
###3.Run the FastAPI app:
uvicorn main:app --reload
###4.Test the API:
http://127.0.0.1:8000/docs
---
## 📈 Model Performance

- **Best Hyperparameters:**  
  - `n_estimators = 200`  
  - `max_depth = None`  
  - `min_samples_split = 2`  
  - `min_samples_leaf = 1`

- **Best CV R²:** 0.931

- **Training Metrics:**  
  - R² = 0.992  
  - MAE = 0.033  
  - MSE = 0.004  
  - RMSE = 0.060

- **Test Metrics:**  
  - R² = 0.981  
  - MAE = 0.063  
  - MSE = 0.0073  
  - RMSE = 0.085

---

## 🔧 Technologies Used

- Python 🐍  
- scikit-learn 🌲  
- pandas 📊  
- NumPy 🔢  
- FastAPI ⚡  
- joblib 💾
---
