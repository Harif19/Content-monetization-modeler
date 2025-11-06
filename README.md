# Content-monetization-modeler
A machine learning web app that predicts YouTube ad revenue using video analytics, built with Python, scikit-learn, and Streamlit.

 
### Predict YouTube Ad Revenue using Machine Learning  

---

##  Project Overview  
**Content Monetization Modeler** is a complete end-to-end machine learning project that predicts **YouTube Ad Revenue** based on video performance metrics such as views, likes, comments, watch time, and engagement rate.  

The goal of this project is to help **content creators and media strategists** estimate potential earnings, identify key factors influencing monetization, and make data-driven decisions for their video strategies.  

---

## 🚀 Key Features  
✅ Predict YouTube ad revenue using 6 regression models  
✅ Visualize and compare model performance (R², MAE, RMSE)  
✅ Identify key features influencing ad revenue (feature importance)  
✅ Interactive Streamlit web app for predictions and insights  
✅ Beautiful dark-themed UI with glowing revenue visuals & CPM tracker  

---

## 📊 Project Workflow  

### 1️⃣ Understand the Dataset  
- Loaded and explored `cleaned_youtube_ad_revenue.csv`  
- Identified numerical and categorical columns  
- Target Variable: `ad_revenue_usd`  

### 2️⃣ Exploratory Data Analysis (EDA)  
- Correlation heatmap and scatter plots to identify trends  
- Outlier detection in `views`, `likes`, and `ad_revenue_usd`  
- Found strong correlation between engagement and revenue  

### 3️⃣ Data Preprocessing  
- Removed duplicates and handled missing values  
- Replaced infinities with NaN and filled numeric columns using median  
- Converted `date` fields to datetime  

### 4️⃣ Feature Engineering  
- Created **engagement_rate** = (`likes` + `comments`) / `views`  
- Created **watch_rate** = `watch_time_minutes` / `video_length_minutes`  
- Scaled data for selected models  

### 5️⃣ Model Building  
Trained and compared 6 regression models:
| Model | Type | Notes |
|--------|-------|-------|
| Linear Regression | Baseline | Fast and interpretable |
| Decision Tree Regressor | Non-linear | Captures patterns easily |
| Random Forest Regressor | Ensemble | Balanced accuracy |
| Gradient Boosting Regressor | Ensemble | Robust with small datasets |
| Lasso Regression | Regularized | Handles feature selection |
| XGBoost | Gradient Boosting | High-performance ensemble |

---

### 6️⃣ Model Evaluation  
Evaluated using regression metrics:

| Model | R² | MAE | RMSE |
|--------|----|-----|------|
| Linear Regression | 0.9526 | 3.11 | 13.48 |
| Decision Tree | 0.9481 | 4.13 | 14.09 |
| Random Forest | 0.9517 | 3.55 | 13.59 |
| Gradient Boosting | 0.9522 | 3.61 | 13.52 |
| Lasso Regression | **0.9526** | **3.09** | **13.47** |
| XGBoost | 0.9522 | 3.51 | 13.53 |

✅ **Best Model:** Lasso Regression (High R², Lowest MAE and RMSE)

---

## 🎨 Streamlit App Overview  

**App Sections:**
1. 🏠 **Home** — Overview and usage guide  
2. 📊 **Train Models** — Train and compare model performance  
3. 💡 **Feature Insights** — Visualize top influential features  
4. 📈 **Predict Revenue** — Input video metrics and predict revenue interactively
5. Reference webpage demo images below

<img width="1910" height="865" alt="Streamlit " src="https://github.com/user-attachments/assets/8ebf719b-9f0d-4f8c-a556-7a5d6ee8c49f" />
<img width="1909" height="896" alt="Streamlit 2" src="https://github.com/user-attachments/assets/94b5a004-ce7d-4934-a104-f4e94d26ac76" />
<img width="1913" height="908" alt="Streamlit 3" src="https://github.com/user-attachments/assets/9c39ea3d-a142-467d-9cd7-ec16d192ccb3" />
<img width="1900" height="900" alt="Streamlit 4" src="https://github.com/user-attachments/assets/a741ea0b-3f6a-443a-9e70-f1678991df63" />
<img width="1850" height="897" alt="Streamlit 5" src="https://github.com/user-attachments/assets/a355edfb-3e76-4880-8afa-a3b2a5d6ea64" />
<img width="1914" height="884" alt="Streamlit 6" src="https://github.com/user-attachments/assets/07987151-106e-4ff4-b659-a5173c8cdb4b" />





**Predict Revenue Features:**
- Glowing animated revenue box 💰  
- CPM (Cost Per 1,000 Views) visualization bar  
- Real-time input sliders and number fields  

---

## 🧮 Technologies Used  
| Category | Tools |
|-----------|-------|
| Language | Python |
| Libraries | pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost |
| Web App | Streamlit |
| Deployment | Ngrok / Streamlit Cloud |
| Version Control | Git + GitHub |

---

## ⚡ How to Run Locally  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/YOUR_USERNAME/content-monetization-modeler.git
cd content-monetization-modeler
