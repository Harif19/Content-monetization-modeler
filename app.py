# 🎬 Content Monetization Modeler — Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

# ------------------------------
# 🎯 Streamlit App Configuration
# ------------------------------
st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 YouTube Content Monetization Modeler")
st.markdown("""
This interactive app helps predict **YouTube Ad Revenue** using machine learning models.
You can train models, explore insights, and make predictions all in one place.
---
""")

# ------------------------------
# 📁 Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_youtube_ad_revenue.csv")

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill only numeric columns with median
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Convert possible date columns
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

df = load_data()

# Sidebar Navigation
menu = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Home", "📊 Train Models", "💡 Feature Insights", "📈 Predict Revenue"]
)

# --------------------------------------------------
# 🏠 HOME PAGE
# --------------------------------------------------
if menu == "🏠 Home":
    st.header("Welcome to the Content Monetization Modeler App 👋")
    st.markdown("""
    ### What this app does:
    - Analyze YouTube video metrics and their impact on ad revenue.
    - Train 5 regression models to predict earnings.
    - Identify key factors driving ad performance.
    - Predict revenue for new video metrics.

    ---
    **Available Models:**
    - Linear Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Lasso Regression
    - (Optional) XGBoost
    ---
    """)

# --------------------------------------------------
# 📊 TRAINING MODULES
# --------------------------------------------------
elif menu == "📊 Train Models":
    st.header("📊 Train & Compare Models")

    # Split data
    X = df.drop(columns=["video_id", "date", "ad_revenue_usd"], errors="ignore")
    y = df["ad_revenue_usd"]

    # Clean and split
    X = X.select_dtypes(include=[np.number])  # Ensure numeric-only
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale data for Linear/Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=50, max_depth=12, min_samples_split=10,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Lasso Regression": Lasso(alpha=0.01, random_state=42)
    }

    if xgb_available:
        models["XGBoost"] = XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=6,
            random_state=42, n_jobs=-1
        )

    results = []
    progress_bar = st.progress(0)

    for i, (name, model) in enumerate(models.items()):
        progress_bar.progress((i + 1) / len(models))
        st.write(f"🚀 Training {name}...")

        if name in ["Linear Regression", "Lasso Regression"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        preds = np.nan_to_num(preds, nan=np.nanmedian(y_test))

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results.append({"Model": name, "R²": r2, "MAE": mae, "RMSE": rmse})

    results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False)
    st.subheader("📊 Model Performance Results")
    st.dataframe(results_df)

    # Chart
    st.subheader("📈 Model R² Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="R²", y="Model", data=results_df, palette="viridis", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# 💡 FEATURE INSIGHTS
# --------------------------------------------------
elif menu == "💡 Feature Insights":
    st.header("💡 Feature Importance Visualization")

    X = df.drop(columns=["video_id", "date", "ad_revenue_usd"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["ad_revenue_usd"]

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top10 = importance.head(10)

    st.write("### Top 10 Influential Features:")
    st.bar_chart(top10)

    st.markdown("**Insight:** Features like `views`, `engagement_rate`, and `likes` strongly impact revenue.")

# --------------------------------------------------
# 📈 PREDICTION MODULE
# --------------------------------------------------
elif menu == "📈 Predict Revenue":
    st.header("🎯 Predict YouTube Ad Revenue")
    st.write("Enter your video metrics below to predict ad revenue:")

    # --- User Inputs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        views = st.number_input("Views", 0)
        watch_time = st.number_input("Watch Time (minutes)", 0)
    with col2:
        likes = st.number_input("Likes", 0)
        video_length = st.number_input("Video Length (minutes)", 0)
    with col3:
        comments = st.number_input("Comments", 0)
        subscribers = st.number_input("Channel Subscribers", 0)

    engagement_rate = (likes + comments) / views if views > 0 else 0

    # --- Train the model ---
    X = df.drop(columns=["video_id", "date", "ad_revenue_usd"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["ad_revenue_usd"]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # --- Prediction Button ---
st.markdown("<br>", unsafe_allow_html=True)
center_button = st.columns([2, 1, 2])
with center_button[1]:
    predict_clicked = st.button("🚀 Predict Revenue", use_container_width=True)

if predict_clicked:
    # --- Prepare Input ---
    input_data = pd.DataFrame([{
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time": watch_time,
        "video_length": video_length,
        "subscribers": subscribers,
        "engagement_rate": engagement_rate
    }])

    # Match training columns
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]

    try:
        pred = model.predict(input_data)[0]
        # --- Fancy Display ---
        st.markdown("""
            <style>
            .big-font {
                font-size:45px !important;
                color:#00FFAA;
                text-align:center;
            }
            .revenue-box {
                background-color:#1E1E1E;
                padding:20px;
                border-radius:15px;
                border:2px solid #00FFAA;
                box-shadow:0px 0px 15px #00FFAA33;
                margin-top: 20px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="revenue-box"><p class="big-font">💰 Estimated Revenue</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">${pred:,.2f} USD</p></div>', unsafe_allow_html=True)

        # --- CPM Visualization ---
        cpm = (pred / views * 1000) if views > 0 else 0
        st.markdown("### 🎯 Estimated CPM (Cost Per 1,000 Views)")
        st.progress(min(cpm / 20, 1.0))  # visualize CPM range up to $20
        st.write(f"**Approx CPM:** ${cpm:,.2f}")

        st.caption("Typical YouTube CPM: $1 – $20 depending on niche, geography, and audience.")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

