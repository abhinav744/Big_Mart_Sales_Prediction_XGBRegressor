# 🛒 Big Mart Sales Prediction – XGBRegressor

This project predicts retail sales for outlets in the Big Mart dataset using an advanced XGBoost regression model (XGBRegressor). The workflow includes data preprocessing, feature engineering, model training, and evaluation.

## 🚀 Live Demo

[Big Mart Sales Prediction App](https://big-mart-wvnjauexpzbue2mq3oth4q.streamlit.app/)

## 📌 Project Overview

Objective: Forecast sales for each outlet-item combination using structured data (e.g., item weight, item visibility, outlet type, etc.).

Model: XGBRegressor — known for performance and handling non-linear relationships effectively.

Includes:

Data loading & preprocessing (imputation, encoding)

Feature engineering (e.g., combining outlet and item features)

Model training with hyperparameter tuning

Evaluation using RMSE and R²

Deployed via a Streamlit web app

## 📂 Repository Structure

/Big_Mart_Sales_Prediction_XGBRegressor

│── Train.csv                # Training dataset

│── app.py                   # Streamlit app

│── requirements.txt         # Python dependencies

│── README.md                # This documentation

## ⚡ Getting Started

### 1. Clone the repository

git clone https://github.com/abhinav744/Big_Mart_Sales_Prediction_XGBRegressor.git

cd Big_Mart_Sales_Prediction_XGBRegressor

### 2. (Optional) Create & activate a virtual environment

python -m venv venv

source venv/bin/activate   # On Windows: venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the app locally

streamlit run app.py

## 📊 Insights & Typical Performance

RMSE typically varies depending on preprocessing and features.

R² score reflects how well the model captures variance in sales data.

Effective feature engineering (e.g., handling missing values, categorical encoding) significantly improves performance.

## 🔮 Future Enhancements

Hyperparameter tuning (e.g., using GridSearchCV or RandomizedSearchCV)

Compare with ensemble models (Random Forest, LightGBM)

Feature importance visualization (via XGBoost or SHAP)
