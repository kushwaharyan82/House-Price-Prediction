# 🏠 House Price Prediction

A Machine Learning project that predicts house values using the California Housing Dataset and a Random Forest Regression model.

## 📌 Project Overview

This project uses Machine Learning to estimate house prices based on different property and location features.

The model is trained using the California Housing Dataset and deployed as an interactive Streamlit web application.

## 🤖 Machine Learning Model

- Algorithm: Random Forest Regressor
- Framework: Scikit-learn
- Problem Type: Regression

## 📊 Dataset

California Housing Dataset

### Features

- MedInc — Median Income
- HouseAge — House Age
- AveRooms — Average Rooms
- AveBedrms — Average Bedrooms
- Population — Population
- AveOccup — Average Occupancy
- Latitude — Latitude
- Longitude — Longitude

### Target

- MedHouseVal — Median House Value

## 📈 Model Performance

| Metric | Score |
|---|---:|
| R² Score | 0.805 |
| MAE | 0.328 |
| RMSE | 0.505 |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Deployment