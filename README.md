# 🏠 House Price Prediction – King County

Predicting house prices using **supervised regression models** on the King County dataset (Kaggle).

## 📘 Dataset

**Source:** [King County House Prices Dataset – Kaggle](https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa)

This dataset includes homes sold in King County (Seattle area) with detail features such as:
- Bedrooms, bathrooms, living area 
- Floors, condition, grade, and location coordinates (latitude, longitude)
- Year built and renovated
- Sale price (`price`) — the **target variable**

## 🚀 Project Highlights

- Built a **full ML workflow**: data cleaning → feature engineering → model training → evaluation → hyperparameter tuning.
- Compared **multiple regression models**:
  - Linear Regression, KNN, Decision Tree, AdaBoost, Gradient Boosting, XGBoost, Random Forest
- Applied advanced techniques:
  - Price capping & log-transformation
  - Feature scaling (Standardization)
  - Hyperparameter tuning with RandomizedSearchCV
- **Best-performing models**: Random Forest vs XGBoost

---

## 🧭 Project Workflow Overview

### 🔹 **Part I – Data Preprocessing**
This notebook focuses on preparing and understanding the dataset before modeling.

1. **Data Cleaning & Exploration**
   - Loaded and explored the dataset.
   - Checked for duplicates, missing values, and data inconsistencies.
   - Visualized distributions and correlations to understand feature relationships.

2. **Feature Selection & Engineering**
   - Identified important predictors using correlation analysis and feature importance.
   - Created new engineered features and dropped irrelevant ones.
   - Prepared clean and processed data for modeling.

---

### 🔹 **Part II – Modelling and Evaluation **
This notebook builds, evaluates, and improves various regression models.

#### **1. Baseline Model Training and Evaluation**
Trained and compared several supervised regression models:
- Linear Regression  
- K-Nearest Neighbors (KNN) Regressor  
- Decision Tree Regressor  
- AdaBoost Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- Random Forest Regressor  

**Evaluation Metrics:**
- Mean Squared Error (MSE)  
- R-squared (R²)

#### **2. Feature Engineering and Model Enhancement**
To improve performance and handle skewed data:
1. **Feature Engineering – Target Variable (`price`)**
   - Applied **price capping over Q3** to handle extreme outliers.
   - Applied **log-transformation** on price for normalization.
2. **Feature Scaling**
   - Used **StandardScaler** for standardization of numeric features.
3. **Hyperparameter Tuning**
   - Applied **RandomizedSearchCV** to optimize model parameters for ensemble methods.

#### **3. Best Model Comparison**
Compared the two top-performing ensemble models — **Random Forest** and **XGBoost** — using:
- R² and MSE (train vs. test)
- Scatter plots of Actual vs. Predicted values
- Overfitting analysis (train–test performance gap)
- Feature importance visualization

---


## 📊 Key Results

| Model | Train R² | Test R² | Train MSE | Test MSE |
|-------|----------|---------|-----------|----------|
| Random Forest | 0.97 | 0.88 | 0.01 | 0.03 |
| XGBoost | 0.90 | 0.88 | 0.03 | 0.03 |

✅ XGBoost generalizes better  
⚠️ Random Forest slightly overfits

---

## 🔹 Tools & Skills Demonstrated

- Python, Jupyter Notebook  
- pandas, numpy, scikit-learn, xgboost  
- Data preprocessing, feature engineering, regression modeling  
- Hyperparameter tuning, model evaluation, ensemble methods  
- Visualization: matplotlib, seaborn

---

## 📈 Visualizations

- Actual vs Predicted prices  
- Model performance comparison (R², MSE)  
- Overfitting analysis  
- Feature importance

---

## 📂 Project Files

- **Part I:** Data cleaning & feature engineering → `Part_I_Data_Preprocessing.ipynb`  
- **Part II:** Modeling & evaluation → `Part_II_Modeling_and_Evaluation.ipynb`  
- **Presentation:** Key insights → `presentation.pdf`  

---

## 👨‍💻 Author

**Hyejeong Hayley Lee**  
📧 hyejeong0617@gmail.com
Github: https://github.com/hyejeong0617


