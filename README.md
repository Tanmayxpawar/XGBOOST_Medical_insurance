# 🏥 Medical Insurance Charges Prediction

## 📖 Project Overview
This project aims to predict medical insurance charges using a machine learning model based on XGBoost. The dataset consists of various features related to patients' medical history, demographic information, and insurance details. The goal is to build an accurate model to estimate the insurance charges incurred by patients.

## 🌟 Key Features
- **🔍 Predictive Modeling:** Utilizes XGBoost, an efficient and powerful gradient boosting framework, for regression tasks.
- **🧹 Data Preprocessing:** Implements thorough data cleaning and preprocessing techniques to prepare the dataset for modeling.
- **📊 Model Evaluation:** Includes metrics such as R² score, Mean Absolute Error (MAE), and Mean Squared Error (MSE) to assess model performance.
- **📈 Visualization:** Features visualizations for exploratory data analysis (EDA) to understand data distributions and relationships between features.

## 📊 Model Performance
- **📈 Training R² Score:** 0.8560
- **📉 Test R² Score:** 0.8548
- **🔄 Cross-Validation Score:** 0.8546

## ⚙️ Technologies Used
- **🐍 Python:** The primary programming language used for data analysis and modeling.
- **🌲 XGBoost:** The machine learning algorithm employed for regression.
- **📊 Pandas:** For data manipulation and analysis.
- **🔢 NumPy:** For numerical operations.
- **📉 Matplotlib/Seaborn:** For data visualization.
- **🔧 Scikit-learn:** For model evaluation and cross-validation.

## 📂 Dataset
The dataset includes the following columns:
- **👤 Age:** Age of the insured individual.
- **🚻 Sex:** Gender of the insured (male/female).
- **⚖️ BMI:** Body mass index of the insured individual.
- **👶 Children:** Number of children/dependents covered by the insurance plan.
- **🚬 Smoker:** Whether the insured individual is a smoker (yes/no).
- **🌍 Region:** Region of residence (e.g., southeast, southwest).
- **💰 Charges:** The medical insurance charges (target variable).

## 📥 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Tanmayxpawar/Medical-Insurance-Charges-Prediction.git
   cd Medical-Insurance-Charges-Prediction
