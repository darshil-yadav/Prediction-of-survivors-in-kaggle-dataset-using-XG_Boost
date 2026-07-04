# Titanic Survival Prediction using XGBoost 🚢

## Overview
This repository contains a complete machine learning pipeline designed to predict passenger survival on the Titanic. The project demonstrates an end-to-end data science workflow, from exploratory data analysis and complex feature engineering to model training and submission generation. The core predictive engine utilizes the eXtreme Gradient Boosting (XGBoost) algorithm.

## Kaggle Submission Result

![Kaggle Submission Score](https://github.com/darshil-yadav/Prediction-of-survivors-in-kaggle-dataset-using-XG_Boost/blob/main/kaggle%20submition.png?raw=true)

## Dataset
The data used in this project is from the famous [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) competition. It is split into `train.csv` (used for model training and validation) and `test.csv` (used for final Kaggle predictions).

## Machine Learning Pipeline

### 1. Data Cleaning & Imputation
Missing values were handled using targeted statistical imputation to preserve data integrity[cite: 4]:
* **Age:** Imputed missing values using the median age of the passenger's respective Ticket Class (`Pclass`)[cite: 4].
* **Fare:** Imputed missing values using the median fare of passengers sharing the same `Pclass` (3) and `Embarked` port ('S')[cite: 4].
* **Embarked:** Filled missing values with 'C' based on fare distribution similarities[cite: 4].
* **Cabin:** Extracted the deck letter, converting 'T' to 'M', and filled missing values with 'M' (Missing)[cite: 4].

### 2. Feature Engineering
Several new features were derived to capture social and economic survival dynamics[cite: 4]:
* **`Num_family`**: Created by combining `SibSp` (siblings/spouses) and `Parch` (parents/children) to calculate total family size[cite: 4].
* **`Age_Bins`**: Grouped continuous age data into 10-year brackets[cite: 4].
* **`adj_fare`**: Calculated the true cost per person by dividing the ticket `Fare` by the number of passengers sharing the same `Ticket` number (`tct_count`)[cite: 4].
* **`Fare_bin`**: Grouped the adjusted fares into logical brackets[cite: 4].

*(Note: Original continuous columns like `Age`, `Fare`, and `Ticket` were dropped after engineering to prevent multicollinearity and reduce noise[cite: 4].)*

### 3. Data Transformation & Scaling
* **Encoding:** Applied `LabelEncoder` to transform text-based categorical columns, followed by One-Hot Encoding (`pd.get_dummies`) for `Sex` and `Embarked`[cite: 4].
* **Scaling:** Normalized the final feature matrix using Scikit-Learn's `MinMaxScaler` to ensure all features contributed proportionately to the model[cite: 4].

### 4. Modeling & Evaluation
* Trained an out-of-the-box `XGBClassifier`[cite: 4].
* Implemented an 80/20 Train-Test split on the training data to monitor for overfitting[cite: 4].
* **Local Validation Metrics:**
  * **Train Accuracy:** ~89.18%[cite: 4]
  * **Test Accuracy:** ~81.56%[cite: 4]
* Applied the identical preprocessing pipeline to the unseen Kaggle test set to generate the final `submission.csv`[cite: 4].

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-Learn, XGBoost

## How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed (`pip install pandas numpy matplotlib scikit-learn xgboost`).
3. Run the Jupyter Notebook `data_pre_3.ipynb` from top to bottom.
4. The script will automatically output a `submission.csv` file ready for Kaggle.
