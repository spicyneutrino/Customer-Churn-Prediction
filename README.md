# Customer Churn Prediction

This repository contains the implementation of a **customer churn prediction classification project** using various machine learning techniques. The goal of this project is to predict whether a customer will churn (leave the company) based on a set of customer features. The dataset used is imbalanced, and we apply **Synthetic Minority Over-sampling Technique (SMOTE)** to handle the class imbalance. The project uses several models, such as Logistic Regression, SVC, and Random Forest, to achieve the best predictive performance.

The dataset used in this project is from Kaggle: [Telco Customer Churn IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).

## Table of Contents

- [Customer Churn Prediction](#customer-churn-prediction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Preprocessing](#preprocessing)
  - [Models](#models)
  - [Model Evaluation](#model-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Running the Project](#running-the-project)
  - [Requirements](#requirements)
  - [Contributing](#contributing)

## Introduction

Customer churn prediction is crucial for businesses to anticipate which customers are likely to leave and take action to retain them. This project uses a publicly available dataset of customer information, including demographic and service-related features, to predict churn.

The model is trained and evaluated using classification techniques, with a focus on addressing data imbalance, improving performance, and optimizing hyperparameters for the best results.

## Data

The dataset used in this project contains information on customers from a telecommunications company, including the following features:

- **CustomerID**: Unique identifier for each customer.
- **Country, State, City**: Geographical information.
- **Gender**: Gender of the customer.
- **Tenure Months**: The duration the customer has been with the company.
- **Monthly Charge**: Monthly charge for the customer.
- **Total Charges**: Total amount paid by the customer.
- **Churn Value**: The target variable indicating whether the customer churned (1) or not (0).

The dataset is preprocessed to handle missing values, encode categorical variables, and apply feature scaling.

## Preprocessing

1. **Handling Missing Values**: The dataset is checked for missing values and appropriate imputation methods are applied.
2. **Encoding Categorical Features**: Categorical features like gender and city are one-hot encoded, and ordinal variables are appropriately encoded.
3. **Feature Engineering**: New features are created from existing ones, such as "Security Focused" and "Streaming."
4. **Class Imbalance**: Since there is a severe class imbalance (more non-churning customers than churned customers), SMOTE is used to generate synthetic data for the minority class.

## Models

The following models are tested and evaluated for churn prediction:

- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Random Forest Classifier**

These models are selected based on their performance and suitability for the problem. Random Forest was found to be the most effective model after applying SMOTE for handling class imbalance.

## Model Evaluation

Model performance is evaluated using the following metrics:

- **Classification Report**: Precision, recall, and F1-score for both classes (churn and no churn).
- **ROC-AUC**: The area under the Receiver Operating Characteristic curve.
- **Precision-Recall AUC**: For imbalanced classification.
- **F1-Score**: The harmonic mean of precision and recall.

## Hyperparameter Tuning

GridSearchCV is used to find the best hyperparameters for each model. The following hyperparameters are tuned:

- **Logistic Regression**: `C`, `solver`, and `max_iter`.
- **SVC**: `C`, `kernel`, `gamma`, and `degree`.
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `bootstrap`.

## Running the Project

To run the project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/churn-prediction.git
   cd churn-prediction

2. Install the necessary dependencies:

    ```bash
   pip install -r requirements.txt

3. Run the preprocessing.ipynb and then model_training.ipynb to train the models and evaluate their performance

## Requirements

This project requires the following Python packages:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
  
You can install these dependencies using the following command:
    ```bash
    pip install -r requirements.txt

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and create a pull request. Please make sure to follow the coding standards and include tests for new features.
