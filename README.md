# Predicting Student Employability Using Machine Learning

This project is focused on predicting the employability of students based on several features such as educational background, academic performance, and participation in extracurricular activities. By utilizing different machine learning algorithms, the project compares models to find the most accurate prediction for student employability.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview

The project applies various machine learning algorithms (Logistic Regression, K-Nearest Neighbors, Random Forest, Decision Tree, and SVM) to predict whether a student is employable based on their academic data. The accuracy of each model is tested and compared to determine the best-performing model.

## Features

- Preprocessing of raw student data (handling missing values, encoding categorical features)
- Training multiple machine learning models
- Predicting employability based on test data
- Generating evaluation metrics such as accuracy, confusion matrix, and classification report

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - NumPy
  - Pandas
  - Scikit-learn
  - Seaborn
  - Matplotlib

## System Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- Development Environment: Google Colaboratory or Jupyter Notebook

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/predicting-student-employability.git
    cd predicting-student-employability
    ```

2. **Install required libraries**:
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn
    ```

## Datasets

The dataset contains features like Degree, Gender, Skills, CGPA, Clubs participation, and whether the student is employable (`Output`). 

You can upload the dataset via Google Colaboratory using:
```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('database1.csv')
  ```

## Configuration

1. **Data Preprocessing**:
   - Replace categorical values (e.g., 'Btech', 'Mtech', 'B.E') with numerical values.
   - Handle missing values by filling with median values or removing unnecessary columns.

2. **Train-Test Split: Spliting the dataset into training and testing data**

## Running the Application
1. **Train the Model**: 
Choose a machine learning model (e.g., Logistic Regression, KNN, Random Forest):
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
  ```

2. **Make Predictions**:
Use the trained model to make predictions on the test dataset:
```python
predictions = model.predict(X_test)
  ```
 
3. **Evaluate the Model**:
Generate accuracy, confusion matrix, and classification report to evaluate the model.

## Evaluation Metrics

The model is evaluated using the following metrics:

- Accuracy: Ratio of correctly predicted observations.
- Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives.
- Precision: Ratio of true positives to the sum of true positives and false positives.
- F1-Score: Harmonic mean of precision and recall.

## Results

Logistic Regression and Random Forest provided high accuracy for larger datasets.
K-Nearest Neighbors performed well with smaller datasets.
Detailed classification reports and confusion matrices help in understanding model performance.

## Conclusion

This project highlights how machine learning can be leveraged to predict student employability. Among the tested models, Logistic Regression showed the most consistent performance across different datasets, making it a reliable option for employability prediction.
   
