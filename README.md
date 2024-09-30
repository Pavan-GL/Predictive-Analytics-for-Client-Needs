Student Performance Prediction

Overview

This project aims to predict student performance using various machine learning models. It demonstrates a complete machine learning workflow, from understanding the problem statement to deploying a trained model using Flask.

Table of Contents

    Problem Statement
    Data Collection
    Data Checks
    Exploratory Data Analysis (EDA)
    Data Pre-Processing
    Model Training
    Choosing the Best Model
    Deployment Steps
    Installation
    Usage

Problem Statement

The goal is to predict student performance based on various factors such as attendance, previous grades, and socioeconomic status. This information can help educators identify students who may need additional support.

Data Collection

Data is collected from various sources, including educational institutions and public datasets. The dataset contains features relevant to student performance.


Before processing the data, the following checks are performed:

    Missing Values: Identify and handle missing data.
    Data Types: Ensure that all features have the correct data types.
    Outliers: Detect and analyze outliers that may affect model performance.
    Exploratory Data Analysis (EDA)
    EDA is conducted to uncover insights about the data, which may include:

    Distribution of scores
    Correlation between features
    Visualization of data patterns using graphs and charts
    Data Pre-Processing
Data pre-processing steps include:

    Handling missing values
    Encoding categorical variables
    Normalizing/Standardizing numerical features
    Splitting the dataset into training and testing sets
    Model Training

Multiple models are trained to find the best-performing one. The following models are implemented:

    Random Forest Regressor
    Decision Tree Regressor
    Gradient Boosting Regressor
    Linear Regression
    XGBRegressor
    CatBoosting Regressor
    AdaBoost Regressor
    Choosing the Best Model

After training, models are evaluated based on metrics such as R² score and Mean Absolute Error (MAE). The best model is selected and serialized into a .pkl file for future use.

Deployment Steps

The trained model is deployed using Flask, allowing users to make predictions via a web interface. The basic steps include:

Set up Flask: Create a Flask application.

Load the Model: Load the saved model from the .pkl file.

Create API Endpoints: Define endpoints to accept input data and return predictions.

Run the Application: Start the Flask server.

Installation
To set up the project locally, follow these steps:

Clone the repository:

    git clone https://github.com/yourusername/student-performance-prediction.git
    cd student-performance-prediction

Install the required dependencies:
    pip install -r requirements.txt

Start the Flask application:

    python app.py
