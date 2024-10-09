# wiremind
This project explores key features in air cargo shipments to predict the price per kilogram. We conducted an in-depth analysis to identify influential factors, trained a regression model, and evaluated its performance, aiming to enhance future understanding and optimize pricing predictions.


# Revenue Prediction Project

## Table of Contents
1. [Exploratory Data Analysis (EDA)](#eda)
   1.1 [Overview](#overview)  
   1.2 [Months Analysis](#months-analysis)  
   1.3 [Cargo Type Analysis](#cargo-type-analysis)  
   1.4 [Product Type Analysis](#product-type-analysis)  
   1.5 [Agents Shipping Analysis](#agents-shipping-analysis)  
2. [Feature Engineering](#feature-engineering)  
3. [Preparing the Data for Our Model](#preparing-the-data-for-our-model)  
4. [Model and Metrics Selection](#model-and-metrics-selection)  
5. [Evaluating Our Models](#evaluating-our-models)  
   5.1 [Train-Test Split Method](#train-test-split-method)  
   5.2 [Cross Validation](#cross-validation)  
6. [Results Interpretation](#results-interpretation)  

## Overview

This project aims to predict revenue based on chargeable weight, utilizing various machine learning models. Through thorough exploratory data analysis (EDA) and feature engineering, we enhance our dataset to improve model performance.

## EDA
### Overview
In this section, we conduct a comprehensive analysis of the dataset to understand its structure and identify key patterns.

### Months Analysis
We analyze the data based on different months to observe seasonal trends in revenue.

### Cargo Type Analysis
This analysis focuses on the types of cargo and their impact on revenue.

### Product Type Analysis
We explore how different product types contribute to the overall revenue.

### Agents Shipping Analysis
This section investigates the performance of various agents in shipping products.

## Feature Engineering
Here, we create new features, such as `Agent_Avg_Revenue`, to enhance the predictive capability of our models.

## Preparing the Data for Our Model
We preprocess the data, removing outlier (or not), and prepare it for model training.

## Model and Metrics Selection
We select appropriate machine learning models and metrics for evaluating their performance.

## Evaluating Our Models
### Train-Test Split Method
We evaluate model performance using the train-test split method.

### Cross Validation
This section outlines our approach to model evaluation using cross-validation techniques.

## Results Interpretation
We analyze the results obtained from our models, discussing their implications and potential areas for improvement.

## Requirements
To run this project, you need the following Python libraries:

```bash
pip install matplotlib seaborn numpy pandas scikit-learn


