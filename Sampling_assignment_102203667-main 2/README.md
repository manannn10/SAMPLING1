# Sampling_assignment_102203667


# Credit Card Fraud Detection - Model Evaluation with Sampling Techniques

This project evaluates the performance of various machine learning models (Logistic Regression, Decision Tree, and Random Forest) on a credit card fraud detection dataset. Different sampling techniques are applied to balance the dataset and evaluate how they impact model performance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Sampling Techniques](#sampling-techniques)
- [Models Used](#models-used)
- [Results](#results)
- [License](#license)

## Introduction

In this project, we apply various sampling techniques to balance the class distribution in the dataset. The techniques include random sampling, stratified sampling, systematic sampling, bootstrap sampling, and cluster sampling. The dataset used is a credit card fraud detection dataset. After balancing the dataset, we evaluate the performance of three machine learning models:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The goal is to analyze how different sampling methods affect model accuracy.

## Installation

To run this project, you need the following Python libraries:

1. `pandas`
2. `numpy`
3. `scikit-learn`
4. `imbalanced-learn`

You can install these dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
