# iris-flower-random-forest-problem

# Iris Flower Classification using Random Forest

This project builds a machine learning model to classify Iris flowers into their species (Setosa, Versicolor, and Virginica) using a Random Forest classifier. It demonstrates basic data handling, preprocessing, model training, and evaluation steps using Python and scikit-learn.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model and Pipeline](#model-and-pipeline)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers with measurements of sepal length, sepal width, petal length, and petal width. The goal of this project is to use these features to classify the flower into one of three species:
- **Setosa**
- **Versicolor**
- **Virginica**

In this project, we use a Random Forest algorithm, an ensemble method that builds multiple decision trees and combines their outputs to improve classification accuracy.

## Dataset

The dataset is provided in the `iris.csv` file and contains the following columns:
- **sepal_length**
- **sepal_width**
- **petal_length**
- **petal_width**
- **species** (0 for Setosa, 1 for Versicolor, 2 for Virginica)

> **Note:** The provided dataset is a complete version of the Iris dataset with all 150 samples.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/iris-flower-classification.git
   cd iris-flower-classification
