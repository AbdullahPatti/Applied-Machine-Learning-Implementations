# Students Grade Classification (One-vs-All Logistic Regression)

## Project Overview
This project implements a **multi-class student grade classification system** using the **One-vs-All (OvA) strategy** with Logistic Regression. The goal is to predict students' final course grades based on **behavioral and demographic features**.

The project demonstrates both:  
1. **Manual implementation of Logistic Regression** from scratch (gradient descent, sigmoid activation, cross-entropy loss).  
2. **Scikit-learn implementation** for comparison and evaluation.

## Dataset
- File: `students_data.csv`  
- Samples: 500 students  
- Features: 14 behavioral and demographic columns  
- Target: `grade` (categorical)  

Dataset is split into **80% training** and **20% testing** sets.

## Features
- Feature normalization using `StandardScaler`  
- **Manual Logistic Regression**:  
  - Gradient descent optimization  
  - Binary Cross-Entropy loss  
  - One-vs-All (OvA) multi-class classification  
  - Training and validation loss curves  
- **Evaluation metrics**: Accuracy, Macro F1-Score, Confusion Matrix  
- **Scikit-learn comparison** with `LogisticRegression(multi_class='ovr')`  
- ROC-AUC curves for each class (OvA approach)  

## Results
### Multiclass Evaluation:
- Accuracy = 0.6200
- Macro F1-Score = 0.6234
- 
### Scikit-learn OvA
- Accuracy: 0.6400
- Macro F1-Score: 0.6584

