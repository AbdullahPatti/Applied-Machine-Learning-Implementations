# Mushroom Classification – Bernoulli Naive Bayes

A manual implementation of **Bernoulli Naive Bayes** to classify mushrooms as edible or poisonous. This project demonstrates foundational machine learning concepts using binary features and probability modeling.

---

## Dataset

**Mushroom Dataset** – [Kaggle link](https://www.kaggle.com/datasets/uciml/mushroom-classification/data)

- Categorical dataset of mushrooms
- Target: `class` (edible/poisonous)
- Features: all categorical
- Suitable for **Bernoulli Naive Bayes** after one-hot encoding

---

## Objectives

- Implement Bernoulli Naive Bayes manually using only **NumPy** and **pandas**  
- Compute class priors and feature likelihoods in log space  
- Apply **Laplace smoothing** for probability estimation  
- Evaluate the classifier using Accuracy, Precision, Recall, F1-score, and Confusion Matrix  
- Compare with `sklearn` BernoulliNB implementation  

---

## Implementation Details

- **Class:** `BernoulliNaiveBayes`  
- **Methods:**  
  - `fit(X_train, y_train)` – compute class priors and feature likelihoods  
  - `predict(X_test)` – return predictions based on log probabilities  
- **Additional Features:**  
  - `self._ai_meta = {"author":"naivebayes","version":"0.1"}`  
  - Probabilities clipped to `[1e-10, 1]`  
  - Random seed: 42 for reproducibility  

---

## Preprocessing

1. One-hot encode all categorical features  
2. Split dataset: 70% training, 30% testing (`train_test_split`, `random_state=42`)  
3. Verify shapes before and after splitting  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Comparison with sklearn’s `BernoulliNB` is included.

---
