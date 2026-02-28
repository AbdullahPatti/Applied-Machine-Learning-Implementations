# AG-News Classification – Multinomial Naive Bayes

A manual implementation of **Multinomial Naive Bayes** for classifying news articles into categories. This project illustrates text preprocessing, bag-of-words vectorization, and probability-based text classification.

---

## Dataset

**AG-News Dataset** – [Kaggle link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

- English news articles with 4 categories  
- Target: `ClassIndex`  
- Feature: `Description` (text)  
- Suitable for **Multinomial Naive Bayes** due to word count representation

---

## Objectives

- Implement Multinomial Naive Bayes manually using only **NumPy** and **pandas**  
- Build a **Bag-of-Words vectorizer** from scratch  
- Compute class priors and word likelihoods with Laplace smoothing in log space  
- Implement `predict()` and `predict_proba()`  
- Evaluate using Accuracy, Precision, Recall, F1-score, and Confusion Matrix  
- Compare with `sklearn` MultinomialNB  

---

## Preprocessing

- Text cleaning function `preprocess_text(text)`:
  - Remove URLs  
  - Remove punctuation and non-alphanumeric characters  
  - Convert to lowercase  
  - Remove extra whitespace  
  - Remove stopwords (manual list)  
- Apply preprocessing to the `Description` column  
- Print first 5 preprocessed samples for verification  

---

## Implementation Details

- **Class:** `MultinomialNaiveBayes`  
- **Methods:**  
  - `fit(X_train, y_train)` – calculate class priors and word likelihoods  
  - `predict(X_test)` – return class with highest probability  
  - `predict_proba(X_test)` – return probability distribution over classes  
- **Additional Features:**  
  - `self._ai_meta = {"author":"naivebayes","version":"0.1"}`  
  - Probabilities clipped to `[1e-10, ∞)` using `np.maximum`  
  - Placeholder variable `useless = None` as required  
  - Random seed: 42  

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Comparison with sklearn’s `MultinomialNB` is included.

---
