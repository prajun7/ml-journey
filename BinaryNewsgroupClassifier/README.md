# 🧠 20 Newsgroups Binary Text Classification

This project implements a **binary text classifier** using a modified version of the **20 Newsgroups** dataset.  
The goal is to classify text documents into one of two broad categories — **Technology & Science** or **Sports, Politics, & Miscellaneous** — using natural language processing and machine learning techniques.

---

## 📂 Dataset Information

The original 20 Newsgroups dataset contains around **18,846 documents** across **20 categories**.  
For this assignment, the categories are grouped as follows:

### **Class 1 – Technology & Science**

- `comp.graphics`
- `comp.os.ms-windows.misc`
- `comp.sys.ibm.pc.hardware`
- `comp.sys.mac.hardware`
- `comp.windows.x`
- `sci.crypt`
- `sci.electronics`
- `sci.med`
- `sci.space`

### **Class 2 – Sports, Politics, & Miscellaneous**

- `rec.autos`
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`
- `talk.politics.misc`
- `talk.politics.guns`
- `talk.politics.mideast`
- `talk.religion.misc`
- `misc.forsale`
- `alt.atheism`
- `soc.religion.christian`

- **Training Set:** 11,314 documents
- **Test Set:** Remaining documents

---

## ⚙️ Project Overview

The notebook `Homework3.ipynb` (Google Colab compatible) performs the following steps:

1. **Dataset Preparation**

   - Downloads and converts the 20 Newsgroups dataset into two classes.

2. **Preprocessing**

   - Tokenization
   - Stopword removal
   - Optional: Lemmatization / stemming
   - TF-IDF vectorization or Bag-of-Words feature extraction

3. **Model Training**

   - Splits the dataset into training and validation sets.
   - Trains multiple binary classification models (e.g., Naive Bayes, Logistic Regression, SVM).
   - Selects the best-performing model based on validation accuracy.

4. **Evaluation**

   - Tests the selected model on unseen test data.
   - Reports performance metrics:
     - Accuracy
     - Precision
     - Recall
     - F1-score

5. **Results & Discussion**
   - Summarizes the performance results.
   - Discusses model behavior, challenges, and key findings.

---

## 📊 Example Output

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.92  |
| Precision | 0.91  |
| Recall    | 0.90  |
| F1-Score  | 0.90  |

_(Example results; actual numbers may vary depending on model and preprocessing choices.)_

---

## 🧩 Technologies Used

- Python 3.x
- Google Colab / Jupyter Notebook
- Scikit-learn
- NLTK or SpaCy (for text preprocessing)
- NumPy, Pandas
- Matplotlib / Seaborn (for visualization)
