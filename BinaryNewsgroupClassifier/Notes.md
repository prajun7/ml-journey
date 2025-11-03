## Model Used for Binary Text Classification

This report analyzes the six machine learning models trained to classify the 20 Newsgroups dataset. All models were trained on the same **TF-IDF vectorized data**, which is a high-dimensional and sparse dataset (many features/words, but most are zero for any given document).

---

### 1. Multinomial Naive Bayes (MNB)

- **Model Type:** **Probabilistic Classifier**
- **How it Works:** It uses Bayes' theorem to calculate the probability of a document belonging to a class based on the words it contains. It's "naive" because it assumes every word is independent of the others (e.g., "operating" and "system" are treated as two unrelated clues).
- **Good Fit (Pros):**
  - Extremely fast and efficient.
  - Works exceptionally well with high-dimensional, sparse text data (like TF-IDF).
  - It's a fantastic **baseline model**; any other model you try should have to beat this one.
  - We were able to extract "feature importance" (most predictive words) from it, making it interpretable.
- **Not-So-Good Fit (Cons):**
  - Its core assumption of word independence is fundamentally wrong, which can limit its accuracy.
- **Hyperparameters:**
  - **Main Parameter:** `alpha` (a smoothing parameter to handle words that don't appear in the training set).
  - **What We Used:** We used the default `alpha=1.0`, which is standard and robust.

---

### 2. Logistic Regression

- **Model Type:** **Linear Model (that outputs probabilities)**
- **How it Works:** It finds a linear boundary (a "line" or "plane" in high-dimensional space) that best separates the two classes. It then passes this linear function through a _sigmoid_ function to output a probability (0.0 to 1.0).
- **Good Fit (Pros):**
  - Very fast to train and predict.
  - Tends to be highly accurate on text data, often rivaling more complex models.
  - Outputs well-calibrated probabilities, which is why it produces a smooth, reliable ROC curve.
- **Not-So-Good Fit (Cons):**
  - It can only capture **linear relationships**. If the "Tech/Sci" class is best separated from "Others" by a complex, non-linear boundary, this model will fail to find it.
- **Hyperparameters:**
  - **Main Parameter:** `C` (controls regularization, or how much to penalize "complex" solutions).
  - **What We Used:** We used the default `C=1.0` and set `max_iter=1000` to give the model enough time to converge (finish training).

---

### 3. Linear Support Vector Machine (LinearSVC)

- **Model Type:** **Linear Model** (a type of Support Vector Machine)
- **How it Works:** Like Logistic Regression, it finds a linear boundary. However, its goal is to find the one that creates the **largest possible margin** (or "street") between the two classes. It's less concerned with probabilities and more focused on this maximum-margin separation.
- **Good Fit (Pros):**
  - Often considered one of the **best all-around text classifiers**.
  - Very effective in high-dimensional spaces (like TF-IDF).
  - Good at avoiding overfitting.
- **Not-So-Good Fit (Cons):**
  - Like Logistic Regression, it is linear and cannot capture non-linear patterns.
  - It doesn't naturally produce probabilities, which is why we had to use `decision_function()` for the ROC curve instead of `predict_proba()`.
- **Hyperparameters:**
  - **Main Parameter:** `C` (regularization strength).
  - **What We Used:** We used the default `C=1.0` and set `dual="auto"` to automatically select the best algorithm for your data's shape.

---

### 4. Random Forest

- **Model Type:** **Ensemble (Bagging / Decision Tree-based)**
- **How it Works:** It builds a large "forest" of many individual, simple **decision trees**. Each tree is trained on a random subset of the data. The final prediction is made by taking a majority vote from all the trees.
- **Good Fit (Pros):**
  - Excellent at capturing complex, **non-linear** patterns.
  - Very hard to overfit, especially with many trees.
- **Not-So-Good Fit (Cons):**
  - **Much slower** to train and predict than linear models.
  - Often performs _worse_ than linear models on very sparse text data. Decision trees can struggle when faced with 100,000+ features where most are zero.
- **Hyperparameters:**
  - **Main Parameters:** `n_estimators` (number of trees), `max_depth` (how deep each tree can be).
  - **What We Used:** We used the defaults (like `n_estimators=100`) and set `n_jobs=-1` to use all your CPU cores for faster training.

---

### 5. K-Nearest Neighbors (KNN)

- **Model Type:** **Instance-Based (or "Lazy") Learner**
- **How it Works:** It doesn't "train" a model at all. It just **memorizes** the entire training dataset. To make a prediction, it finds the _k_ (e.g., 5) most similar documents from its memory and takes a majority vote.
- **Good Fit (Pros):**
  - Extremely simple concept.
- **Not-So-Good Fit (Cons):**
  - **Very bad choice for this task.**
  - **Extremely slow** at prediction time (it has to compare a new document to all 9,000+ training documents).
  - Suffers badly from the "curse of dimensionality." In a space with 100,000+ dimensions (features), the idea of "distance" or "nearest" becomes almost meaningless.
- **Hyperparameters:**
  - **Main Parameter:** `n_neighbors` (the _k_ value).
  - **What We Used:** We used the default `n_neighbors=5` and set `n_jobs=-1` to speed up the distance calculations.

---

### 6. Multi-layer Perceptron (MLP)

- **Model Type:** **Neural Network**
- **How it Works:** This is a simple (or "shallow") neural network. Data flows through an input layer, one or more "hidden layers" (where non-linear computations happen), and an output layer that makes the final prediction.
- **Good Fit (Pros):**
  - It's a "universal approximator," meaning it can learn **any non-linear relationship** given enough neurons.
  - Often performs at the top level, competing with LinearSVC and Logistic Regression.
- **Not-So-Good Fit (Cons):**
  - Can be slow to train.
  - Can be sensitive to hyperparameters. Choosing the right number of layers and neurons can require some tuning.
  - Can overfit if not regulated.
- **Hyperparameters:**
  - **Main Parameters:** `hidden_layer_sizes`, `activation` (math function in the neuron), `alpha` (regularization).
  - **What We Used:** We used `hidden_layer_sizes=(100,)` (one layer of 100 neurons), `max_iter=300`, and `early_stopping=True` (a very smart way to stop training automatically before it starts to overfit).
