# Spam Classifier Using TF-IDF and Naive Bayes

## Overview

This project builds a **Spam Message Classifier** using Python. It demonstrates fundamental Natural Language Processing (NLP) and Machine Learning concepts:

* Text Preprocessing
* TF-IDF (Term Frequency–Inverse Document Frequency)
* Naive Bayes Classification
* Train-Test Split
* Model Evaluation
* New Message Prediction

The project is implemented **without using Scikit-Learn**. The TF-IDF calculation and Naive Bayes classifier are implemented manually using Python, Pandas, NumPy, and built-in libraries.

---

## Objective

The main objective is to develop a program that can determine whether a given text message is:

* **Spam** — unwanted or promotional message
* **Ham** — normal/non-spam message

For example:

```text
Congratulations! You won a free prize. Claim now!
```

Output:

```text
SPAM MESSAGE
```

Whereas:

```text
Can you send me the assignment notes?
```

Output:

```text
HAM (NOT SPAM)
```

---

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Regular Expressions (`re`)
* Math
* Collections

No Scikit-Learn is used in this project.

---

## Project Structure

```text
SpamClassifier/
│
├── spam_dataset.csv
├── spam_classifier.py
└── README.md
```

---

## Dataset

The program reads the dataset from a CSV file.

The CSV file should contain two columns:

```text
Message,Label
```

Example:

```csv
Message,Label
"Congratulations! You won a free lottery ticket. Claim now","spam"
"Win cash prize now! Click this link","spam"
"Meeting is scheduled at 10 AM tomorrow","ham"
"Please submit the assignment before Friday","ham"
```

### Labels

| Label  | Meaning                      |
| ------ | ---------------------------- |
| `spam` | Unwanted/promotional message |
| `ham`  | Normal message               |

Save the dataset as:

```text
spam_dataset.csv
```

---

## Installation

Make sure Python 3.x is installed.

Install the required libraries:

```bash
pip install pandas numpy
```

---

## How to Run

Run the Python program:

```bash
python spam_classifier.py
```

The program will ask for the CSV filename:

```text
Enter CSV file name:
```

Enter:

```text
spam_dataset.csv
```

---

# Project Workflow

The complete workflow of the project is:

```text
CSV Dataset
     ↓
Text Preprocessing
     ↓
Train-Test Split
     ↓
Create Vocabulary
     ↓
TF-IDF
     ↓
Naive Bayes
     ↓
Prediction
     ↓
Evaluation
     ↓
New Message Classification
```

---

# 1. Text Preprocessing

Raw text cannot be directly given to a machine learning model.

The program performs basic preprocessing.

### Operations performed

1. Convert text to lowercase
2. Remove punctuation and special characters
3. Split text into individual words
4. Remove common stop words

Example:

```text
Original:
Congratulations! You won a FREE prize!

After preprocessing:
['congratulations', 'won', 'free', 'prize']
```

---

# 2. Train-Test Split

The dataset is divided into two parts:

```text
80% → Training Data
20% → Testing Data
```

### Training Data

Used by the model to learn patterns from messages.

### Testing Data

Used to evaluate the model on messages it has not seen during training.

---

# 3. TF-IDF

TF-IDF converts text into numerical values so that the machine learning algorithm can process it.

TF-IDF stands for:

**Term Frequency–Inverse Document Frequency**

It consists of two main components.

### Term Frequency (TF)

Measures how frequently a word occurs in a document.

```text
TF = Number of times a word appears / Total number of words
```

### Inverse Document Frequency (IDF)

Measures how important a word is across all documents.

The project uses a smoothed IDF calculation:

```text
IDF = log((N + 1) / (DF + 1)) + 1
```

Where:

* `N` = Total number of documents
* `DF` = Number of documents containing the word

### TF-IDF

```text
TF-IDF = TF × IDF
```

The result is a numerical representation of each message.

---

# 4. Naive Bayes

The project implements a **Naive Bayes classifier from scratch**.

Naive Bayes calculates the probability that a message belongs to each class.

The two classes are:

```text
spam
ham
```

The model learns statistical patterns from the TF-IDF vectors.

For this implementation, a Gaussian probability model is used for the numerical TF-IDF features.

---

# 5. Prediction

After training, the model predicts the labels of the testing messages.

Example:

```text
Actual:
['spam', 'ham', 'spam', 'ham']

Predicted:
['spam', 'ham', 'spam', 'ham']
```

---

# 6. Evaluation Metrics

The model is evaluated using four metrics.

## Accuracy

Measures the percentage of correctly classified messages.

```text
Accuracy = Correct Predictions / Total Predictions
```

A higher accuracy indicates better overall classification performance.

---

## Precision

Precision measures how many messages predicted as spam were actually spam.

```text
Precision = TP / (TP + FP)
```

Where:

* `TP` = True Positive
* `FP` = False Positive

---

## Recall

Recall measures how many actual spam messages were correctly identified.

```text
Recall = TP / (TP + FN)
```

Where:

* `TP` = True Positive
* `FN` = False Negative

---

## F1 Score

F1 Score combines Precision and Recall.

```text
F1 = 2 × (Precision × Recall) /
     (Precision + Recall)
```

A higher F1 Score generally indicates a better balance between precision and recall.

---

# 7. New Message Classification

After evaluating the model, the program allows the user to enter a new message.

Example:

```text
Enter a message to classify:
Congratulations! You won a free gift. Claim now!
```

Output:

```text
Prediction:
SPAM MESSAGE
```

Another example:

```text
Enter a message to classify:
Can you send me the Python notes?
```

Output:

```text
Prediction:
HAM (NOT SPAM)
```

---

# Sample Output

```text
============================================================
DATASET LOADED SUCCESSFULLY
============================================================

Number of messages: 20
Number of columns: 2

Labels:
spam    10
ham     10

============================================================
TRAIN-TEST SPLIT
============================================================

Total messages : 20
Training data  : 16
Testing data   : 4

============================================================
VOCABULARY
============================================================

Total unique words: ...

============================================================
TF-IDF
============================================================

Training TF-IDF shape: (16, ...)
Testing TF-IDF shape : (4, ...)

============================================================
NAIVE BAYES MODEL TRAINED
============================================================

============================================================
MODEL EVALUATION
============================================================

Accuracy : ... %
Precision: ... %
Recall   : ... %
F1 Score : ... %

============================================================
SPAM CLASSIFIER
============================================================

Enter a message to classify:
Win a free prize now!

Prediction:
SPAM MESSAGE
```

---

# Learning Outcomes

After completing this project, you will understand:

* How text data is cleaned
* How words are converted into numerical representations
* How TF-IDF works
* How a Naive Bayes classifier works
* How to split data into training and testing sets
* How classification predictions are made
* How to calculate Accuracy, Precision, Recall, and F1 Score
* How to implement a basic NLP classifier without Scikit-Learn

---

# Advantages

* Simple and beginner-friendly
* No Scikit-Learn dependency
* Demonstrates the internal working of TF-IDF
* Demonstrates Naive Bayes implementation
* Supports user-entered messages
* Easy to modify with a larger dataset

---

# Limitations

* The dataset is relatively small.
* Text preprocessing is basic.
* The stop-word list is manually defined.
* The model may not perform well on messages that are very different from the training dataset.
* The implementation uses a simple Gaussian Naive Bayes approach for TF-IDF features.

---

# Future Enhancements

The project can be improved by adding:

* Larger real-world SMS spam datasets
* Advanced text preprocessing
* Stemming and lemmatization
* N-grams
* Better stop-word handling
* Confusion matrix visualization
* Precision/Recall graphs
* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine
* Web interface using Flask
* Real-time email/SMS spam detection

---

# Conclusion

This project demonstrates how a basic **Natural Language Processing classification system** can be built using Python without relying on Scikit-Learn.

The system processes text messages, converts them into numerical TF-IDF representations, trains a manually implemented Naive Bayes classifier, evaluates its performance, and predicts whether new messages are **Spam** or **Ham**.

It provides a practical introduction to the fundamentals of **Text Preprocessing, TF-IDF, Naive Bayes, and Text Classification**.
