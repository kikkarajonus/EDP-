import pandas as pd
import numpy as np
import re
import math
from collections import Counter
filename = input("Enter CSV file name: ")

df = pd.read_csv(spam_dataset.csv)

print("\n" + "=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(df)
print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Number of messages:", len(df))
print("Number of columns:", len(df.columns))

print("\nLabels:")
print(df["Label"].value_counts())
def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Split sentence into words
    words = text.split()

    # Simple stop words
    stop_words = {
        "a", "an", "the", "is", "are", "am",
        "to", "for", "of", "in", "on", "at",
        "and", "or", "you", "your", "i",
        "me", "my", "this", "that", "be",
        "our", "we", "it"
    }

    # Remove stop words
    words = [
        word for word in words
        if word not in stop_words
    ]

    return words


# Apply preprocessing

df["Processed_Text"] = df["Message"].apply(preprocess_text)

print("\n" + "=" * 60)
print("TEXT AFTER PREPROCESSING")
print("=" * 60)

for i in range(min(5, len(df))):
    print("Original :", df["Message"].iloc[i])
    print("Processed:", df["Processed_Text"].iloc[i])
    print()
# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(0.8 * len(df))

train_df = df[:split_index]
test_df = df[split_index:]

print("=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Total messages :", len(df))
print("Training data  :", len(train_df))
print("Testing data   :", len(test_df))
vocabulary = set()

for words in train_df["Processed_Text"]:

    for word in words:
        vocabulary.add(word)

vocabulary = sorted(list(vocabulary))

print("\n" + "=" * 60)
print("VOCABULARY")
print("=" * 60)

print("Total unique words:", len(vocabulary))

print(vocabulary)
def calculate_tf(words):

    word_count = Counter(words)

    total_words = len(words)

    tf = {}

    for word in vocabulary:

        if total_words == 0:
            tf[word] = 0
        else:
            tf[word] = word_count[word] / total_words

    return tf


def calculate_idf(documents):

    total_documents = len(documents)

    idf = {}

    for word in vocabulary:

        document_count = 0

        for document in documents:

            if word in document:
                document_count += 1

        # Smooth IDF
        idf[word] = math.log(
            (total_documents + 1) /
            (document_count + 1)
        ) + 1

    return idf


# Calculate IDF using training data

training_documents = train_df["Processed_Text"].tolist()

idf = calculate_idf(training_documents)
def tfidf_vector(words):

    tf = calculate_tf(words)

    vector = []

    for word in vocabulary:

        value = tf[word] * idf[word]

        vector.append(value)

    return np.array(vector)


X_train = np.array([
    tfidf_vector(words)
    for words in train_df["Processed_Text"]
])

X_test = np.array([
    tfidf_vector(words)
    for words in test_df["Processed_Text"]
])

print("\n" + "=" * 60)
print("TF-IDF")
print("=" * 60)

print("Training TF-IDF shape:", X_train.shape)
print("Testing TF-IDF shape :", X_test.shape)
class NaiveBayes:

    def __init__(self):

        self.classes = None
        self.class_prior = {}
        self.mean = {}
        self.variance = {}

    def fit(self, X, y):

        self.classes = np.unique(y)

        for class_value in self.classes:

            X_class = X[y == class_value]

            # Prior probability
            self.class_prior[class_value] = (
                len(X_class) / len(X)
            )

            # Mean
            self.mean[class_value] = np.mean(
                X_class,
                axis=0
            )

            # Variance
            self.variance[class_value] = (
                np.var(X_class, axis=0) + 1e-9
            )

    def gaussian_probability(
        self,
        x,
        mean,
        variance
    ):

        exponent = np.exp(
            -((x - mean) ** 2) /
            (2 * variance)
        )

        return (
            1 /
            np.sqrt(2 * np.pi * variance)
        ) * exponent

    def predict_one(self, x):

        probabilities = {}

        for class_value in self.classes:

            # Start with prior probability
            log_probability = math.log(
                self.class_prior[class_value]
            )

            mean = self.mean[class_value]
            variance = self.variance[class_value]

            probabilities_values = (
                self.gaussian_probability(
                    x,
                    mean,
                    variance
                )
            )

            # Use logarithm to avoid numerical underflow
            log_probability += np.sum(
                np.log(
                    probabilities_values + 1e-9
                )
            )

            probabilities[class_value] = log_probability

        return max(
            probabilities,
            key=probabilities.get
        )

    def predict(self, X):

        predictions = []

        for x in X:

            predictions.append(
                self.predict_one(x)
            )

        return np.array(predictions)
model = NaiveBayes()

model.fit(
    X_train,
    train_df["Label"].values
)

print("\n" + "=" * 60)
print("NAIVE BAYES MODEL TRAINED")
print("=" * 60)
y_test = test_df["Label"].values

y_pred = model.predict(X_test)

print("\nActual Labels:")
print(y_test)

print("\nPredicted Labels:")
print(y_pred)
correct = np.sum(y_test == y_pred)

accuracy = correct / len(y_test)


# Spam is considered the positive class

tp = 0
fp = 0
fn = 0

for actual, predicted in zip(y_test, y_pred):

    if actual == "spam" and predicted == "spam":
        tp += 1

    elif actual == "ham" and predicted == "spam":
        fp += 1

    elif actual == "spam" and predicted == "ham":
        fn += 1


if tp + fp == 0:
    precision = 0
else:
    precision = tp / (tp + fp)


if tp + fn == 0:
    recall = 0
else:
    recall = tp / (tp + fn)


if precision + recall == 0:
    f1_score = 0
else:
    f1_score = (
        2 * precision * recall
        / (precision + recall)
    )


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1_score * 100, 2), "%")
print("\n" + "=" * 60)
print("SPAM CLASSIFIER")
print("=" * 60)

new_message = input(
    "Enter a message to classify: "
)

processed_message = preprocess_text(
    new_message
)

new_vector = tfidf_vector(
    processed_message
)

prediction = model.predict(
    np.array([new_vector])
)

print("\nMessage:")
print(new_message)

print("\nPrediction:")

if prediction[0] == "spam":
    print("🚨 SPAM MESSAGE")
else:
    print("✅ HAM (NOT SPAM)")
