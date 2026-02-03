
import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

nltk.download("stopwords")


print("Loading dataset...")

df = pd.read_csv("data/spam.csv", encoding="latin-1")


df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print(df.head())
print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df['label'].value_counts())


df['label'] = df['label'].map({'ham': 0, 'spam': 1})
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)       # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

print("\nCleaned text sample:")
print(df[['text', 'clean_text']].head())


X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


print("\nTraining Naive Bayes model...")

model_nb = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model_nb.fit(X_train, y_train)


y_pred = model_nb.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model_nb, "models/nb_spam_model.pkl")
