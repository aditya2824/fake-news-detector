# 1. IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 2. LOAD BOTH CSV FILES
fake = pd.read_csv("Fake.csv")   # news labeled as fake
true = pd.read_csv("True.csv")   # news labeled as real

# 3. ADD LABELS
fake["label"] = 0   # Fake news = 0
true["label"] = 1   # Real news = 1

# 4. COMBINE & SHUFFLE
data = pd.concat([fake, true])           # combine both
data = data.sample(frac=1).reset_index(drop=True)  # shuffle

# 5. CREATE TEXT COLUMN
data["text"] = data["title"] + " " + data["text"]  # merge title + text
X = data["text"]     # input (features)
y = data["label"]    # output (target)

# 6. TEXT PREPROCESSING & VECTORIZE
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# 7. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# 8. TRAIN MODEL
model = LogisticRegression()
model.fit(X_train, y_train)

# 9. EVALUATE MODEL
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 10. SAVE MODEL & VECTORIZER
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ model.pkl and vectorizer.pkl saved successfully!")
