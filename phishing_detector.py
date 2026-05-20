import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("emails.csv")

# Features and labels
X = df["email"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Convert text to numerical form
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# User testing
while True:
    msg = input("\nEnter email text (or type 'exit'): ")

    if msg.lower() == "exit":
        break

    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    if prediction[0] == 1:
        print("⚠️ Phishing Email")
    else:
        print("✔️ Safe Email")