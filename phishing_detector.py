import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset (small, built-in)
data = {
    "email": [
        "Click here to win money now",
        "Your account has been suspended login now",
        "Verify your bank details immediately",
        "Urgent: update your password now",
        "Meeting scheduled at 10am",
        "Project deadline is tomorrow",
        "Let's have lunch tomorrow",
        "Your order has been shipped"
    ],
    "label": [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = phishing, 0 = safe
}

df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["email"], df["label"], test_size=0.25, random_state=42
)

# Convert text → numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# User input testing
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