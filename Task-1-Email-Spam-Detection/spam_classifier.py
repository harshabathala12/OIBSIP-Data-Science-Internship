import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and clean dataset
df = pd.read_csv('Task-1-Email-Spam-Detection/dataset/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# 2. Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 4. Train Logistic Regression model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]  # Prob of spam

# 6. Threshold tuning to find best F1-score for spam
from sklearn.metrics import classification_report

best_f1 = 0
best_threshold = 0
best_report = None
best_accuracy = 0

print("🔬 Threshold Tuning Results:\n")
for t in [i / 100 for i in range(30, 61, 1)]:  # 0.30 to 0.60
    y_pred_t = (y_probs > t).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    report = classification_report(y_test, y_pred_t, target_names=["Ham", "Spam"], output_dict=True)

    spam_f1 = report['Spam']['f1-score']
    spam_recall = report['Spam']['recall']

    print(f"📌 Threshold: {t:.2f} | Accuracy: {acc:.4f} | Spam Recall: {spam_recall:.4f} | Spam F1: {spam_f1:.4f}")

    if spam_f1 > best_f1:
        best_f1 = spam_f1
        best_threshold = t
        best_report = report
        best_accuracy = acc

# 7. Final prediction using best threshold
y_pred = (y_probs > best_threshold).astype(int)

# 8. Final evaluation printout
print("\n✅ Best Threshold Selected Automatically")
print(f"🎯 Threshold: {best_threshold:.2f}")
print(f"📊 Accuracy: {best_accuracy:.4f}")
print(f"🧾 Final Spam Recall: {best_report['Spam']['recall']:.4f}")
print(f"🏆 Final Spam F1-Score: {best_report['Spam']['f1-score']:.4f}")

# 9. Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
os.makedirs("Task-1-Email-Spam-Detection/results", exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix (Optimized Spam Classifier)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("Task-1-Email-Spam-Detection/results/confusion_matrix.png")
print("📁 Confusion matrix saved at: Task-1-Email-Spam-Detection/results/confusion_matrix.png")