# 📬 Email Spam Detection using Machine Learning

This project classifies emails as **Spam** or **Ham** using machine learning techniques. Built as part of the **Oasis Infobyte Data Science Internship**, this project demonstrates the full data science workflow: data cleaning, preprocessing, model training, evaluation, threshold tuning, and visualization.

---

## 🧠 Project Overview

- **Goal**: Detect spam emails accurately using a supervised learning model.
- **Dataset**: [SMS Spam Collection Dataset from Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Model Used**: Logistic Regression
- **Libraries**: Scikit-learn, Pandas, Matplotlib, Seaborn
- **Techniques**: TF-IDF Vectorization, Class Weight Balancing, Threshold Optimization

---

## 🚀 Final Model Performance (Optimized)

| Metric            | Value        |
|-------------------|--------------|
| 🎯 Threshold       | **0.47**     |
| 📊 Accuracy        | **97.85%**   |
| 🧾 Spam Recall     | **91.33%**   |
| 🏆 Spam F1-Score   | **91.95%**   |

> Threshold tuning was applied to enhance real-world spam classification, prioritizing spam recall and F1-score.

---

## 🛠 Workflow Summary

1. **Data Loading & Cleaning**
   - Loaded dataset from Kaggle, removed unnecessary columns, mapped labels to binary (ham = 0, spam = 1).

2. **Text Preprocessing**
   - Used `TfidfVectorizer` to convert SMS text into feature vectors with stopword removal.

3. **Model Training**
   - Trained `LogisticRegression` with `class_weight='balanced'` to handle class imbalance.

4. **Threshold Tuning**
   - Evaluated thresholds from 0.30 to 0.60 to find the optimal trade-off between recall and precision.

5. **Evaluation**
   - Generated a classification report and confusion matrix.
   - Best model saved and evaluated using optimized threshold.

---

## 📊 Confusion Matrix

A visual representation of predictions vs actual labels.

![Confusion Matrix](Task-1-Email-Spam-Detection/results/confusion_matrix.png)

---

## 📂 Project Structure
Email-Spam-Detection/
├── dataset/
│ └── spam.csv
├── results/
│ └── confusion_matrix.png
├── spam_classifier.py
└── README.md

---

## 📌 Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

Install dependencies using:

```bash
pip install -r requirements.txt

---

🏁 Output Sample

✅ Best Threshold Selected Automatically
🎯 Threshold: 0.47
📊 Accuracy: 0.9785
🧾 Final Spam Recall: 0.9133
🏆 Final Spam F1-Score: 0.9195
📁 Confusion matrix saved at: results/confusion_matrix.png

---

👤 Author
Harsha Bathala
🎓 Final-Year B.Tech CSE | AI/ML & Full-Stack Developer
🔗 GitHub: harshabathala12
🔗 LinkedIn: bathalaharsha