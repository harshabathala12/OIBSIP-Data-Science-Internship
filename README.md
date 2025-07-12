# 🧠 Data Science Internship Projects – Oasis Infobyte (Harsha Bathala)

This repository contains **three end-to-end Data Science projects** completed as part of the **AICTE-approved Oasis Infobyte Virtual Internship Program**. Each project demonstrates skills in data preprocessing, model building, evaluation, and visualization using Python and popular ML libraries.

---

## 📁 Projects Overview

| Task | Project Name                  | Description |
|------|-------------------------------|-------------|
| 1️⃣  | [Email Spam Detection](#-1-email-spam-detection)        | Binary classification of SMS messages into spam or ham using Logistic Regression |
| 2️⃣  | [Car Price Prediction](#-2-car-price-prediction)        | Regression model to estimate used car selling prices based on features like age, fuel, and kms driven |
| 3️⃣  | [Unemployment Analysis](#-3-unemployment-analysis)      | Visual analytics project exploring unemployment rates across Indian states using heatmaps and time trends |

---

## ✅ Folder Structure

```bash
.
├── Task-1-Email-Spam-Detection/
│   ├── dataset/
│   ├── results/
│   ├── spam_classifier.py
│   ├── README.md
│   └── requirements.txt
│
├── Task-2-Car-Price-Prediction/
│   ├── dataset/
│   ├── results/
│   ├── car_price_predictor.py
│   ├── README.md
│   └── requirements.txt
│
├── Task-3-Unemployment-Analysis/
│   ├── dataset/
│   ├── results/
│   ├── unemployment_analysis.py
│   ├── README.md
│   └── requirements.txt
│
└── requirements.txt  # combined (optional)

---

📦 Installation
To install all dependencies for all projects at once, run:

pip install -r requirements.txt
Or, run inside any individual task folder:
pip install -r Task-1-Email-Spam-Detection/requirements.txt

---

🔍 1. Email Spam Detection

Goal: Classify messages as spam or ham using Logistic Regression
Dataset: SMS Spam Collection Dataset
Tech Stack: scikit-learn, pandas, matplotlib, seaborn
Highlight: Applied threshold tuning to improve spam detection recall

🎯 Final Results
Metric	Value
Accuracy	97.85%
Spam Recall	91.33%
Spam F1-Score	91.95%
Optimal Threshold	0.47

📁 Output:
Confusion Matrix: results/confusion_matrix.png

---

🚗 2. Car Price Prediction

Goal: Predict selling price of used cars using regression
Dataset: Car Price Prediction - Used Cars
Tech Stack: scikit-learn, joblib, pandas, seaborn
Highlight: Achieved high accuracy using feature engineering (car age) and ensemble modeling

🎯 Final Results
Metric	Value
R² Score	95.81%
RMSE	0.9824
Model Used	Random Forest Regressor

📁 Output:
Predicted vs Actual Plot: results/actual_vs_predicted.png

---

📊 3. Unemployment Analysis

Goal: Visualize and analyze unemployment trends across Indian states
Dataset: Unemployment in India (CMIE)
Tech Stack: pandas, matplotlib, seaborn
Highlight: State-wise heatmap, urban vs rural boxplot, top 10 states ranked by unemployment

📈 Visualizations Saved
Plot	Filename
National Unemployment Trend (Line)	results/national_unemployment_trend.png
State-wise Heatmap	results/statewise_heatmap.png
Latest State-wise Barplot	results/statewise_barplot_latest.png
Urban vs Rural Comparison	results/urban_vs_rural_boxplot.png
Labour Participation Trend	results/labour_participation_trend.png
Top 10 Highest Unemployment States	results/top10_unemployment_states.png

---

👨‍💻 Author
Harsha Bathala
🎓 Final-Year B.Tech CSE | AI/ML & Full-Stack Developer
🔗 GitHub: harshabathala12
🔗 LinkedIn: bathalaharsha

---

🙌 Acknowledgements

Oasis Infobyte for providing a structured internship framework
Kaggle for open-access datasets

