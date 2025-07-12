# 🚘 Car Price Prediction using Machine Learning

This project builds a regression model to predict **used car prices** based on features like mileage, fuel type, transmission, and car age. Built as part of the **Oasis Infobyte Data Science Internship**, this project demonstrates the complete data science pipeline including feature engineering, model optimization, and performance evaluation.

---

## 🧠 Project Overview

- **Goal**: Predict the selling price of used cars using machine learning.
- **Dataset**: [Used Car Price Dataset from Kaggle](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)
- **Model Used**: Random Forest Regressor
- **Libraries**: Scikit-learn, Pandas, Matplotlib, Seaborn
- **Techniques**: Label Encoding, Feature Engineering, Hyperparameter Tuning

---

## 🚀 Final Model Performance (Optimized)

| Metric            | Value        |
|-------------------|--------------|
| 🎯 R² Score        | **95.81%**   |
| 📉 RMSE            | **0.9824**   |
| 🧠 Model           | RandomForestRegressor (n_estimators=200)

> Achieved high accuracy using an ensemble model with careful feature engineering and tuning.

---

## 🛠 Workflow Summary

1. **Data Loading & Cleaning**
   - Loaded dataset and removed `Car_Name`.
   - Calculated `Car_Age` from `Year`.

2. **Feature Encoding**
   - Applied label encoding to categorical columns: `Fuel_Type`, `Selling_type`, `Transmission`.

3. **Model Training**
   - Trained a `RandomForestRegressor` with `n_estimators=200` using train-test split.

4. **Evaluation**
   - Evaluated the model using R² score and RMSE.
   - Visualized predicted vs actual values with a scatter plot.

---

## 📊 Prediction Visualization

A scatter plot to compare predicted car prices with actual ones.

![Prediction vs Actual](Task-2-Car-Price-Prediction/results/actual_vs_predicted.png)

---

## 📂 Project Structure

Task-2-Car-Price-Prediction/
├── dataset/
│ └── car_data.csv
├── results/
│ └── actual_vs_predicted.png
├── car_price-predictor.py
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

🏁 Output Sample:

📊✅ Model Evaluation Complete!
🎯 R² Score: 0.9581
📉 RMSE: 0.9824
📁 Plot saved to: Task-2-Car-Price-Prediction/results/actual_vs_predicted.png

---

👤 Author
Harsha Bathala
🎓 Final-Year B.Tech CSE | AI/ML & Full-Stack Developer
🔗 GitHub: harshabathala12
🔗 LinkedIn: bathalaharsha