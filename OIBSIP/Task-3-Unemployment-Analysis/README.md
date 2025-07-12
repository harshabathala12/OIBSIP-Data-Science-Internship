# 📊 Unemployment Analysis in India

This project analyzes the **unemployment trends across Indian states** using real-world data. Created as part of the **Oasis Infobyte Data Science Internship**, it covers end-to-end **EDA**, **data visualization**, and **insights extraction** from a structured dataset.

---

## 🧠 Project Overview

- **Goal**: Analyze unemployment patterns across states and over time, and visualize the trends by geography and region type (Urban/Rural).
- **Dataset**: [Unemployment in India – Kaggle](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)
- **Format**: Monthly unemployment stats from different Indian states, including area type and labor metrics.
- **Libraries Used**: `Pandas`, `Matplotlib`, `Seaborn`

---

## 📈 Visualizations Generated

| 📊 Plot Type | Description |
|-------------|-------------|
| National Trend Line | Average unemployment rate across India over time |
| State-wise Heatmap | Unemployment variation across states (calendar view) |
| Latest State Barplot | Top states sorted by most recent unemployment rates |
| Urban vs Rural Boxplot | Compare distribution between urban and rural regions |
| Labour Participation Trend | Avg labour participation % trend over time |
| Top 10 States Barplot | Top 10 states with highest recent unemployment |

---

## 🛠 Workflow Summary

1. **Data Loading & Cleaning**
   - Loaded CSV, renamed columns for consistency, converted date columns to `datetime`.
   - Mapped values like `"Region"` to `"state"` and cleaned whitespace.

2. **National Trend Analysis**
   - Grouped by date to plot overall unemployment rate progression.

3. **State-Level Heatmap**
   - Used pivot tables to display unemployment across time and geography.

4. **Area Type Comparison**
   - Compared unemployment between Urban and Rural states using boxplots.

5. **Top States Ranking**
   - Ranked states with the highest recent unemployment.

---

## 📊 Sample Output

Here are examples of the visual outputs:

- **📈 National Trend**  
  ![Trend](Task-3-Unemployment-Analysis/results/national_unemployment_trend.png)

- **🗺️ State-wise Heatmap**  
  ![Heatmap](Task-3-Unemployment-Analysis/results/statewise_heatmap.png)

- **📊 Latest State Ranking**  
  ![Barplot](Task-3-Unemployment-Analysis/results/statewise_barplot_latest.png)

- **📦 Area Type Boxplot**  
  ![Boxplot](Task-3-Unemployment-Analysis/results/area_comparison_boxplot.png)

- **📉 Labour Participation Rate**  
  ![Lineplot](Task-3-Unemployment-Analysis/results/labour_participation_trend.png)

- **🏆 Top 10 Unemployment States**  
  ![Top10](Task-3-Unemployment-Analysis/results/top10_unemployment_states.png)

---

## 📂 Project Structure

Task-3-Unemployment-Analysis/
├── dataset/
│ └── Unemployment in India.csv
├── results/
│ ├── national_unemployment_trend.png
│ ├── statewise_heatmap.png
│ ├── statewise_barplot_latest.png
│ ├── area_comparison_boxplot.png
│ ├── labour_participation_trend.png
│ └── top10_unemployment_states.png
├── unemployment_analysis.py
└── README.md

---

## 📌 Requirements

- Python 3.x
- pandas
- seaborn
- matplotlib

Install them via:

```bash
pip install pandas matplotlib seaborn

---

🧠 Insights 

📉 National unemployment showed visible fluctuation between 2019–2020 due to economic events.
🏙️ Urban areas generally have higher unemployment variance compared to rural areas.
📍 States like Tripura, Haryana, Rajasthan appeared frequently among the top unemployed.
📊 Labour participation rate and unemployment often show inverse patterns.

---

👤 Author
Harsha Bathala
🎓 Final-Year B.Tech CSE | AI/ML & Full-Stack Developer
🔗 GitHub: harshabathala12
🔗 LinkedIn: bathalaharsha