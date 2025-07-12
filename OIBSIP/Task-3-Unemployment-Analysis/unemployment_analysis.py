import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"  # Avoid emoji warnings

# Step 1: Load Dataset
df = pd.read_csv("Task-3-Unemployment-Analysis/dataset/Unemployment in India.csv")
print("📦 Dataset Loaded. Sample:\n")
print(df.head())
print(f"\n🧮 Dataset Shape: {df.shape}")

# Step 2: Data Cleaning & Renaming
df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
df = df.rename(columns={
    'region': 'state',
    'estimated_unemployment_rate_(%)': 'unemployment_rate',
    'estimated_employed': 'employed',
    'estimated_labour_participation_rate_(%)': 'labour_participation_rate'
})
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['state'] = df['state'].str.strip()

# Step 3: Create results folder
os.makedirs("Task-3-Unemployment-Analysis/results", exist_ok=True)

# Step 4: National Unemployment Trend
plt.figure(figsize=(12, 6))
national_avg = df.groupby('date')['unemployment_rate'].mean()
sns.lineplot(x=national_avg.index, y=national_avg.values, color='red')
plt.title("National Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("Task-3-Unemployment-Analysis/results/national_unemployment_trend.png")
plt.close()
print("✅ Saved national unemployment trend plot.")

# Step 5: Heatmap - Unemployment by State Over Time
pivot_table = df.pivot_table(values='unemployment_rate', index='state', columns='date')
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_table, cmap="YlOrRd", cbar_kws={'label': 'Unemployment Rate (%)'})
plt.title("State-wise Unemployment Rate Heatmap")
plt.xlabel("Date")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("Task-3-Unemployment-Analysis/results/statewise_heatmap.png")
plt.close()
print("✅ Saved state-wise heatmap.")

# Step 6: Bar Plot - Latest Statewise Unemployment
latest_date = df['date'].max()
latest_df = df[df['date'] == latest_date].sort_values(by='unemployment_rate', ascending=False)
plt.figure(figsize=(12, 7))
sns.barplot(x='unemployment_rate', y='state', data=latest_df, palette='magma', hue=None, legend=False)
plt.title(f"State-wise Unemployment Rate ({latest_date.date()})")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("Task-3-Unemployment-Analysis/results/statewise_barplot_latest.png")
plt.close()
print("✅ Saved barplot for latest unemployment rates.")

# Step 7: Urban vs Rural Unemployment
plt.figure(figsize=(8, 6))
sns.boxplot(x='area', y='unemployment_rate', data=df, palette='Set2', hue=None, legend=False)
plt.title("Unemployment Rate by Area (Urban vs Rural)")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("Task-3-Unemployment-Analysis/results/area_comparison_boxplot.png")
plt.close()
print("✅ Saved Urban vs Rural unemployment boxplot.")

# Step 8: Labour Participation Trend
plt.figure(figsize=(12, 6))
labour_avg = df.groupby('date')['labour_participation_rate'].mean()
sns.lineplot(x=labour_avg.index, y=labour_avg.values, color='green')
plt.title("National Labour Participation Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Labour Participation Rate (%)")
plt.tight_layout()
plt.savefig("Task-3-Unemployment-Analysis/results/labour_participation_trend.png")
plt.close()
print("✅ Saved labour participation rate trend.")

# Step 9: Top 10 Unemployment States (Latest Date)
plt.figure(figsize=(10, 6))
top10 = latest_df.head(10)
sns.barplot(x='unemployment_rate', y='state', data=top10, palette='Reds', hue=None, legend=False)
plt.title("Top 10 States by Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("Task-3-Unemployment-Analysis/results/top10_states_barplot.png")
plt.close()
print("✅ Saved Top 10 states barplot.")

print("\n🎉 All visualizations saved in: Task-3-Unemployment-Analysis/results/")
