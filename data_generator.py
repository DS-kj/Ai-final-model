import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
np.random.seed(42)      # reproducibility
n_samples = 20     # number of study sessions to simulate

# -----------------------------
# Feature Generation
# -----------------------------

# 1. Study duration (minutes) – normal distribution, realistic study times
study_duration = np.clip(np.random.normal(loc=60, scale=20, size=n_samples), 10, 120)

# 2. Number of repetitions – discrete 1–6, realistic probabilities
repetitions = np.random.choice([1,2,3,4,5,6], size=n_samples, p=[0.2,0.25,0.2,0.15,0.12,0.08])

# 3. Time since last revision (hours) – exponential decay to mimic forgetting
time_since_revision = np.clip(np.random.exponential(scale=24, size=n_samples), 1, 168)

# 4. Spacing interval (hours) – normal distribution, realistic spacing
spacing_interval = np.clip(np.random.normal(loc=18, scale=10, size=n_samples), 1, 72)

# 5. Focus level (0–1) – beta distribution for realistic concentration levels
focus_level = np.clip(np.random.beta(a=5, b=2, size=n_samples), 0.2, 1.0)

# 6. Sleep quality (0–1) – normal distribution, reasonable mean
sleep_quality = np.clip(np.random.normal(loc=0.7, scale=0.15, size=n_samples), 0.3, 1.0)

# 7. Time of day studied – 0=morning, 1=afternoon, 2=night
time_of_day = np.random.choice([0,1,2], size=n_samples, p=[0.35,0.4,0.25])

# 8. Days since first learning – gamma distribution to simulate learning history
days_since_first_learning = np.clip(np.random.gamma(shape=2, scale=10, size=n_samples), 1, 180)

# 9. Previous retention score (%) – realistic memory history
previous_retention = np.clip(np.random.normal(loc=0.65, scale=0.15, size=n_samples), 0.2, 0.95)

# -----------------------------
# Target Variable: Retention Score
# -----------------------------
# Combines cognitive principles: forgetting curve, spacing, repetition, focus, sleep
forgetting_factor = np.exp(-time_since_revision / 72)
spacing_bonus = np.log1p(spacing_interval) / 5
repetition_boost = np.log1p(repetitions) / 3
effort_factor = (study_duration / 120) * 0.4 + focus_level * 0.6
sleep_factor = sleep_quality * 0.3 + 0.7
time_of_day_factor = np.where(time_of_day==0, 1.05, np.where(time_of_day==1, 1.0, 0.95))

retention_score = (
    previous_retention * 0.3 +
    forgetting_factor * 0.25 +
    spacing_bonus * 0.1 +
    repetition_boost * 0.15 +
    effort_factor * 0.1 +
    sleep_factor * 0.07
)
retention_score *= time_of_day_factor
retention_score = np.clip(retention_score, 0, 1) * 100  # convert to %

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "study_duration": study_duration.round(1),
    "repetitions": repetitions,
    "time_since_revision": time_since_revision.round(1),
    "spacing_interval": spacing_interval.round(1),
    "focus_level": focus_level.round(2),
    "sleep_quality": sleep_quality.round(2),
    "time_of_day": time_of_day,
    "days_since_first_learning": days_since_first_learning.round(1),
    "previous_retention": (previous_retention * 100).round(1),
    "retention_score": retention_score.round(1)
})

# -----------------------------
# Save to CSV
# -----------------------------
output_file = r"C:\Users\ASUS\Desktop\4 ai\memory_dataset.csv"
df.to_csv(output_file, index=False)
print(f"Dataset saved as {output_file}")
print(df.head())


# -----------------------------
# Data Validation
# -----------------------------
print("\n--- Dataset Summary ---")
print(df.describe())

print("\n--- Feature Types ---")
print(df.dtypes)

print("\n--- Time of Day Distribution ---")
print(df['time_of_day'].value_counts())

# Histograms for numerical features
numeric_features = ["study_duration", "repetitions", "time_since_revision", 
                    "spacing_interval", "focus_level", "sleep_quality",
                    "days_since_first_learning", "previous_retention", "retention_score"]

for col in numeric_features:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Scatter plots to see correlation with retention_score
for col in numeric_features[:-1]:  # exclude retention_score itself
    sns.scatterplot(x=col, y='retention_score', data=df)
    plt.title(f'{col} vs Retention Score')
    plt.show()