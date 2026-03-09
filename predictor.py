import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1 Load Dataset
# -----------------------------
file_path = r"C:\Users\ASUS\Desktop\4 ai\memory_dataset.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2 Features and Target
# -----------------------------
X = df.drop("retention_score", axis=1)
y = df["retention_score"]

# -----------------------------
# 3 One-hot encode categorical features
# -----------------------------
X = pd.get_dummies(X, columns=["time_of_day"], drop_first=True)

# -----------------------------
# 4 Normalize numerical features
# -----------------------------
scaler = MinMaxScaler()
numerical_cols = ["study_duration", "time_since_revision", "spacing_interval", 
                  "focus_level", "sleep_quality", "days_since_first_learning", "previous_retention"]

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# -----------------------------
# 5 Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6 Train Model (Linear Regression)
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 7 Predictions
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# -----------------------------
# 9 Example Prediction for a New Student
# -----------------------------
new_student = pd.DataFrame({
    "study_duration": [60],
    "repetitions": [3],
    "time_since_revision": [48],
    "spacing_interval": [18],
    "focus_level": [0.8],
    "sleep_quality": [0.7],
    "days_since_first_learning": [20],
    "previous_retention": [70],
    "time_of_day_1": [0],  # afternoon
    "time_of_day_2": [0]   # night
})

# Scale numerical features
new_student[numerical_cols] = scaler.transform(new_student[numerical_cols])

# -----------------------------
# Predicted Retention
# -----------------------------
predicted_retention = model.predict(new_student)[0]
print(f"\nPredicted retention for new student: {predicted_retention:.2f}%")

# -----------------------------
# Schedule Revision (Threshold = 60%)
# -----------------------------

threshold = 60
t = 0
current_retention = predicted_retention
revision_times = []

spacing_interval_hours = new_student["spacing_interval"].iloc[0]

# Simulate memory decay over next 7 days (168 hours)
while t <= 168:
    hours_since_last_revision = t - revision_times[-1] if revision_times else t
    retention = current_retention * np.exp(-hours_since_last_revision / 72)  # forgetting curve

    if retention < threshold:
        revision_times.append(t)
        current_retention = predicted_retention  # reset retention after revision

    t += spacing_interval_hours if spacing_interval_hours > 0 else 1

# -----------------------------
# Output Revision Info
# -----------------------------
if revision_times:
    first_rev = revision_times[0]
    days = first_rev // 24
    hours = first_rev % 24
    if first_rev == 0:
        print("\nRevision is immediately recommended.")
    else:
        print(f"\nOptimal time for revision is after {days} day(s) and {hours} hour(s).")
else:
    print("\nNo revision needed in the next 7 days.")
    if revision_times:
        first_rev = revision_times[0]
        if first_rev == 0:
            print("\nRevision is immediately needed!")
        else:
            days = first_rev // 24
            hours = first_rev % 24
            print(f"\nOptimal time for revision is after {days} day(s) and {hours} hour(s).")
    else:
        print("\nNo revision needed in the next 7 days.")

        from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root of MSE
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# R² Score (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([0, 100], [0, 100], 'r--')  # Diagonal line y=x
plt.xlabel("Actual Retention (%)")
plt.ylabel("Predicted Retention (%)")
plt.title("Predicted vs Actual Retention")
plt.grid(True)
plt.show()
import pandas as pd

# Get coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

# Plot
plt.figure(figsize=(10,6))
plt.barh(coef_df["Feature"], coef_df["Coefficient"], color='teal')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Linear Regression Feature Coefficients")
plt.grid(True)
plt.show()