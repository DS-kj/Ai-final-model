import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
# 7 User Input for New Student
# -----------------------------
print("\nEnter the following details for the new student:")

def get_input(prompt, dtype=float):
    while True:
        val = input(f"{prompt}: ")
        if val == "":
            print("This value is required, please enter a number.")
            continue
        try:
            return dtype(val)
        except:
            print("Invalid input, please enter a number.")

# Always ask these
study_duration = get_input("Study duration in minutes", float)
repetitions = get_input("Number of repetitions", int)
time_since_revision = get_input("Time since last revision in hours (0 if first time studying)", float)
focus_level = get_input("Focus level (0-1)", float)
sleep_quality = get_input("Sleep quality (0-1)", float)
days_since_first_learning = get_input("Days since first learning (0 if first time studying)", int)

# Determine if first-ever study
first_time_student = (time_since_revision == 0 and days_since_first_learning == 0)

if first_time_student:
    previous_retention = 90  # first-ever study, use 90% as previous retention
    spacing_interval = 0      # no spacing interval yet
else:
    spacing_interval = get_input("Spacing interval in hours between revisions", float)
    previous_retention = get_input("Previous retention (%)", float)

# Time of day input
print("\nSelect time of day:")
print("0 = Morning, 1 = Afternoon, 2 = Night")
time_of_day_val = get_input("Time of day (0,1,2)", int)

# Prepare one-hot columns
time_of_day_1 = 1 if time_of_day_val == 1 else 0
time_of_day_2 = 1 if time_of_day_val == 2 else 0

# Create dataframe for model
new_student = pd.DataFrame({
    "study_duration": [study_duration],
    "repetitions": [repetitions],
    "time_since_revision": [time_since_revision],
    "spacing_interval": [spacing_interval],
    "focus_level": [focus_level],
    "sleep_quality": [sleep_quality],
    "days_since_first_learning": [days_since_first_learning],
    "previous_retention": [previous_retention],
    "time_of_day_1": [time_of_day_1],
    "time_of_day_2": [time_of_day_2]
})

# Scale numerical features
new_student[numerical_cols] = scaler.transform(new_student[numerical_cols])

# -----------------------------
# 8 Predicted Retention
# -----------------------------
predicted_retention = model.predict(new_student)[0]  # always calculated by model
print(f"\nPredicted retention for new student: {predicted_retention:.2f}%")

# -----------------------------
# 9 Schedule Revision (Threshold = 60%)
# -----------------------------
threshold = 60
t = 0
current_retention = predicted_retention
revision_times = []

spacing_interval_hours = new_student["spacing_interval"].iloc[0]
spacing_interval_hours = spacing_interval_hours if spacing_interval_hours > 0 else 1

# Simulate memory decay over next 7 days (168 hours)
while t <= 168:
    hours_since_last_revision = t - revision_times[-1] if revision_times else t
    retention = current_retention * np.exp(-hours_since_last_revision / 72)

    if retention < threshold:
        revision_times.append(t)
        current_retention = predicted_retention

    t += spacing_interval_hours

# -----------------------------
# 10 Output Revision Info
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

# -----------------------------
# 11 Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
