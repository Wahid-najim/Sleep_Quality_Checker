# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Load and inspect the data
df = pd.read_csv('data.csv')

# Display initial information
print(df.head())
print(df.isnull().sum())
print(df.info())
print(df.describe())

# Exploratory Data Analysis (EDA)
le = LabelEncoder()
df['SleepQuality'] = le.fit_transform(df['SleepQuality'])

# Split the data into features (X) and target (y)
X = df[['HeartRate', 'SleepDuration', 'PhysicalActivity']]
y = df['SleepQuality']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Visualize the data
sns.pairplot(df, hue='SleepQuality')
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Define the user input function for prediction
def predict_sleep_quality():
    try:
        heart_rate = int(heart_rate_entry.get())
        sleep_duration = float(sleep_duration_entry.get())
        physical_activity = int(physical_activity_entry.get())

        # Check if inputs are within a reasonable range
        if heart_rate < 0 or sleep_duration < 0 or physical_activity < 0:
            messagebox.showerror("Invalid Input", "All inputs must be positive numbers.")
            return

        # Predict sleep quality based on the input values
        prediction = model.predict([[heart_rate, sleep_duration, physical_activity]])
        sleep_quality = le.inverse_transform(prediction)[0]

        # Show the result
        messagebox.showinfo("Sleep Quality Prediction", f"Predicted Sleep Quality: {sleep_quality}")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers.")

# Create the main window
root = tk.Tk()
root.title("Sleep Quality Predictor")

# Create input labels and entry fields
tk.Label(root, text="Heart Rate:").grid(row=0, column=0)
heart_rate_entry = tk.Entry(root)
heart_rate_entry.grid(row=0, column=1)

tk.Label(root, text="Sleep Duration (hours):").grid(row=1, column=0)
sleep_duration_entry = tk.Entry(root)
sleep_duration_entry.grid(row=1, column=1)

tk.Label(root, text="Physical Activity (steps):").grid(row=2, column=0)
physical_activity_entry = tk.Entry(root)
physical_activity_entry.grid(row=2, column=1)

# Add predict button
predict_button = tk.Button(root, text="Predict Sleep Quality", command=predict_sleep_quality)
predict_button.grid(row=3, columnspan=2)

# Run the main loop
root.mainloop()
