# Task 1: Iris Flower Classification (Internship)
# Author: Ananya A.
# Description: Loads dataset, visualizes features, trains and evaluates a model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load the dataset (full path added here)
df = pd.read_csv(r"C:\Users\ananya\Desktop\iris_task\Iris.csv")

# STEP 2: Display first few rows
print("📄 First 5 rows of the dataset:")
print(df.head())

# STEP 3: Dataset info
print("\nℹ️ Dataset Info:")
print(df.info())

# STEP 4: Data Visualization
# Histogram for each feature
df.hist(figsize=(10, 8))
plt.suptitle("📊 Feature Distributions")
plt.show()

# Pairplot with species
sns.pairplot(df, hue="Species")
plt.suptitle("🔍 Pairplot of Iris Features", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("📌 Feature Correlation Heatmap")
plt.show()

# STEP 5: Data Preprocessing
# Drop 'Id' column
df = df.drop("Id", axis=1)

# Split into features (X) and target (y)
X = df.drop("Species", axis=1)
y = df["Species"]

# Encode labels (Species)
le = LabelEncoder()
y = le.fit_transform(y)

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Data Preprocessing Done!")
print("🔹 X_train shape:", X_train.shape)
print("🔹 X_test shape:", X_test.shape)

# STEP 6: Model Training & Evaluation
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict test results
y_pred = model.predict(X_test)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Training Complete!")
print("🎯 Accuracy on test data:", accuracy)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
