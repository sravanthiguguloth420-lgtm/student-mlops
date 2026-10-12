import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load student data
df = pd.read_csv("data/student_data.csv")

print("Student Dataset:")
print(df.head())

# Step 2: Separate inputs and output
X = df[["study_hours", "attendance"]]
y = df["result"]

# Step 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Step 4: Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

mlflow.set_experiment("Student-Performance")

with mlflow.start_run():

    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Step 6: Save the trained model
    joblib.dump(model, "student_model.pkl")

    # Step 7: Track the experiment in MLflow
    mlflow.log_param("algorithm", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    print("\nModel saved as student_model.pkl")
    print("Experiment logged to MLflow")
