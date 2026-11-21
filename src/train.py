import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/iris.csv")

# Separate features and target
X = df.drop("species", axis=1)
y = df["species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("Iris_Flower_Classification")

with mlflow.start_run():

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print("Model trained successfully!")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Log parameters and metrics in MLflow
    mlflow.log_param("algorithm", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
    model,
    name="iris_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    # Save model
    os.makedirs("models", exist_ok=True)

    with open("models/iris_model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Model saved successfully!")
    print("MLflow tracking completed!")