import pickle
import numpy as np
from fastapi import FastAPI

app = FastAPI(title="Iris Flower Classification API")

# Load trained model
with open("models/iris_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.get("/")
def home():
    return {"message": "Iris Flower Classification API is running"}


@app.post("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(data)

    return {"predicted_species": prediction[0]}