import pickle
import numpy as np

def test_model_prediction():
    with open("models/iris_model.pkl", "rb") as file:
        model = pickle.load(file)

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])

    prediction = model.predict(sample)

    assert prediction[0] == "setosa"