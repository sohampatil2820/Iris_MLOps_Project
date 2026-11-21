import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add species names
df["species"] = [iris.target_names[i] for i in iris.target]

# Save dataset
df.to_csv("data/iris.csv", index=False)

print("Iris dataset created successfully!")
print("Dataset shape:", df.shape)
print(df.head())