import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('iris.data', header=None)

# Define feature names and target names
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_names = np.unique(data.iloc[:, -1].values)

# Split the data into features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

# Define the range of depths to evaluate
depth_vec = np.arange(1, 111, 2)
PK_1D_depth = np.zeros(len(depth_vec))

# Evaluate the Decision Tree Classifier for each depth
for depth_ind, depth in enumerate(depth_vec):
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    PK = accuracy_score(y_test, y_pred) * 100

    print(f"Depth {depth} | PK {PK:.2f}")
    PK_1D_depth[depth_ind] = PK

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(depth_vec, PK_1D_depth, marker="o")
plt.xlabel("Depth")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Depth for Decision Tree on Iris Dataset")
plt.grid(True)

# Save the plot
plt.savefig("PK_vs_Depth_DecisionTree_Iris.png", bbox_inches="tight")
plt.show()
