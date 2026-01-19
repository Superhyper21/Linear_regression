import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("auto-mpg[1].csv")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower", "mpg"])

df = df.head(100)

X = df["horsepower"].astype(float).values
Y_cont = df["mpg"].astype(float).values

median_mpg = np.median(Y_cont)
Y = np.where(Y_cont >= median_mpg, 1, 0)

X_norm = (X - X.mean()) / X.std()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

learning_rate = 0.1
iterations = 5000

b = 0
m = 0

for i in range(iterations):
    z = m * X_norm + b
    predictions = sigmoid(z)
    dm = np.dot((predictions - Y), X_norm) / len(X_norm)
    db = np.sum(predictions - Y) / len(X_norm)
    m -= learning_rate * dm
    b -= learning_rate * db

print("Weight (m):", m)
print("Bias (b):", b)

plt.scatter(X, Y)

x_curve = np.linspace(min(X), max(X), 200)
x_curve_norm = (x_curve - X.mean()) / X.std()
y_curve = sigmoid(m * x_curve_norm + b)

plt.plot(x_curve, y_curve, linewidth=3)
plt.xlabel("Horsepower")
plt.ylabel("Probability of High MPG")
plt.title("Manual Logistic Regression")
plt.grid(True)
plt.show()