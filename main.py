import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("auto-mpg[1].csv")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower", "mpg"])

df = df.head(100)

X = df["horsepower"].astype(float).values
Y = df["mpg"].astype(float).values

# Normalize X
X_norm = (X - X.mean()) / X.std()

# Learning params
learning_rate = 0.1
iterations = 5000

# Initialize weights
m = 0   # slope
b = 0   # intercept

# Gradient Descent for Linear Regression
for i in range(iterations):
    y_pred = m * X_norm + b
    error = y_pred - Y
    
    dm = (2/len(X_norm)) * np.dot(error, X_norm)
    db = (2/len(X_norm)) * np.sum(error)
    
    m -= learning_rate * dm
    b -= learning_rate * db

print("Weight (m):", m)
print("Bias (b):", b)

# Plot
plt.scatter(X, Y, label="Data Points")

# Regression line
x_line = np.linspace(min(X), max(X), 200)
x_line_norm = (x_line - X.mean()) / X.std()
y_line = m * x_line_norm + b

plt.plot(x_line, y_line, linewidth=3, label="Linear Regression Line")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Manual Linear Regression")
plt.grid(True)
plt.legend()
plt.show()