# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Generate Synthetic Data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + np.random.randn(100) * 2 + 5

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pred_linear = linear.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)

# Evaluation
print("Linear R²:", r2_score(y_test, y_pred_linear))
print("Ridge R²:", r2_score(y_test, y_pred_ridge))
print("Polynomial R²:", r2_score(y_test, y_pred_poly))

# Visualization
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='gray', label='Actual Data')
plt.plot(X, linear.predict(X), color='blue', label='Linear Regression')
plt.plot(X, ridge.predict(X), color='green', linestyle='--', label='Ridge Regression')
plt.plot(X, poly_model.predict(poly.transform(X)), color='red', linestyle='-.', label='Polynomial Regression')
plt.xlabel('House Size (1000 sq ft)')
plt.ylabel('Price (Lakh ₹)')
plt.title('House Price Prediction using Regression Models')
plt.legend()
plt.grid(True)
plt.show()
