import pandas as pd
import numpy as np
filename = input("Enter CSV file name: ")

df = pd.read_csv(salary.csv)

print("\nDataset Loaded Successfully")
print(df)

X = np.array(df["Experience"])
Y = np.array(df["Salary"])

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]

print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

mean_x = np.mean(X_train)
mean_y = np.mean(Y_train)

numerator = np.sum((X_train - mean_x) * (Y_train - mean_y))
denominator = np.sum((X_train - mean_x) ** 2)

m = numerator / denominator
c = mean_y - (m * mean_x)

print("\nModel Trained Successfully!")
print("Slope (m):", round(m,2))
print("Intercept (c):", round(c,2))
Y_pred = m * X_test + c

print("\nActual Salary")
print(Y_test)

print("\nPredicted Salary")
print(Y_pred.astype(int))
mae = np.mean(np.abs(Y_test - Y_pred))

mse = np.mean((Y_test - Y_pred) ** 2)

rmse = np.sqrt(mse)

ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
ss_res = np.sum((Y_test - Y_pred) ** 2)

r2 = 1 - (ss_res / ss_total)

print("\nEvaluation Metrics")
print("-------------------------")
print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R² Score:", round(r2,4))

experience = float(input("\nEnter Employee Experience (Years): "))

predicted_salary = (m * experience) + c

print("\nPredicted Salary = ₹", round(predicted_salary,2))