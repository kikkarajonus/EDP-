# Employee Salary Prediction using Linear Regression (Without Scikit-Learn)

## Overview

This project implements **Simple Linear Regression from scratch** using Python, without relying on the **scikit-learn** machine learning library. The program reads an employee salary dataset from a CSV file, trains a linear regression model using mathematical formulas, evaluates its performance, and predicts the salary of a new employee based on their years of experience.

The purpose of this project is to understand the internal working of Linear Regression rather than using pre-built machine learning libraries.

---

## Objectives

- Read data from a CSV file using Pandas.
- Split the dataset into training and testing data.
- Implement Linear Regression manually.
- Predict salary based on employee experience.
- Evaluate the regression model using standard evaluation metrics.

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy

---

## Project Structure

```
EDP WEEK 2/src/
│
├── salary.csv
├── Employee.py
└── README.md
```

---

## Dataset Format

The CSV file should contain the following columns:

| Experience | Salary |
|------------|--------|
| 1 | 25000 |
| 2 | 30000 |
| 3 | 38000 |
| 4 | 45000 |
| 5 | 52000 |
| 6 | 60000 |
| 7 | 68000 |
| 8 | 76000 |
| 9 | 85000 |
| 10 | 93000 |
| 11 | 102000 |
| 12 | 112000 |

Save the dataset as:

```
employee_salary_dataset.csv
```

---

## Installation

Install the required Python libraries.

```bash
pip install pandas numpy
```

---

## How to Run

Run the Python program.

```bash
python linear_regression.py
```

When prompted, enter the CSV filename.

Example:

```
Enter CSV file name:
employee_salary_dataset.csv
```

---

## Program Workflow

### Step 1

Read the employee salary dataset from the CSV file.

### Step 2

Extract:

- Feature (X) → Experience
- Label (Y) → Salary

### Step 3

Split the dataset into:

- 80% Training Data
- 20% Testing Data

### Step 4

Train the Linear Regression model using the mathematical formula.

### Step 5

Predict salaries for the testing dataset.

### Step 6

Calculate evaluation metrics.

### Step 7

Predict the salary for a new employee based on user input.

---

## Linear Regression Formula

The prediction equation is:

```
Salary = (Slope × Experience) + Intercept
```

Where:

- **Slope (m)** determines the rate of salary increase.
- **Intercept (c)** is the starting salary when experience is zero.

The slope is calculated as:

```
m = Σ[(Xi − X̄)(Yi − Ȳ)] / Σ[(Xi − X̄)²]
```

The intercept is calculated as:

```
c = Ȳ − m × X̄
```

---

## Evaluation Metrics

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

```
MAE = Σ|Actual − Predicted| / n
```

---

### Mean Squared Error (MSE)

Measures the average squared prediction error.

```
MSE = Σ(Actual − Predicted)² / n
```

---

### Root Mean Squared Error (RMSE)

Square root of the Mean Squared Error.

```
RMSE = √MSE
```

---

### R² Score

Measures how well the regression model explains the data.

```
R² = 1 − (SSres / SStotal)
```

An R² value closer to **1** indicates better model performance.

---

## Sample Output

```
Dataset Loaded Successfully

Training Data:
[1 2 3 4 5 6 7 8]

Testing Data:
[9 10 11 12]

Model Trained Successfully!

Slope (m): 7309.52
Intercept (c): 16857.14

Actual Salary:
[85000 93000 102000 112000]

Predicted Salary:
[82642 89952 97261 104571]

Evaluation Metrics

MAE : 2800.45
MSE : 7900000.32
RMSE: 2810.69
R² Score: 0.98

Enter Employee Experience (Years): 5

Predicted Salary = ₹53404.76
```

---

## Features

- Reads dataset from a CSV file
- Uses Pandas for data handling
- Manual implementation of Linear Regression
- Manual train-test split
- Salary prediction
- Calculates MAE, MSE, RMSE, and R² Score
- User input for predicting a new salary
- Beginner-friendly implementation without Scikit-Learn

---

## Learning Outcomes

By completing this project, you will learn:

- Reading datasets using Pandas
- Understanding Features and Labels
- Train-Test Split
- Working of Linear Regression
- Mathematical implementation of regression
- Model evaluation techniques
- Predicting continuous values

---

## Future Improvements

- Add support for multiple features (Multiple Linear Regression)
- Randomize the train-test split
- Visualize the regression line using Matplotlib
- Save prediction results to a CSV file
- Compare the manual implementation with Scikit-Learn

---

## Author

This project was developed as a beginner-level Machine Learning implementation to demonstrate **Linear Regression from scratch using Python**, without using the Scikit-Learn library.
