import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score

data_train = pd.read_csv('xy_train.csv', header=None, names=['Feature 1', 'Feature 2', 'Label'])
data_test = pd.read_csv('xy_test.csv', header=None, names=['Feature 1', 'Feature 2', 'Label'])

X = np.array(data_train[['Feature 1', 'Feature 2']])
y = np.array(data_train['Label'])

X_test = np.array(data_test[['Feature 1', 'Feature 2']])
y_test = np.array(data_test['Label'])

n, p = X.shape

C = 1.0
beta = cp.Variable(p)
beta0 = cp.Variable()
epsilon = cp.Variable(n)

constraints = [
    epsilon >= 0,
    cp.multiply(y, X @ beta + beta0) >= 1 - epsilon
]
obj = cp.Minimize(1/2 * cp.norm(beta) ** 2 + C * cp.sum(epsilon))
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.ECOS)

criv = problem.value

print("Optimal Criterion Value:", criv)
print("Optimal Coefficients (beta):", beta.value)
print("Optimal Intercept (beta0):", beta0.value)

a_values = [np.arange(-5, 6), []]
misclassification_errors = []

t_values = np.arange(-5, 6)
C_values = 2.0 ** t_values

for i in range(-5, 6):
    C = 2**i
    obj = cp.Minimize(1/2 * cp.norm(beta) ** 2 + C * cp.sum(epsilon))
    constraints = [epsilon >= 0,
                   cp.multiply(y, (X @ beta + beta0)) >= 1 - epsilon]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.ECOS)
    y_i = beta0.value + X_test @ np.array(beta.value).T >= 0
    y_i = y_i * 2 - 1
    a_values[1].append(sum(y_i != y_test))
    misclassification_error = 1 - accuracy_score(y_test, y_i)
    misclassification_errors.append(misclassification_error)

plt.semilogx(C_values, misclassification_errors, marker='o')
plt.grid(True)
plt.show()