# -------- Programming Assignment --------
#
# * Author: Laxman Desai ($190020066$)
# * Course: CL $202$ (Data Analysis)
# * Date: $20^{th}$ April $2021$

# Initialization
from scipy import stats
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('dataset/data.csv')
df['Z1'] *= 100 # Scaling
df['Z2'] *= 1000 # Scaling
df['Y'] *= 1000 # Scaling


# A) Correlation Coefficient
corr_coeff_Y_Z1 = df['Y'].corr(df['Z1'])
print(f'corr_coeff_Y_Z1 = {round(corr_coeff_Y_Z1, 4)}')
corr_coeff_Y_Z2 = df['Y'].corr(df['Z2'])
print(f'corr_coeff_Y_Z2 = {round(corr_coeff_Y_Z2, 4)}')


# B) Linear Model
X = df.drop('Y', axis=1) # Independant Variables
y = df['Y'] # Dependant Variables
N = len(X)
p = len(X.columns) + 1  # '+1' because LinearRegression adds an intercept term

model = LinearRegression()
model.fit(X, y)
a, b = model.coef_
c = model.intercept_
print(f'a = {round(a, 5)}, b = {round(b, 5)}, c = {round(c, 5)}')


# C) $95\%$ Confidence Interval
y_hat = model.predict(X)
err = y - y_hat # residual

RSS = err.T @ err # dot product, equivalent to sum(err**2)
S = np.sqrt(RSS / (N - p))

# M = matrix (20 x 3) with columns as [Z1, Z2, 1]
M = np.ones(shape=(N, p), dtype=float)
M[:, 0] = X.iloc[:, 0]
M[:, 1] = X.iloc[:, 1]

var_beta_hat = 1/(M.T @ M) * S**2 # Varience of beta hat

std_err_a = var_beta_hat[0, 0] ** 0.5
std_err_b = var_beta_hat[1, 1] ** 0.5
std_err_c = var_beta_hat[2, 2] ** 0.5

t = abs(stats.t.ppf((1-0.95)/2, N-p)) # 95% CI

# 95% CI Lower Bounds
ci_l_a = a - t * std_err_a
ci_l_b = b - t * std_err_b
ci_l_c = c - t * std_err_c
# 95% CI Upper Bounds
ci_u_a = a + t * std_err_a
ci_u_b = b + t * std_err_b
ci_u_c = c + t * std_err_c

print(f'a ∈ [{round(ci_l_a, 6)}, {round(ci_u_a, 6)}]' )
print(f'b ∈ [{round(ci_l_b, 6)}, {round(ci_u_b, 6)}]' )
print(f'c ∈ [{round(ci_l_c, 6)}, {round(ci_u_c, 6)}]' )


# D) $95\%$ Prediction Interval
# - Apartment size = 12 \* 100 ft2 and assessed value of 60 * 1000$
X_test = [[1200, 60000]]
y_test_hat = model.predict(X_test)[0]

pi_l_Y = y_test_hat - t * S
pi_u_Y = y_test_hat + t * S
print(f'lower value bound = {pi_l_Y}, upper value bound = {pi_u_Y}')


# E) Mean & varience of residuals
mean_resid = err.mean()
print(f'mean of residual = {mean_resid}')
var_resid = err.var()
print(f'varience of residual = {var_resid}')


# F) $R^2$ of the fit
y_diff = y - y.mean()
TSS = y_diff.T @ y_diff
Rsq = 1 - RSS/TSS
print(f'R^2 of the fit (calculated) = {Rsq}')
