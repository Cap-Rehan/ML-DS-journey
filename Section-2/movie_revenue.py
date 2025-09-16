# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ml_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Linear Regression

# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
data = pd.read_csv('cost_revenue_clean.csv')
data

# %%
data.describe()

# %% [markdown]
# ### Splitting Features & Target

# %%
x = pd.DataFrame(data, columns=['production_budget_usd'])
y = pd.DataFrame(data, columns=['worldwide_gross_usd'])

# %% [markdown]
# ### Plotting with matplotlib

# %%
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, marker='*')
plt.title('Production Budget vs Worldwide Gross')
plt.xlabel('Production Budget (USD)')
plt.ylabel('Worldwide Gross (USD)')
plt.xlim(0, 450000000)
plt.ylim(0, 3000000000)
plt.show()

# %% [markdown]
# ## Applying Linear Regression with Scikit-Learn

# %%
linreg= LinearRegression() # Create a linear regression object
linreg.fit(x, y) # Fit the model on the data

# %%
linreg.coef_ # Theta1 / Slope of the linear regression

# %%
linreg.intercept_ # Theta0 / Intercept of the linear regression

# %%
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, marker='*')
# preedicting the values for the regression line
plt.plot(x, linreg.predict(x), color='red', linewidth=2, label='Regression Line')
plt.legend(loc='upper left')

plt.title('Production Budget vs Worldwide Gross')
plt.xlabel('Production Budget (USD)')
plt.ylabel('Worldwide Gross (USD)')
plt.xlim(0, 450000000)
plt.ylim(0, 3000000000)
plt.show()

# %% [markdown]
# ### Model Evaluation

# %%
# Goodness of fit, R^2
# equals 55 means that 55% of the variance in worldwide gross can be
# explained by the production budget
linreg.score(x, y)
