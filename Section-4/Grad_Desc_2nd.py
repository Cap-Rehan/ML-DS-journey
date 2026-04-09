
# %% [markdown]
# ## All import statements

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# importing color map module:
from matplotlib import cm

# %% [markdown]
# # MSE: Mean Squared Error
# moving from toy cost functions to a real regression cost: the Residual Sum of Squares (RSS) and its normalized form Mean Squared Error (MSE).

# %% [markdown]
# ### RSS (intuition): sum of squared residuals:
# ### $$ RSS = \sum_{i=1}^{n} \big( y^{(i)} - h_\theta (x^{(i)}) \big)^2 $$
#
# Measures how much of the dependent variable’s variation the model does not explain.
# RSS = 0 ⇒ perfect fit.

# %% [markdown]
# ### MSE (why divide by n?):
# ### $$ MSE = \frac{1}{n} \sum_{i=1}^{n} \big( y^{(i)} - h_\theta (x^{(i)}) \big)^2 $$
# MSE = average of the squared residuals; more numerically stable for large datasets.
# Dividing by n keeps the magnitude of the cost manageable and helps avoid floating-point overflow when n is large.

# %% [markdown]
# Notation: h_\theta(x) (hypothesis) and \hat{y} (y-hat) are equivalent ways to denote predicted values. The course uses ŷ (y-hat) in code.
#
# ### $$ MSE = \frac{1}{n} \sum_{i=1}^{n} \big( y - \hat{y}) \big)^2 $$

# %% [markdown]
# - **LaTeX** tips used in notebook:
#
# \$$ ... \$$ for centered display math.
#
# \sum_{i=1}^{n} for summation with limits.
#
# \frac{1}{n} for fractions.
#
# \hat{y} for y-hat.
#
# \big( ... \big)^2 or \left( ... \right)^2 for large parentheses.

# %%
# Practical impl. (Python):

# RSS (raw sum):
# rss = np.sum((y_true - y_pred)**2)
# MSE (recommended):
# mse = np.mean((y_true - y_pred)**2)
# or
# mse = np.sum((y_true - y_pred)**2) / len(y_true)

# %% [markdown]
# Use MSE in gradient descent to compute cost and gradients for linear regression.
#
# Takeaway: use MSE as your cost when working with real datasets — it’s conceptually the same as RSS but safer numerically and comparable across dataset sizes.

# %% [markdown]
# **KEY TERMS / Questions**
#
# RSS — total squared error (not averaged).
#
# MSE - RSS divided by n — average squared error; preferred cost for regression.
#
# ŷ (y-hat): symbol for predicted value (same as h_θ​(x)).
#
# Overflow error: when computed numbers exceed floating-point limits; dividing by n reduces risk.
#
# Practical question: when you implement gradient descent with MSE, are you using the vectorized gradient formula (fast) or computing gradients pointwise (slow)? (Vectorized is recommended.)
#
# **Summary**
#
# RSS measures total squared error; MSE is the average squared error (RSS / n). MSE is preferred for real datasets because averaging keeps cost magnitudes reasonable and helps avoid overflow when n is large. In notebooks, use LaTeX (\$$...\$$) to display RSS/MSE nicely. Implement MSE in Python with np.mean((y_true - y_pred)**2) and use it as the cost function in gradient descent.

# %% [markdown]
# ## Work begins here:

# %%
# creating sample data:
x5= np.array([0.2, 1.6, 2.7, 4.0, 6.6, 7.5, 9.9])
y5= np.array([2.4, 1.9, 3.2, 3.0, 5.1, 6.7, 6.5])

print("Shape of x5:", x5.shape)
print("Shape of y5:", y5.shape)

# %% [markdown]
# - Both arrays are 1-D vectors with 7 samples.
#
# - scikit-learn LinearRegression needs X as 2-D
#
# - X must be shaped (n_samples, n_features) even if n_features == 1.
#
# Two ways to convert a 1-D x_5 into the required 2-D X:

# %%
x5 = np.array([x5]).T       # double brackets + .T  => shape (7,1)
# or
# x5 = x5.reshape(7, 1)       # reshape(n_samples, n_features)

# %%
x5.shape

# %% [markdown]
# (Note: many sklearn estimators accept y as a 1-D array (n_samples,); we can reshape y to (n_samples, 1) for consistency with how we access coef/intercept.)

# %%
y5 = np.array([y5]).T
y5.shape

# %%
# Quick Linear Regression fit:
regr = LinearRegression()
regr.fit(x5, y5)

# %%
theta_0= regr.intercept_[0]
theta_1= regr.coef_[0][0]
print("Theta_0 (intercept):", theta_0)
print("Theta_1 (slope):", theta_1)

# %% [markdown]
# ## Plotting time

# %%
plt.scatter(x5, y5, color='blue')
plt.plot(x5, regr.predict(x5), color= 'yellow')
plt.style.use('ggplot')
plt.show()

# %% [markdown]
# - Reminder: this plot is of data (x,y), not the cost surface over θ parameters.
#
# - Important conceptual distinction:
#
# Previous lessons plotted cost functions in parameter space (e.g., cost as a function of θ₀, θ₁).
#
# Here we plot data & fitted line (x → ŷ). The cost surface depends on θ values, not the x/y axes of this plot.

# %% [markdown]
# **KEY TERMS / Questions**
#
# - X shape requirement: X must be 2-D (n_samples, n_features). Use .reshape(n,1) or np.array([x]).T.
#
# - y shape: sklearn commonly accepts y as 1-D (n_samples,). Lesson used (n_samples, 1)—be consistent with how you index intercept_ / coef_.
#
# - regr.intercept_ / regr.coef_: intercept is often an array ([b]) and coef is an array (shape (1,1) for single feature) — index accordingly.
#
# - ŷ (y-hat): predicted values from the model regr.predict(X).
#
# - Question to check: do you pass y as 1-D or 2-D in your code? If 2-D, remember how you index intercept_ and coef_.
#
# **Summary**
#
# You generated a small sample dataset (x_5, y_5), learned that scikit-learn requires the feature matrix X to be 2-D, and saw two ways to convert a 1-D vector into (n_samples, 1) (.reshape or double brackets + .T). You fitted a LinearRegression model, extracted the intercept and slope (θ₀, θ₁), and plotted the data with the fitted line. Finally, you were reminded of the key conceptual difference: the regression plot shows data and predictions (x → ŷ), whereas earlier lessons were plotting cost surfaces in parameter (θ) space.

# %% [markdown]
# ## Implementing MSE concept

# %%
# theta0 and theta1 were extracted from the fitted model
y_hat = theta_0 + theta_1 * x5
print(y_hat.shape)
print("Estimated values of y (y_hat):\n", y_hat)
print("Actual values of y:\n", y5)
print("Differences between these:\n", abs(y5 - y_hat))


# %% [markdown]
# ## Equivalent ways to implement MSE

# %%
def mse_calc(y, y_hat):
    # return 1/y.size * sum((y - y_hat)**2)
    return np.mean((y - y_hat)**2) # or np.average() 


# %% [markdown]
# Check against scikit-learn built-in:

# %%
result= mse_calc(y5, y_hat)

print("MSE calculated manually:", result)
print("MSE calculated via sklearn with manually calculated y_hat values:\n",
      mean_squared_error(y5, y_hat))
print("MSE calculated via sklearn with regr.predict() values:\n",
      mean_squared_error(y5, regr.predict(x5)))

# %% [markdown]
# **Practical notes / gotchas**
#
# - Don’t hardcode 1/n — use y.size or len(y) so function works for any dataset.
#
# - regr.predict(...) expects the same shape as used to fit (i.e. X shaped (n_samples, n_features)). If you used X = x_5.reshape(-1,1) when fitting, call regr.predict(X) (or regr.predict(x_5.reshape(-1,1))).
#
# - np.average((y-y_hat)**2, axis=0) does both sum-of-squares and divide-by-N in one call; for 1D arrays axis=0 is appropriate.

# %% [markdown]
# **KEY TERMS / Questions**
#
# - ŷ (y-hat) — model predicted values (Ŷ = θ₀ + θ₁·x for linear regression).
#
# - MSE (Mean Squared Error) — average of squared residuals: (1/n) Σ (yᵢ − ŷᵢ)².
#
# - np.average vs sum + divide — both compute the mean; np.average is concise and works well with NumPy arrays.
#
# - Shape requirement for sklearn — X must be 2-D: (n_samples, n_features) even if n_features == 1.
#
# Questions to check yourself
#
# - Did you reshape x_5 as (n,1) when fitting, and are you calling regr.predict with the same shape?
#
# - Are you using .size (or len) instead of a hardcoded n in your mse function?
#
# - Do mse(y, y_hat) and mean_squared_error(y, y_hat) return the same numeric result?
#
# **Summary**
#
# You computed predicted values y_hat = θ₀ + θ₁·x from the fitted linear model, implemented a general mse(y, y_hat) function in three ways (pure Python sum, using y.size, and np.average), and validated the manual calculation against sklearn.metrics.mean_squared_error. Important practical points: always avoid hardcoding sample count, ensure X has the correct 2-D shape for sklearn calls, and use np.average for concise numeric code.

# %% [markdown]
# # 3D plot for the MSE cost function
# ### Make data for thetas

# %%
# Goal: visualize the MSE cost surface over θ₀ and θ₁ by computing MSE
# for many (θ₀, θ₁) combinations and plotting a surface.

nr_thetas = 100
th_0 = np.linspace(-1, 3, nr_thetas)   # θ₀ values
th_1 = np.linspace(-1, 3, nr_thetas)   # θ₁ values

# %%
# converting to 2D grid:
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)
# plot_t0.shape == plot_t1.shape == (nr_thetas, nr_thetas)

# %% [markdown]
# plot_t0[i,j] and plot_t1[i,j] are the θ₀ and θ₁ at grid cell (i,j).

# %%
plot_cost = np.zeros((nr_thetas, nr_thetas))

# %% [markdown]
# ## Calc MSE over the grid with nested-loops

# %%
for i in range(nr_thetas):
    for j in range(nr_thetas):
        theta0 = plot_t0[i, j]
        theta1 = plot_t1[i, j]
        y_hat = theta0 + theta1*x5      # predicted values for dataset x_5
        plot_cost[i, j] = mse_calc(y5, y_hat)      # store MSE at grid cell

# %% [markdown]
# This is the standard pattern: outer loop over rows (i), inner loop over columns (j).
#
# mse is your previously defined MSE function (use y.size inside it — do not hardcode N).

# %% [markdown]
# Plotting the surface
#
# - Use plot_t0, plot_t1, and plot_cost as the X, Y, Z inputs to the 3D surface/contour plot functions.
#
# - Ensure the arrays are the same shape: plot_t0.shape == plot_t1.shape == plot_cost.shape.
#
# - If you want a denser, smoother surface, increase nr_thetas (e.g., 50 or 100) — beware of compute time.
#
# A few practical notes
#
# - Keep nr_thetas in a variable (no hardcoding) so changing resolution updates everything automatically.
#
# - The nested loop approach is straightforward and easy to read. For large grids you can vectorize or use broadcasting, but nested loops are fine for teaching/medium sizes.
#
# - When plotting, check orientation: sometimes meshgrid order (np.meshgrid(x,y) vs np.meshgrid(y,x)) affects i,j indexing and which axis corresponds to θ₀ vs θ₁. Confirm by quick prints.
#
# - i = row index, j = column index. Order matters for traversal but either order visits every cell.

# %% [markdown]
# **KEY TERMS / Questions**
#
# - meshgrid — converts 1-D coordinate vectors into 2-D coordinate matrices for surface plotting.
#
# - np.zeros((r,c)) — creates a 2-D array initialized to zero with r rows and c columns.
#
# - Nested loop — loop inside a loop, used to visit every element of a 2-D array (for i in ...: for j in ...:).
#
# - f-string — Python string formatting f"{var}" allowing inline variable interpolation (very handy for debug prints).
#
# - Question: Why must plot_cost be 2-D (not 1-D)?
# Answer: Because a surface needs a Z value for each (X,Y) grid cell; you need a matrix of costs matching the mesh shape.
#
# - Question: If the plotted surface looks transposed/rotated, what to check?
# Answer: Confirm meshgrid order and which axis you used for i (rows) and j (columns); swap th_0/th_1 in meshgrid if needed.
#
# **Summary**
#
# Create ranges for θ₀ and θ₁ with linspace, turn those 1-D vectors into 2-D grids with np.meshgrid, and allocate a plot_cost matrix with np.zeros((nr_thetas, nr_thetas)). Use nested for i / for j loops to compute y_hat = θ₀ + θ₁ * x_5 for each grid cell, evaluate mse(y_5, y_hat), and store it in plot_cost[i,j]. The result is a 2-D MSE surface that you can pass to your 3D plotting routine. Use f-strings for readable debug output, and increase nr_thetas for smoother surfaces (watch computation time).

# %% [markdown]
# # * * *

# %% [markdown]
# **Goal**: compute MSE for a grid of θ₀ and θ₁ values and plot the MSE surface, then locate the (θ₀, θ₁) that gives the minimum MSE.

# %% [markdown]
# Indexing the mesh arrays: plot_t0 and plot_t1 are 2-D (from np.meshgrid). Access a grid point with plot_t0[i, j] and plot_t1[i, j].
#
# If you print plot_t0[i, j] across the loops you will see row-by-row values if you use i then j. Using j then i traverses column-wise.

# %% [markdown]
# ### Plot the surface

# %%
fig = plt.figure(figsize=[16,12])
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.hot, alpha=0.7)
plt.show()

# %% [markdown]
# ### Find the minimum on the grid:

# %%
print("Minimum value of MSE on the grid:", plot_cost.min())
ij_min = np.unravel_index(indices= plot_cost.argmin(), shape= plot_cost.shape)
print("Minimum occurs at grid indices (i,j):", ij_min)

print('Min MSE for Theta 0 at plot_t0[38][65]', plot_t0[38][65])
print('Min MSE for Theta 1 at plot_t1[38][65]', plot_t1[38][65])
# This yields the θ₀ and θ₁ associated with the smallest grid MSE
# — useful to compare with analytic or sklearn results.

# %% [markdown]
# ### if you want a cleaner output

# %%
# ...existing code...
ij_min = np.unravel_index(indices=plot_cost.argmin(), shape=plot_cost.shape)
print("Minimum occurs at grid indices (i,j): ({},{})".format(int(ij_min[0]), int(ij_min[1])))
# ...existing code...

# %% [markdown]
# Practical notes
#
# - The nested-loop grid search is brute force — simple and very instructive, but not efficient for high-dimensional parameter spaces. For finer resolution or higher dimensions use vectorized evaluation, optimization methods, or analytic formulas when available.
#
# - Check meshgrid ordering if axes appear swapped on the plot (swap inputs or indices if needed).
#
# - Use cmap (colormap) to make the surface readable and highlight minima.

# %% [markdown]
# **KEY TERMS / Questions**
#
# - meshgrid: converts 1-D coordinate vectors into 2-D coordinate matrices for surface plotting.
#
# - plot_cost (2-D array): stores MSE values for each (θ₀, θ₁) grid cell.
#
# - nested loop: for i in ...: for j in ...: — used to visit every element in a 2-D grid.
#
# - np.unravel_index: maps a flat index (e.g. from argmin()) back to 2-D indices (i, j).
#
# - argmin vs min: plot_cost.min() gives the minimum value; plot_cost.argmin() gives the flattened index of that minimum.
#
# - Question: If your surface looks transposed, what should you check?
# Answer: Confirm the order you passed arrays to np.meshgrid and whether you index plot_t0[i,j] (row, col) or swapped.
#
# **Summary**
#
# You built a brute-force grid search: create θ₀/θ₁ ranges → np.meshgrid → plot_cost = np.zeros(...) → nested loops compute y_hat and MSE for each grid cell → plot the surface with ax.plot_surface(...) (use cmap for clarity) → find the minimum with argmin + np.unravel_index and read off the corresponding θ₀, θ₁ from the mesh. Increasing nr_thetas smooths the surface but increases compute cost; the grid method is easy to understand but scales poorly in dimensionality.

# %% [markdown]
# ## Partial derivatives of MSE wrt $\theta_0$ and $\theta_1$
#
# ### $$ \frac{\partial MSE}{\partial \theta_0} = \frac{-2}{n} \sum_{i=1}^{n} \big( y^{(i)} - \theta_0 - \theta_1 x^{(i)} \big) $$
# ### $$ \frac{\partial MSE}{\partial \theta_1} = \frac{-2}{n} \sum_{i=1}^{n} \big( y^{(i)} - \theta_0 - \theta_1 x^{(i)} \big) x^{(i)} $$

# %% [markdown]
# Substituted the linear model: $ y= \theta_0 + \theta_1x $ into the MSE

# %%
# x values, y values, theta values
def grad(x, y, thetas):
    n = y.size
    theta0_slope = (-2/n) * sum(y - thetas[0] - thetas[1]*x)
    theta1_slope = (-2/n) * sum((y - thetas[0] - thetas[1]*x)*x)

    return np.concatenate((theta0_slope, theta1_slope), axis = 0)


# %% [markdown]
# Derived the partial derivatives (gradients) analytically and implemented them in Python.
#
# Implemented gradient descent that updates the parameter vector thetas using those gradients, and verified it matches the earlier sklearn solution.

# %%
multiplier = 0.01
thetas = np.array([2.5, 2.5])

# collecting data to plot later

plot_values = thetas.reshape(1, 2)
mse_values = mse_calc(y5, thetas[0] + thetas[1]*x5)

for u in range(500):
    thetas = thetas - multiplier * grad(x5, y5, thetas)

    # append the values to our numpy arrays:
    plot_values= np.concatenate((plot_values, thetas.reshape(1, 2)), axis= 0)
    mse_values= np.append(arr= mse_values, values= mse_calc(y5, thetas[0]+thetas[1]*x5))

#Results 
print("Min occurs at theta0: ", thetas[0])
print("Min occurs at theta1: ", thetas[1])
print("MSE is: ", mse_calc(y5, thetas[0] + thetas[1]*x5))

# %% [markdown]
# **Practical tips / reminders**
#
# Use vectorized operations (as above) rather than Python for-loops over samples for performance.
#
# Use y.size or len(y) for n (instead of hardcoding).
#
# Choose learning rate carefully: too large → divergence, too small → very slow convergence.
#
# Print intermediate values (or plot cost over iterations) to inspect convergence behavior.
#
# If you want a convergence stopping condition, break the loop when np.linalg.norm(multiplier*grad) < tol.
#
# Always verify with a trusted implementation (e.g. sklearn) when possible.

# %%
#plotting MSE
plt.style.use('dark_background')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 10))

ax.set_box_aspect([1.5, 1, 0.7])
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection= '3d')
ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('Cost - MSE', fontsize=20)

ax.scatter(plot_values[:, 0], plot_values[:, 1], mse_values, c= 'black')
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.rainbow, alpha=0.8)

plt.show()

# %% [markdown]
# Key observations
#
# Large initial steps when far from the minimum, then step sizes shrink as slope flattens.
#
# The path reaches the bottom “bowl” region quickly but then requires many small steps to approach the exact minimum (very shallow slope near optimum).
#
# Visualizing the 3D path makes it easy to see step-size dynamics and how learning rate + initialization affect convergence.

# %% [markdown]
# **Summary**
#
# You derived the partial derivatives of MSE with respect to θ₀ and θ₁, implemented them in Python using vectorized NumPy operations, and used these gradients in a gradient descent loop to iteratively update the parameters. You verified your results against sklearn's LinearRegression. Key practical tips include using y.size for sample count, choosing an appropriate learning rate, and monitoring convergence. Visualizing the optimization path on the MSE surface helps understand step-size dynamics and convergence behavior.
