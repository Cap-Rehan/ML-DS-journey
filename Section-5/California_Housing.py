
# %% [markdown]
# # Notebook Imports

# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
# %matplotlib inline

import seaborn as sns

# %%
# from sklearn.datasets import load_boston
# ImportError: 
# `load_boston` has been removed from scikit-learn since version 1.2.

# In this special case, you can fetch the dataset from the original source:

# import pandas as pd
# import numpy as np

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

# %%
# print(type(raw_df))
# print(type(data))
# print(type(target))

# Output:
# <class 'pandas.core.frame.DataFrame'>
# <class 'numpy.ndarray'>
# <class 'numpy.ndarray'>

# %% [markdown]
# 1. Problem Definition
#
# - First step in data science/ML: define the problem clearly and ask the right questions.
#
# - Boston housing example: friend asks “How much does a house cost in Boston?” — without more info, the only valid answer is the average home price (~$567,500).
#
# - Shows why vague or poorly phrased questions → weak solutions.

# %% [markdown]
# 2. Real-World Example (Boston Housing)
#
# - Factors affecting price: size of house, location (downtown/suburbs), features (rooms, crime, schools, etc.).
#
# - Boss’s project: build a valuation tool for real estate agents in Boston.
#
# Tool must:
#
# - Predict house price based on features.
#
# - Show contribution of each feature (interpretable, not a black box).
#
# - Provide quick benchmark prices.
#
# - Be user-friendly (could even be public like Zillow/Zoopla).
#
# 3. Key Takeaways
#
# - Always start with clear, well-phrased goals.
#
# - Need both relevant features and interpretable models.
#
# - The “average price” approach works only when no other info is available, but it’s too simplistic for real-world use.

# %% [markdown]
# ## It didn't work as expected...
# ### sklearn has removed the boston dataset due to ethical issues.
# ### Here are the notes anyway:
#
# 1. Data Gathering
#
# - After defining the problem, the next step in DS/ML is gathering the data.
#
# Typical sources:
#
# - Downloading CSVs from online (Google search).
#
# - Using practice datasets from Python libraries (e.g., scikit-learn).
#
# - Scikit-learn provides clean, user-friendly toy datasets (few missing values, fewer formatting issues).
#
# - Examples: Boston housing, Iris (flowers), Diabetes, Digits, Wine, Breast Cancer.

# %% [markdown]
# **Summary:**
#
# The second stage in ML is data gathering. Scikit-learn used to offer clean practice datasets like the Boston housing dataset, which contained 506 samples and 13 features. After importing with load_boston, the data was stored in a Bunch object. While raw output was messy, the dataset was available in Jupyter, ready for exploration and preprocessing.

# %% [markdown]
# # CHANGE OF PLANS:
# ### We will use the California housing dataset instead.

# %% [markdown]
# # IMPORTING DATASET

# %%
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# %% [markdown]
# We can also Add original paper as a clickable link using Markdown [text]\(URL\).
#
# [Click here to see the original dataset source](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

# %%
type(housing) # Note: convert to DataFrame later for easier handling.

# %%
housing

# %% [markdown]
# 1. **Exploring the Dataset**
#
# After defining the problem and gathering data, next step = exploring the dataset.
#
# Exploration, visualization, and cleaning often happen together (issues only appear once you dig in).
#
# Good starting questions for any dataset:
#
# - What’s the source of the data?
#
# - Is there a description/context of how it was collected?
#
# - How many data points (rows)?
#
# - How many features (columns)?
#
# - What are the names of the features?
#
# - What are the descriptions/units of the features?

# %%
dir(housing)

# %%
print(type(housing.DESCR))
print(housing.DESCR)

# %%
print(type(housing.data))
print(housing.data.shape)
print(housing.data)

# %%
print(type(housing.target))
print(housing.target) 

# Target (house price) is not in features; found in boston_dataset.target.
# Prices look small (24, 21, etc.) → units are in $1000s (e.g., 24 = $24,000).

# %%
print(type(housing.feature_names))
print(housing.feature_names)

# boston_dataset.feature_names → prints all feature names (array).

# %%
print(type(housing.frame))
print(housing.frame)

# %% [markdown]
# Dot notation & nesting:
#
# - california_housing → Bunch object.
#
# - .data → numpy array.
#
# - .shape → tuple (rows, columns).
#
# - Like Inception, dreams within dreams.

# %% [markdown]
# **Summary:**
#
# Exploring a dataset begins with six key questions about its source, size, and features. The California dataset has 2000+ samples and 8 attributes, with context provided by its original research study. Using '.DESCR' and '.shape' in Python helps confirm this information. Data exploration not only reveals structure but also teaches how Python objects nest attributes inside one another.

# %% [markdown]
# # Continuing Exploration

# %% [markdown]
# ### 1. Attributes, Features & Targets
#
# Important distinction:
#
# - In Python → “attribute” = property of an object.
#
# - In ML → “attribute” usually means feature/independent variable (columns in dataset).

# %% [markdown]
# ### 2. Converting to Pandas DataFrame
#
# Pandas DataFrame = main workhorse in ML/DS.

# %%
housing_data = pd.DataFrame(data= housing.data, columns= housing.feature_names)
housing_data['House Price']= housing.target

# %%
# Explore DataFrame quickly:
housing_data.head()

# %%
housing_data.tail()


# %%
housing_data.count()

# %% [markdown]
# ### Vocabulary check:
#
# Instance (ML) = a row/data point.
#
# Instance (programming) = an object of a class.

# %% [markdown]
# ## 3. Checking for Missing Values
#
# Why? ML algorithms break if fed missing data.

# %%
# pandas tools:
pd.isnull(housing_data)

# %%
pd.isnull(housing_data).any() # better way

# %%
housing_data.info() # more details with a single command

# %% [markdown]
# **Summary:**
#
# This lesson clarified the meaning of attributes/features in ML, showed how to extract feature names and targets, and converted the California dataset into a Pandas DataFrame for easier work. We practiced quick exploration with .head(), .tail(), .count(), and checked for missing values using isnull() and info(). Good news: the dataset has no missing values, so we’re ready to start digging into feature meanings.

# %% [markdown]
# ## 1. Why Visualize Data?
#
# Visualization helps at exploration stage, not just final reporting.
#
# Two main goals when exploring:
#
# - Understand distribution of variables.
#
# - Spot outliers (values far from the rest).

# %% [markdown]
# ### First visualization tool:
# ### Histogram (bar chart showing frequency of values).
#
# A “normal distribution” looks like a bell curve; real-world data (like California house prices) is usually messier with outliers.

# %% [markdown]
# ### 2. Creating Histograms with Matplotlib

# %%
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.hist(housing_data["House Price"], bins= 50, ec= 'black', color= 'steelblue') # customization
plt.xlabel("Price in $100,000s")
plt.ylabel("No. of houses")

plt.show()

# %% [markdown]
# ### 3. Key Insights & Next Steps
#
# - Histogram of prices = uneven distribution with clear outliers (very high house values).
#
# - Good reminder: real-world data ≠ neat bell curve.
#
# - Ronald Reagan quote: “Trust but verify” → always check visuals yourself.
#
# - Matplotlib works, but there are other libraries for visualization → next up is Seaborn.

# %% [markdown]
# **Summary:**
#
# Histograms are the first visualization step in data exploration. They reveal data distributions and highlight outliers, making them essential before modeling. Using matplotlib, we can plot, style, and label histograms of house prices. The Boston housing data shows a skewed distribution with high-value outliers, reminding us that real-world data rarely follows perfect statistical patterns.

# %% [markdown]
# ## 1. Seaborn Basics
#
# Seaborn = built on matplotlib, but with extra features and prettier defaults.

# %%
# plt.figure(figsize=(16, 10))
# sns.displot(housing_data["House Price"], kde= True, ec= 'None')
# plt.show()

# histogram + probability density function (PDF)

plt.figure(figsize=(16, 10))
sns.histplot(data=housing_data, x="House Price", kde=True, ec='None')
plt.show()

# %% [markdown]
# ### 2. Comparing Features
#
# Explore Average Rooms

# %%
plt.figure(figsize=(16, 10))
sns.histplot(housing_data["AveRooms"], kde= True, color="#008080")  # teal
plt.xlabel("Average number of rooms")
plt.ylabel("No. of houses")
plt.xlim(0, 15)
plt.show()

# %% [markdown]
# Histogram shows peak around 5-6 rooms.
#
# Exact calculation:

# %%
print(housing_data["AveRooms"].mean())

# %%
print(housing_data["AveRooms"].min())

# %%
print(housing_data["AveRooms"].max())

# %% [markdown]
# ### 3. Key Insights & Next Steps
#
# - Seaborn makes visualization simpler and more informative (with PDF overlay).
#
# - Distribution of PRICE: messy, skewed, outliers.
#
# - Distribution of AvgRM: centered around 5-6 rooms, mean = 5.4...
#
# - Combining visual inspection (histograms) + statistical summaries (mean) = stronger understanding of features.

# %% [markdown]
# **Summary:**
#
# Seaborn builds on matplotlib, offering easier syntax and attractive defaults like adding PDFs to histograms. By exploring the RM feature, we see Boston homes average about 6.28 rooms. Together, visualizations and summary statistics help confirm insights about dataset features, preparing us for deeper analysis.

# %%
housing_data.head()

# %% [markdown]
# ## 1. Exploring the HouseAge Feature
#
# First attempt: histogram looked odd (automatic binning hid info).

# %%
plt.figure(figsize=(16, 10))
plt.hist(housing_data["HouseAge"], ec= 'black', color= "#a200ff", rwidth= 0.8)
plt.xlabel("Median house age in block group")
plt.ylabel("No. of houses")

plt.show()

# %%
freq = housing_data["HouseAge"].value_counts() # Shows frequency of each HouseAge value.
print(freq)

# %%
# Fix: specify bins to match index values:
plt.figure(figsize=(16, 10))
plt.hist(housing_data["HouseAge"], bins=52, color="#6A0DAD", rwidth=0.5)  # royal purple
plt.xlabel("Median house age in block group")
plt.ylabel("No. of houses")

plt.show()

# %% [markdown]
# ### 2. Bar Charts with Matplotlib
#
# Created bar chart using value counts for flexibility
#
# Advantage: no need to hardcode number of bins.

# %%
plt.figure(figsize=(16,10))
plt.bar(freq.index, freq, color="#6A0DAD")
plt.xlabel("Age of the House")
plt.ylabel("Number of Properties")
plt.show()

# %% [markdown]
# ### 3. CHAS Feature (Charles River Dummy Variable)
# ### was for Boston housing dataset
#
# CHAS = dummy variable (binary):
#
# 1 = property bounds Charles River.
#
# 0 = property not on river.
#
# Count with:
#
# data["CHAS"].value_counts()
#
#
# Output: 35 properties on river (CHAS=1), rest not.
#
# Dummy variables capture binary info (e.g., yes/no, male/female, employed/unemployed).

# %% [markdown]
# **Summary:**
#
# We explored RAD, an index feature, and learned that histograms need tailored bins when working with discrete values. Using value_counts() + bar charts provides a flexible alternative to histograms. We also introduced CHAS, a dummy variable for riverfront properties, finding 35 such cases. Indexes and dummy variables are powerful tools for encoding categorical or binary data in ML.

# %% [markdown]
# ## 1. Why Descriptive Statistics Matter
#
# Complement visualizations with summary statistics.
#
# Politician story:
#
# - Mean (average) = total ÷ count, influenced by outliers.
#
# - Median = middle value when sorted, resistant to outliers.
#
# Both can give different pictures depending on distribution shape.
#
# - Normal distribution: mean ≈ median.
#
# - Skewed distribution: mean pulled toward outliers (e.g., a few ultra-wealthy families).

# %% [markdown]
# ### 2. Pandas Methods for Statistics

# %%
# single column
print("$", round(housing_data["House Price"].min() * 100000, 3))
print("$", round(housing_data["House Price"].max() * 100000, 3))

# %%
# whole dataframe
print(housing_data.min(), "\n")
print(housing_data.max(), "\n")
print(housing_data.mean(), "\n")
print(housing_data.median())

# %%
# All in one
housing_data.describe()

# Outputs count, mean, std, min, 25%, 50% (median), 75%, max.
# Automatically ignores NaN values.

# %% [markdown]
# 3. Insights from California Dataset
#
# - Prices range: $15k → $500k (1970s values).
#
# - Mean/median differences highlight skewness in features.
#
# - Outlier in rooms (AveRooms):
#
# - Mean ≈ 5.42, median ≈ 5.22, but one property has nearly 141 rooms → flagged for analysis later.
#
# - describe() = quick, reliable snapshot for any dataset.

# %% [markdown]
# **Summary:**
#
# Descriptive statistics provide a numeric overview of data distributions. Using pandas methods like .min(), .max(), .mean(), .median(), and .describe(), we can quickly summarize central tendency and spread. In the California dataset, we confirmed house price ranges and spotted an outlier in the AveRooms feature. These summaries guide deeper exploration before modeling.

# %% [markdown]
# ## 1. What Correlation Means
#
# Definition: Correlation = how strongly two variables move together.
#
# Types:
#
# - Positive correlation: both increase/decrease together (e.g., sun ↑ → ice cream consumption ↑).
#
# - Negative correlation: one increases while the other decreases (e.g., time on crowded train ↑ → happiness ↓).
#
# - No correlation: variables move independently (e.g., margarine consumption vs divorces in Maine).
#
# Strength:
#
# - Weak → cloud of points.
#
# - Strong → points line up more tightly.
#
# - Perfect correlation → straight line.
#
# Mathematical notation: ρ (rho).
#
# - Range: -1 → +1.
#
# - -1 = perfect negative, 0 = no correlation, +1 = perfect positive.

# %% [markdown]
# ### 2. Why Correlation Matters in ML
#
# Helps identify which features are useful for predicting target (house price).
#
# We care about:
#
# - Strength (magnitude): correlation should be far from 0.
#
# - Direction (sign): tells us whether features move in the same (+) or opposite (-) direction as target.
#
# High-correlation features = strong candidates for model inputs.
#
# Also important to check feature-to-feature correlations → avoid redundancy (multicollinearity).

# %% [markdown]
# ### 3. Preparing for Analysis
#
# We’ll explore correlations in two ways:
#
# - Calculate correlations (numeric summary).
#
# - Visualize correlations (plots/heatmaps).
#
# - Jupyter notebook tools (e.g., pandas .corr(), matplotlib, seaborn) make this quick and interpretable.

# %% [markdown]
# **Summary:**
#
# Correlation measures how variables move together, with values between -1 and +1 showing strength and direction. In machine learning, we care about correlations with the target variable (house price) to select predictive features, and also correlations between features to avoid overlap. Next step: compute and visualize correlations in Python.

# %% [markdown]
# ## 1. Calculating Correlations in Pandas

# %% [markdown]
# ### Correlation  
# ### $$\rho_{xy} = corr(x, y)$$  
# ### $$-1.0 \leq \rho_{xy} \leq +1.0$$

# %%
# Example:
print(housing_data["House Price"].corr(housing_data["AveRooms"]))
# → Larger houses (more rooms) tend to be more expensive.

# %%
print(housing_data["House Price"].corr(housing_data["MedInc"]))
# → Higher-income areas tend to have more expensive houses.

# %%
print(housing_data["House Price"].corr(housing_data["Population"]))
# → More populated areas tend to have slightly cheaper houses.

# %% [markdown]
# ### 2. Correlation Matrix & Pearson Coefficients
#
# One-by-one is slow → get all correlations at once:

# %%
housing_data.corr()

# %% [markdown]
# Produces a full symmetric table:
#
# - Last column = correlations with target (PRICE).
#
# - Diagonal = 1 (variable correlated with itself).
#
# - Upper & lower halves are duplicates (symmetric).
#
# - Default method = Pearson correlation (linear relationships).
#
# This gives both:
#
# - Strength (how far from 0).
#
# - Direction (positive or negative).

# %% [markdown]
# ### 3. Multicollinearity & Why It Matters
#
# - Definition: When two or more predictors are highly correlated, making their effects hard to distinguish.
#
# - Example: body fat ↔ weight (very correlated). Both affect bone density, but they move together → regression model can’t separate their contributions.
#
# - Problem: regression estimates become unreliable; model gets “confused.”
#
# - Takeaway: High correlations between features don’t always mean multicollinearity, but they’re a warning sign worth investigating.

# %% [markdown]
# **Summary:**
#
# We learned how to compute correlations both individually (.corr()) and across the whole dataset (data.corr()). Strong positive or negative correlations help us identify features useful for predicting house prices. But correlations between features themselves may indicate multicollinearity, which can distort regression results. This is something to revisit in the regression stage.

# %% [markdown]
# ## 1. Visualizing correlations (goal & idea)
#
# - We want a clean, report-ready correlation view — hide duplicate values in the correlation matrix and show just one triangle.
#
# - Why hide duplicates? The correlation matrix is symmetric; showing both halves is redundant and visually noisy.
#
# - Approach: build a mask (boolean array) that selects the upper triangle, and supply it to Seaborn’s heatmap so only the lower triangle (or vice-versa) is shown.
#
# - Add annotations (numbers) and increase font sizes so the chart is readable for reports.

# %%
# correlation matrix and mask (hide upper triangle)
corr = housing_data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True   # mask upper triangle

# plot
# sns.set_style('dark')
plt.style.use('dark_background')
plt.figure(figsize=(16,10))
sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size":14})

plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.show()

# %% [markdown]
# ### 2. What the heatmap tells you (examples & cautions)
#
# Colors encode direction & strength: dark red = strong positive, dark blue = strong negative, pale ≈ 0.
#
# Example findings in Boston data:
#
# - AveRooms & AveBedrms have strong positive correlation (~0.85) — makes sense.
#
# - Latitude & Longitude: strong negative correlation (~-0.92) — location-based relationship.
#
# Caveats / limitations to keep in mind:
#
# - Pearson correlation (default) assumes continuous variables and linear relationships — not valid for dummy variables (CHAS) or discrete indices (RAD) in the strict sense.
#
# - Correlation ≠ causation. Spurious correlations exist.
#
# - Pearson only captures linear relationships. Low Pearson value doesn’t guarantee no relationship (nonlinear or outliers can hide patterns) — see Anscombe’s Quartet.
#
# - High feature-feature correlation can warn about multicollinearity (redundant predictors) — not a guaranteed problem, but a red flag to investigate in regression stage.

# %% [markdown]
# ### 3. Practical next steps & decisions for regression
#
# Use the heatmap to spot:
#
# - Features strongly correlated with PRICE → good candidates for predictors.
#
# - Features strongly correlated with each other → possible multicollinearity; decide whether to drop/transform/combine features.
#
# For discrete or dummy variables (RAD, CHAS) consider alternative association measures or treat them appropriately (e.g., bar charts, categorical tests) rather than relying only on Pearson.
#
# Complement correlation analysis with scatter plots for suspicious pairs (to spot nonlinearity or influential outliers) before final model building.

# %% [markdown]
# **Summary:**
#
# Make a masked, annotated correlation heatmap to get a compact, readable view of relationships. Use it to find promising predictors and possible multicollinearity, but always remember the limitations: Pearson assumes continuous linear relationships and correlation does not imply causation. Always follow up high/low correlation findings with visualization (scatterplots, histograms) and domain thinking before changing your model.

# %% [markdown]
# ## 1. Why scatter plots?
#
# - Scatter plots show pairwise relationships and reveal linear vs non-linear patterns, clusters, densities, and outliers.
#
# - Always use scatter + descriptive stats (correlations, mean/median, histograms) together — numbers + pictures catch things that each alone misses (Anscombe’s Quartet).
#
# - Use scatter plots to confirm what the correlation matrix hinted at and to decide if features should be transformed or excluded.

# %% [markdown]
# ### 2. Matplotlib scatter (basic → polished)

# %%
plt.figure(figsize=(12,8))
plt.style.use('classic')
plt.scatter(housing_data["Longitude"], housing_data["Latitude"], s=80, alpha=0.4, color="indigo")
plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.title(f"Longitude vs Latitude (Correlation = {round(housing_data['Longitude'].corr(housing_data['Latitude']), 2)})", fontsize=14)
plt.show()

# %% [markdown]
# ### 3. Seaborn jointplot (scatter, hex, reg, kde, resid)
#
# sns.jointplot adds marginal histograms and prints the Pearson r by default:

# %%
# plt.figure(figsize=(10,6)) not working here, use height
sns.set_context("talk")

sns.jointplot(x=housing_data["Longitude"], y=housing_data["Latitude"], height= 10,color="darkblue", joint_kws={"alpha":0.6})

plt.show()

# %%
# sns.set_style('darkgrid')
plt.style.use('classic')

sns.set_context("poster")
sns.jointplot(x=housing_data["Longitude"], y=housing_data["Latitude"], height= 11, kind='hex', color='indigo')

plt.show()

# %%
sns.set_context("talk")
sns.jointplot(x=housing_data["AveRooms"], y=housing_data["AveBedrms"], height= 11, kind='reg', color='indigo')

plt.show()

# %%
# sns.set_context("talk")
# sns.jointplot(x=housing_data["Longitude"], y=housing_data["Latitude"], height= 11, kind='kde', color='indigo')

# plt.show()
# # image saved as kde.png

# %% [markdown]
# When to use which kind:
#
# - scatter: general pairwise view.
#
# - hex: dense data / show 2D density (hex bins).
#
# - reg: adds regression line and CI — helpful to see linear trend + fit.
#
# - kde: smoothed density contours.
#
# - resid: residuals from linear fit (diagnostic).
#
# Use joint_kws dictionary to pass marker params like alpha and s.

# %% [markdown]
# **Summary:**
#
# Scatter plots are essential for checking the real shape of relationships suggested by correlations — they expose nonlinearity, outliers, and density that correlations alone can’t show. Use matplotlib scatter for full control, and sns.jointplot for quick, attractive paired views (with marginal histograms and multiple “kinds” like hex or reg). Always add labels, tune alpha/s, and include the correlation (rounded) in the title so the chart speaks for itself.

# %% [markdown]
# ## 1. Scatter plots + regression — what & why
#
# - Scatter plots reveal real relationships (shape, clusters, outliers) that correlations alone can mask.
#
# - Example: TAX vs RAD looked highly correlated (ρ ≈ 0.91) in the table, but the scatter revealed that the high correlation was driven by a few discrete values / outliers — visual check overturned the blind trust in the number.
#
# - Use sns.lmplot (or sns.regplot / plt.scatter + manual line) to quickly fit and plot a linear regression line; the line can be pulled by outliers, so inspect plots before trusting fit coefficients.
#
# - Always visualize before modeling: a regression line forced on bad / discrete / outlier-driven data gives misleading results.

# %%
plt.figure(figsize=(16,10))
corr_val= round(housing_data["MedInc"].corr(housing_data["House Price"]), 2)

plt.scatter(housing_data["MedInc"], housing_data["House Price"], s=80, alpha=0.6, color="steelblue")
plt.xlabel("MedInc", fontsize=14)
plt.ylabel("House Price", fontsize=14)
plt.title(f"MedInc vs House Price (Correlation = {corr_val})", fontsize=16)

plt.show()

# %%
# seaborn lmplot (quick linear fit)
sns.lmplot(x='MedInc', y='House Price', data=housing_data, height=10, line_kws={'color':'red'})
plt.show()

# %%
# plotting correlation matrix again

# correlation matrix and mask (hide upper triangle)
corr = housing_data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True   # mask upper triangle

# plot
# sns.set_style('dark')
plt.style.use('dark_background')
plt.figure(figsize=(16,10))
sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size":14})

plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.show()

# %% [markdown]
# ### 2. Practical pairwise exploration (MedInc vs PRICE, etc.)
#
# - MedInc vs Price: clear positive relationship (ρ ≈ +0.7). Use scatter + lmplot to see the fit and a possible ceiling effect at the top (data artifact / collection effect).
#
# - Latitude vs Longitude: strong negative relationship; location based results.
#
# - sns.pairplot(data) graphs all pairwise scatterplots + histograms on the diagonal (very lazy / powerful). Use %%time to measure runtime because pairplot can be slow on larger datasets.
#
# - For pairplot regression lines: sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color':'cyan'}}) — nested dict lets you style the regression line separately from points.

# %%
# # pairplot with regression lines + colored regression line
# sns.pairplot(housing_data, kind='reg', plot_kws={'line_kws':{'color':'cyan'}})
# plt.show()
# # output image saved in the same directory as this notebook 

# %% [markdown]
# ### 3. Jupyter + plotting tips & diagnostics
#
# - Styling / density tricks: alpha for transparency, s for marker size; use hex bins (kind='hex') when scatter is overplotted; jointplot shows marginal histograms and Pearson r.
#
# - Save large images if you want to inspect full resolution (right-click → Save as).
#
# - Notebook microbenchmarking: use %%time at top of a cell to see how long plotting / computations take. Watch the kernel status (idle / busy) indicator.
#
# Interpretation rules:
#
# - A high Pearson number can be misleading for discrete/indexed variables or when outliers dominate.
#
# - Always combine numeric summaries + scatter plots (Anscombe’s Quartet lesson).
#
# - Pairwise regressions are helpful but true power is in multivariable regression (next step) — pairwise checks guide feature selection and spotting multicollinearity.

# %% [markdown]
# **Summary:**
#
# Scatterplots (matplotlib or seaborn) let you confirm or challenge what correlation numbers imply: they expose outliers, discrete/indexed patterns, ceilings, and nonlinearity that can distort correlations and regressions. Use lmplot/regplot to add fit lines, pairplot to scan many combinations (benchmarked with %%time), and always follow a suspicious correlation with a plotted check before building a regression model.

# %% [markdown]
# ## 1. Key Concepts of Multivariable Regression
#
# - Multivariable regression (aka multiple linear regression) models outcomes using multiple explanatory variables (features).
#
# - Simple linear regression: one feature (e.g., movie budget → movie revenue).
#
# - Multivariable regression: many features (e.g., 8 housing features → price).
#
# Model equation:
#
# 𝑦 = 𝜃_0 + 𝜃_1*𝑥_1 + 𝜃_2*𝑥_2 + ⋯ + 𝜃_𝑛*𝑥_𝑛
#
# Each 𝜃 is a coefficient learned from data, showing how much a feature contributes.
#
# ### 2. Model Application:
#
# - Housing dataset: 8 features used to predict property price.
#
# - Model remains linear because the prediction is a linear combination of features.
#
# - After training, Python outputs values for each coefficient (𝜃).
#
# - These coefficients = “weights of importance” for each feature in prediction.
#
# ### 3. Historical Insight:
#
# - Term “regression” comes from Sir Francis Galton (Victorian England).
#
# - Galton’s “Galton board” illustrated normal distribution (bell curve).
#
# - He studied inheritance: tall fathers often had shorter sons → coined “Regression to the Mean.”
#
# - Real-world example: NBA players’ sons are usually shorter than their very tall fathers (e.g., Shaquille O’Neal, Michael Jordan).
#
# - Core idea: extreme values tend to be followed by less extreme ones in the next generation.
#
# **Summary:**
#
# Multivariable regression extends simple linear regression by using multiple features to predict outcomes (like house prices). Each coefficient shows the feature’s contribution. The term “regression” originates from Galton’s observation of regression to the mean—extreme traits in one generation tending to be less extreme in the next.

# %% [markdown]
# ## 1. What & why: train / test split
#
# - Goal: split your data into a training set (used to learn θ parameters) and a test set (used only to evaluate out-of-sample performance).
#
# - Why shuffle: raw datasets can be ordered (like an unshuffled deck). Random shuffling prevents order-based bias in which rows go to train vs test.
#
# - Typical split: 70/30, 80/20, 90/10 are common. You picked 80/20 (test_size=0.2) — good default for moderate datasets.

# %% [markdown]
# ### 2. Key API & mechanics (pandas + scikit-learn)
#
# - Create prices (target) and features (all columns minus PRICE): use df.drop('PRICE', axis=1). axis=1 means "drop a column".
#
# - train_test_split(features, prices, test_size=..., random_state=...) returns four objects: X_train, X_test, y_train, y_test. That’s tuple unpacking in Python.
#
# - random_state fixes the RNG seed so your split is reproducible (same rows in train/test each run).

# %%
# prepare target & features

tgt = housing_data['House Price']
feat = housing_data.drop('House Price', axis=1) # axis=1 means "drop a column"

# split: 80% train, 20% test

X_train, X_test, y_train, y_test = train_test_split(feat, tgt, test_size=0.2)

# quick checks
print("X_train.shape:", X_train.shape)
print("X_test.shape: ", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape: ", y_test.shape)

# %% [markdown]
# ### 3. Quick sanity checks after splitting
#
# - Check shapes: X_train.shape, X_test.shape (rows, columns).
#
# - Verify split fraction: len(X_train)/len(features) ≈ 0.8 and len(X_test)/len(features) ≈ 0.2.
#
# - If your dataset is small and target distribution matters, consider stratified approaches (not typical for continuous regression targets).

# %% [markdown]
# **Summary:**
#
# Always shuffle & split before training so your algorithm learns from one set and is evaluated on unseen data. Use train_test_split(..., test_size=0.2, random_state=...) for a reproducible 80/20 split, then quickly check shape or len(...) ratios to confirm.

# %%
# verify fractions (should be ~0.8 and ~0.2)
print("Train fraction:", len(X_train) / len(feat))
print("Test fraction:", len(X_test) / len(feat))

# %% [markdown]
# ## 1. Training
#
# - Created LinearRegression() and fit on training data: regr.fit(X_train, y_train).
#
# - Stored intercept with regr.intercept_ and coefficients with regr.coef_ (mapped to feature names via pd.DataFrame).

# %%
regr = LinearRegression()
regr.fit(X_train, y_train)

# %%
coef_df = pd.DataFrame(regr.coef_, index=X_train.columns, columns=["COEFs"])
print("Intercept:", regr.intercept_)
coef_df

# %% [markdown]
# ### 2. Coefficients & interpretation
#
# - Coefficient signs matched expectations:
#
# - Negative: AveRooms, Population, AveOccup, Latitiude, Longitude (bad for price).
#
# - Positive: MedInc, AveBedrms (good for price).
#
# - Units: PRICE in hundred-thousands → coefficient c means ≈ c * $100000 change per 1-unit feature change.
#
# - Example: MedInc → 0.44 ≈ $ 44,000 premium for Higher Income Society;per extra unit.

# %%
print("Training data r-squared:", regr.score(X_train, y_train))
print("Test data r-squared:    ", regr.score(X_test, y_test))

# %% [markdown]
# ### 3. Evaluation (R²)
#
# - R² via regr.score(...).
#
# - Training R² ≈ 0.60; Test R² ≈ 0.58.
#
# - Test R² is lower because the model was trained on training data only; test R² measures out-of-sample predictive power.

# %% [markdown]
# **Summary:**
#
# Trained a multivariable linear model, inspected interpretable coefficients (units in thousands → convert to dollars), and evaluated generalization with train R² ≈ 0.60 and test R² ≈ 0.58.

# %% [markdown]
# ## 1. Model evaluation stage
#
# - Final step in workflow: evaluate, refine, and deploy model.
#
# - Similar to a medical check-up → many stats to examine model health.
#
# - Beyond R², other key statistics: p-values, Variance Inflation Factor (VIF), Bayesian Information Criterion (BIC).
#
# ### 2. Data transformations (log of target)
#
# - House price distribution is left-right-both-skewed (skew ≈ 1.1).
#
# - Goal: reduce skew → improve linear regression fit.
#
# - Applied log transformation:

# %%
print(housing_data['House Price'].skew())

# %%
sns.displot(housing_data['House Price'], kde=True, height=10)
plt.show()

# %%
y_log = np.log(housing_data['House Price'])
new_housing_data = housing_data.copy().drop(columns=['House Price'])
new_housing_data['y_log'] = y_log

# %%
print(y_log.skew())

# %%
sns.displot(y_log, kde=True, height=10)
plt.show()

# %%
sns.lmplot(x='MedInc', y='House Price', data=housing_data, height=9, line_kws={'color':'red'})
plt.show()

# %%
sns.lmplot(x='MedInc', y='y_log', data=new_housing_data, height=9, line_kws={'color':'red'})
plt.show()

# %% [markdown]
# - Skew after log ≈ -0.17 → closer to 0 (normal distribution).
#
# - Visualization with sns.displot(y_log) confirmed more symmetry.
#
# - Scatterplots: PRICE vs MedInc less linear; LOG_PRICE vs MedInc more linear.

# %% [markdown]
# ### 3. Regression with log prices
#
# - Re-trained model with log prices as target:

# %%
X_train, X_test, y_train, y_test = train_test_split(feat, y_log, test_size=0.2, random_state= 0)

regr = LinearRegression()
regr.fit(X_train, y_train)

print('Training data r-squared:', regr.score(X_train, y_train))
print('Test data r-squared:', regr.score(X_test, y_test))

print('Intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])

# %% [markdown]
# - Performance improved: Train R² ↑ from 0.60 → 0.61; Test R² ↑ from 0.58 → 0.60.
#
# - Coefficients changed meaning:
#
# - Example: MedInc coefficient ≈ 0.18.
#
# - Reversed log transform to interpret in dollars:

# %%
np.e ** 0.185359

# %% [markdown]
# **Summary:**
#
# Introduced model evaluation stage, showed how right-skewed house prices reduce model fit, applied log transformation to reduce skew, retrained regression, and saw somewhat higher R² values. Interpretation of coefficients now requires reversing the log transform.

# %% [markdown]
# ## 1. Coefficient significance & p-values
#
# - Beyond sign & size of coefficients, we must check statistical significance.
#
# - p-value is the key metric → measures how likely a coefficient is meaningful.
#
# - Rule of thumb: p < 0.05 → significant; p > 0.05 → not significant.
#
# - scikit-learn does not provide p-values → need Statsmodels library.

# %% [markdown]
# ### 2. Using Statsmodels for regression

# %%
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const) # OLS = Ordinary Least Squares
results = model.fit()

# get coefficients and p-values
pd.DataFrame({'Coefficients': results.params, 'p-values': round(results.pvalues, 3)})

# %% [markdown]
# ## 3. Interpretation of results
#
# - Coefficients match scikit-learn results.
#
# - p-values show which features matter statistically.
#
# In this case:
#
# - All features are statistically significant.
#
# - Suggestion: consider dropping insignificant features later.

# %% [markdown]
# **Summary:**
#
# Introduced p-values as the "vital stats" of regression coefficients. Used Statsmodels to run OLS regression and extract both coefficients and p-values. Found that there's no feature that does not significantly contribute. Next step → check for multicollinearity.

# %% [markdown]
# ## 1. What multicollinearity is & why care
#
# - Definition: multicollinearity = two or more predictors highly correlated → they don’t provide independent information.
#
# - Consequences: unstable coefficient estimates, large variability when model changes, coefficients can flip sign, misleading inference.
#
# - Symptoms: sense-check signs/logic (passed here) but also need formal tests (VIF) and sensitivity checks.
#
# ### 2. Variance Inflation Factor (VIF) — concept & workflow
#
# Idea: for each feature (e.g., AveRooms) regress that feature on all other features → get $R_{feature}^{2}$
#
# VIF formula: 
#
# $VIF_{j} = \frac{1}{1−R_{j}^{2}}$​
#
# Interpret: large VIF → strong collinearity. Common thresholds: >10 (problematic), more conservative >5.
#
# Required steps (code flow):

# %%
# VIF for one column (exog must be ndarray)
vif = variance_inflation_factor(exog= X_incl_const.values, exog_idx=1)

# %%
vif_list = [variance_inflation_factor(X_incl_const.values, i) for i in range(X_incl_const.shape[1])]

vif_df = pd.DataFrame({'Feature': X_incl_const.columns, 'VIF': np.around(vif_list, 2)})
vif_df

# %% [markdown]
# ### 3. Results & interpretation (this dataset)
#
# - Computed VIFs for all features (constant + 8 features).
#
# - All VIF values < 10 (many well below) → no strong evidence of problematic multicollinearity by common rule-of-thumb.
#
# - Some researchers use cutoff 5; still acceptable here.
#
# - Historical note: original paper had multicollinearity when including multiple pollution measures → they removed a redundant pollutant variable.
#
# **Summary:**
#
# Tested multicollinearity with VIF (regress each feature on the others, compute $1/(1−R^{2})$). VIFs are below threshold (≤10), so multicollinearity isn’t a severe problem for this model; removing redundant features remains a valid remedy if needed.

# %% [markdown]
# ## 1. Why simplify & the metric (BIC)
#
# - Simpler models preferred “all else equal” (Occam’s razor / Zen of Python).
#
# - BIC (Bayesian Information Criterion) compares models: lower BIC = better (penalizes complexity).
#
# - Use BIC + fit (R²) to decide if dropping a feature reduces complexity without hurting fit.
#
# ### 2. Procedure (how we compared models)
#
# - Use Statsmodels RegressionResults to get BIC and R²:

# %%
import this # huehuehue

# %%
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const) # OLS = Ordinary Least Squares
results = model.fit()

# get coefficients and p-values
org_coef = pd.DataFrame({'Coefficients': results.params,
                         'p-values': np.around(results.pvalues, 3)})
print(results.bic)
print(results.rsquared)

# %%
a = X_incl_const.drop(columns=['AveBedrms', 'MedInc'], axis=1) 
# drop two major columns

model2 = sm.OLS(y_train, a) # OLS = Ordinary Least Squares
results2 = model2.fit()

# get coefficients and p-values
reduced_coef = pd.DataFrame({'Coefficients': results2.params,
                         'p-values': np.around(results2.pvalues, 3)})
print(results2.bic)
print(results2.rsquared)

# worse...

# %%
b = X_incl_const.drop(columns=['Population'], axis=1) # trying something else

model3 = sm.OLS(y_train, b) # OLS = Ordinary Least Squares
results3 = model3.fit()

# get coefficients and p-values
reduced_coef2 = pd.DataFrame({'Coefficients': results3.params,
                         'p-values': np.around(results3.pvalues, 3)})
print(results3.bic)
print(results3.rsquared)

# better...

# %% [markdown]
# ### 3. Experiments & findings (this dataset)
#
# - Original (log-target) model: BIC ≈ 12604, R² ≈ 0.615
#
# - Remove Population → BIC lower (~ 12598), R² stable → improvement.
#
# - Removing other important features (e.g., MedInc) increases BIC and lowers R² → don’t drop them.
#
# - Check coefficient stability across models by concatenating coefficient DataFrames:

# %%
frames = [org_coef, reduced_coef, reduced_coef2]
pd.concat(frames, axis=1)
# coefficients remained stable (no sign flips) → no strong evidence of harmful multicollinearity; dropping column(s) is safe here.
# except AveRooms changed its sign when AveBedrms was removed.
# we'll see to it...

# %% [markdown]
# **Summary:**
#
# Used BIC + R² to guide feature removal; dropping one column reduced model complexity (BIC ↓ ~10) with no material loss in R² and coefficients stayed stable (except rooms case) —so simplified model is preferred (maybe).

# %% [markdown]
# ## 1. Residuals & why they matter
#
# - Residual = difference between actual target 𝑦 and predicted value 𝑦^ .
#
# - Example: actual = 50, predicted = 48 → residual = 2.
#
# - With 20k training samples → 20k predicted values → 20k residuals.
#
# - Residuals help check regression assumptions: if assumptions hold → model is useful; if not → results may be misleading.
#
# Assumptions include:
#
# - Linearity (fit is roughly linear; we log-transformed target to help).
#
# - Residuals should be random (no clear pattern).
#
# - Ideally, residuals are normally distributed (mean ≈ 0, skew ≈ 0).
#
# ### 2. What to look for in residuals
#
# Problematic patterns in residual plots:
#
# - Clear curve/shape → missing non-linear relationship.
#
# - Cone shape → heteroscedasticity (variance grows with predictions).
#
# - Vertical clusters → missing features or interactions.
#
# - Outliers → single data points dominate.
#
# - Unbalanced axis → some very large residuals → poor fit/need transformation.
#
# Healthy residuals:
#
# - No obvious pattern.
#
# - Cloud centered around 0, roughly symmetric.
#
# - Normal distribution assumption applies to residuals (not to features or targets).
#
# ### 3. Interpretation & key takeaway
#
# - Residuals show what the model fails to capture.
#
# - If residuals have structure → missing variables, poor transformations, or mis-specified relationships.
#
# - If residuals look random & centered → model assumptions reasonably hold.
#
# - Quote by George Box: “All models are wrong, but some are useful.”
#
# **Summary:**
#
# Residuals are the “health check” of a regression. We want them random, symmetric, and centered around zero. Patterns, clusters, or cones signal problems like missing variables or heteroscedasticity. Residual analysis validates whether our regression is a useful simplification of reality.

# %% [markdown]
# ## 1. Residuals setup & calculation
#
# Modified model:
#
# - Used log prices (transformation).
#
# - Dropped a feature: Population.
#
# - Continued with Statsmodels for residual analysis.
#
# Residual formula: 
#
# - residual = y − $\hat y$
#
# - results.fittedvalues → fitted predictions.
#
# - y_train - results.fittedvalues → residuals.
#
# - Shortcut: results.resid.
#
# Residuals stored as pandas Series → can use methods like .describe().
#
# - Mean of residuals ≈ 0.

# %%
results3.fittedvalues

# %%
results3.resid

# %%
# type(results3.resid)

print(round(results3.resid.mean(), 5))

# %% [markdown]
# ### 2. Visualization (Actual vs Predicted)
#
# Correlation between actual & predicted:

# %%
corr = round(y_train.corr(results3.fittedvalues), 3)
print(corr)

# %%
# Scatter plot of actual vs predicted log prices:
plt.figure(figsize=(12,8))
plt.style.use('fivethirtyeight')
plt.scatter(y_train, results3.fittedvalues, c='navy', alpha=0.5)
plt.plot(y_train, y_train, color='cyan') # 45-degree line
plt.title(f"Actual vs Predicted log prices (Correlation = {round(corr, 3)})", fontsize=16)
plt.xlabel("Actual log prices, $y _i$", fontsize=14)
plt.ylabel("Predicted log prices, $\hat y _i$", fontsize=14)
plt.show()

# %% [markdown]
# Extended plot to actual prices (reverse transformation):
#
# - Applied exponential: np.e** y_train, np.e** results.fittedvalues.
#
# - Labeled axes: “Actual prices (10000s)” vs “Predicted prices (10000s)”.
#
# - Scatter points in blue.

# %% [markdown]
# ### 3. Interpretation & key takeaway
#
# - Strong correlation (0.78) → predictions generally close to actuals.
#
# - Many points lie near cyan perfect-fit line → low residuals.
#
# - Concern: top price bracket (~$500k properties) shows large residuals → model struggles with outliers at high end.
#
# - Matches earlier suspicion from histogram (outliers in distribution).
#
# Overall: Model performs well, but improvement needed for expensive properties.
#
# **Summary:**
#
# Residual analysis confirms the model is fairly accurate (corr ≈ 0.78), with most predictions close to actual values. Scatter plots vs perfect-fit line show good alignment, though expensive properties behave like outliers and weaken performance at the top end. Model is useful but leaves room for refinement.

# %% [markdown]
# ## 1. Residuals Analysis
#
# - Predictions vs actuals: 𝑦 ≠ 𝑦^. The difference 𝑦 − 𝑦^ is the residual.
#
# - Residuals tell us how wrong the model is. If residuals are random, model assumptions hold. If patterns appear, model is missing information.
#
# Assumptions checked with residuals:
#
# - Linearity (linear model fits the data reasonably well).
#
# - Residuals should be random, centered around zero, symmetric.
#
# - Ideally, residuals are normally distributed (mean ≈ 0, skew ≈ 0).
#
# Problematic residual plots:
#
# - Cone shape → variance grows with predictions.
#
# - Curved/parabolic pattern → model missing non-linear relation.
#
# - Clusters → missing features/interactions.
#
# - Outliers / heavy tails → unusual points or data collection issues.
#
# A “healthy” residual plot looks like a random cloud, centered around 0, symmetric.

# %% [markdown]
# ### 2. Code & Implementation

# %%
# Residuals & Residual Plots
# Modified model: log prices, simplified (drop a column)

# Calculate residuals
# residuals = y_train - results.fittedvalues
residuals = results3.resid   # built-in

# Check mean & skew
resid_mean = round(results3.resid.mean(), 3)
resid_skew = round(results3.resid.skew(), 3)

# Plot Actual vs Predicted
plt.figure(figsize=(12,8))
plt.scatter(y_train, results3.fittedvalues, c='navy', alpha=0.6)
plt.plot(y_train, y_train, c='cyan')
plt.xlabel("Actual log prices $y_i$", fontsize=14)
plt.ylabel("Predicted log prices $\hat y_i$", fontsize=14)
plt.title(f"Actual vs Predicted log prices (Corr {corr})", fontsize=17)
plt.show()

# Residuals vs Predicted
plt.figure(figsize=(12,8))
plt.scatter(results3.fittedvalues, results3.resid, c='navy', alpha=0.6)
plt.xlabel("Predicted log prices $\hat y_i$", fontsize=14)
plt.ylabel("Residuals $y_i - \hat y_i$", fontsize=14)
plt.title("Residuals vs Fitted Values")
plt.show()

# Distribution of residuals
sns.displot(results3.resid, color='#008080', height= 10, kde= True)  # teal
plt.title(f"Residuals (Mean {resid_mean}, Skew {resid_skew})")
plt.show()

# %%
X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(feat, tgt, test_size=0.2)

X_incl_const = sm.add_constant(X_train_org)
model = sm.OLS(y_train_org, X_incl_const) # OLS = Ordinary Least Squares
results = model.fit()

# Residuals vs Predicted ORG model
plt.figure(figsize=(12,8))
plt.scatter(results.fittedvalues, results.resid, c='navy', alpha=0.6)
plt.xlabel("Predicted prices $\hat y_i$", fontsize=14)
plt.ylabel("Residuals $y_i - \hat y_i$", fontsize=14)
plt.title("Residuals vs Fitted Values")
plt.show()

# Distribution of residuals of ORG model
resid_mean_org = round(results.resid.mean(), 3)
resid_skew_org = round(results.resid.skew(), 3)

sns.displot(results.resid, color='#008080', height= 10, kde= True)  # teal
plt.title(f"Residuals (Mean {resid_mean_org}, Skew {resid_skew_org})")
plt.show()

# %% [markdown]
# ### 3. Observations / Takeaways
#
# - Actual vs Predicted: correlation ~0.78, many points close to cyan line, but high-price homes ($500k) deviate strongly.
#
# - Residuals vs Predicted: residuals mostly random, centered around 0, symmetric → good. But expensive homes create a visible pattern (line-up).
#
# Distribution of Residuals:
#
# - Mean ≈ 0 (by design).
#
# - Skew ≈ -0.044 (close to normal).
#
# - Shape: symmetric but with a lil long tails → more extreme residuals than normal distribution.
#
# Comparison with original model (no log transform, all features):
#
# - Residuals show sinking wedge → log transform was helpful.
#
# - Skew ~1.069 → non-normal, confirms transformation improved model fit.
#
# - Omitting key variables creates visible clusters (banding), showing missing info leaks into residuals.
#
# **Summary:**
#
# Residual analysis shows that our simplified log-price model performs well: residuals are random, symmetric, and nearly normal. The log transformation improved normality and reduced skew, while omitted features caused visible clustering. High-value homes remain outliers, pointing to limitations in our model.

# %% [markdown]
# ## 1. Predictions, MSE & R-Squared
#
# Mean Squared Error (MSE):
#
# - Formula: 
#
# $𝑀𝑆𝐸 = \frac{1}{n} ∑(𝑦 − \hat 𝑦)^2$
#
# - Equivalent to squared residuals averaged → penalizes large errors heavily.
#
# - Sensitive to outliers because squaring magnifies big residuals.
#
# R-Squared:
#
# - Relative measure of fit, unitless, always between 0–1.
#
# - Explains proportion of variance captured by the model.
#
# Comparison:
#
# - R-squared → relative, unitless.
#
# - MSE → absolute, depends on units of target variable (e.g., $10000s).
#
# Model Comparison Example:
#
# - Reduced Log Model → highest R², lower MSE.
#
# - Full Normal Price Model → lower R², higher MSE (~0.524, meaning $52.4k average squared error).
#
# - Omitted Variable Model → worst R² and high error, clear sign of missing info.
#
# Prediction ranges:
#
# - Real-world estimators (Zoopla/Zillow) provide a range (estimate ± margin).
#
# - Range width reflects confidence, tied to residuals & MSE.

# %% [markdown]
# ### 2. Code & Implementation

# %%
# Mean Squared Error & R-squared with Statsmodels

# Reduced log model
reduced_log_mse = round(results3.mse_resid, 3)
reduced_log_rsquared = round(results3.rsquared, 3)

# Full model (normal prices, all features)
normal_mse = round(results.mse_resid, 3)
normal_rsquared = round(results.rsquared, 3)

# Omitted variable model (log prices, missing features)
omitted_var_mse = round(results2.mse_resid, 3)
omitted_var_rsquared = round(results2.rsquared, 3)

# Compare in DataFrame
comparison = pd.DataFrame({"R-Squared": [reduced_log_rsquared, normal_rsquared, 
            omitted_var_rsquared], "MSE": [reduced_log_mse, normal_mse, omitted_var_mse]},
            index=["Reduced Log Model", "Normal Price Model", "Omitted Var Model"])
print(comparison)

# %% [markdown]
# ### 3. Observations / Takeaways
#
# - R² is consistent and comparable across models, MSE is scale-dependent.
#
# - Log transformation improved residual distribution, gave higher R², and more meaningful MSE.
#
# - Models missing key variables perform worst → banding/clustering in residuals.
#
# - Professional estimators (Zoopla/Zillow) provide estimate + range, showing prediction confidence.
#
# Our future goal: provide not only a point prediction, but also a confidence interval, tied to residual spread and MSE.
#
# **Summary:**
#
# MSE and R-squared complement each other. R² shows relative explanatory power, while MSE gives absolute average error in the units of the target. Simplified log model performs best, omitted variable model worst. Real-world practice mirrors this — predictions are always given with a confidence range, which we’ll connect back to residuals and MSE in the next step.

# %% [markdown]
# ## 1. Prediction Intervals & RMSE
#
# Prediction needs two parts:
#
# - Estimated value $(\hat 𝑦)$ & A range (prediction interval) around it.
#
# Normal distribution connection:
#
# - ~68% of values lie within ±1 standard deviation (σ).
#
# - ~95% of values lie within ±2σ.
#
# Residuals: Distribution of residuals tells us the variance of predictions.
#
# RMSE (Root Mean Squared Error):
#
# - RMSE = $\sqrt{MSE}$
#
# - Represents the standard deviation of residuals.
#
# - Used to build prediction intervals around $\hat y$
#
# 95% Prediction Interval:
#
# - Upper bound: $\hat y + 2 RMSE$
#
# - Lower bound: $\hat y - 2 RMSE$
#
# Key Contrast:
#
# - R² measures fit quality (variance explained).
#
# - RMSE measures prediction accuracy and gives interpretable error ranges.

# %% [markdown]
# ### 2. Code & Implementation

# %%
# RMSE calculation
rmse = np.sqrt(reduced_log_mse)
print("1 S.D. in log prices: ", rmse)
print("2 S.D. in log prices: ", 2 * rmse)

# Example prediction: y_hat = $300,000
upper_bound = np.log(3) + rmse
lower_bound = np.log(3) - rmse

# Convert back to dollar values
upper_price = np.e**upper_bound * 100000
lower_price = np.e**lower_bound * 100000

print("Upper bound (68% PI): ", upper_bound)
print("Lower bound (68% PI): ", lower_bound)

print("Upper price (68% PI): $", upper_price)
print("Lower price (68% PI): $", lower_price)

# %% [markdown]
# ### 3. Observations / Takeaways
#
# - RMSE connects residuals to prediction intervals → direct measure of prediction error.
#
# - Prediction intervals communicate uncertainty, just like professional tools (Zoopla/Zillow).
#
# - Correct order of applying logs and RMSE adjustment is crucial — wrong order gives wrong bounds.
#
# - Example: For $30,000 estimate, 68% PI gave bounds roughly between $210k and $427k.
#
# - R² ≠ prediction accuracy → only RMSE-based intervals reveal how good/bad predictions are in practice.
#
# **Summary:**
#
# Prediction requires not only a point estimate but also a confidence range. By using the residual distribution and RMSE, we can calculate intervals (e.g., ±σ for 68% coverage). This bridges model fit with real-world predictive usefulness, highlighting why RMSE is often preferred when evaluating predictive power.
