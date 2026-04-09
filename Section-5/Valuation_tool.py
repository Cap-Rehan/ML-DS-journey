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

# %%
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# %% [markdown]
# ## 1. Concepts / key ideas
#
# Build a simple valuation tool that uses the trained linear model and a single 1×7 feature row (one property) to return a predicted price and prediction interval.
#
# - Input row must match features order used to train the model; use log prices as the model target here.
#
# - Many features are hard for a user to specify; use reasonable defaults (feature means) and let the user override a few easy ones.
#
# - Give numeric column indices readable names (e.g. AVGRM_IDX = 0) so code is self-documenting and robust.
#
# - Train LinearRegression() on features with log(target); use regr.predict() to get fitted log-prices.
#
# - Compute MSE via mean_squared_error(y_true, y_pred) and RMSE = np.sqrt(MSE). RMSE in log-space = 1σ for residuals → use for prediction intervals.

# %% [markdown]
# ### 2. Coding

# %%
housing = fetch_california_housing()

data = pd.DataFrame(data= housing.data, columns= housing.feature_names)
features = data.drop(columns= ['Population'], axis= 1)
# features
log_prices = np.log(housing.target)
target = pd.DataFrame(data= log_prices, columns= ['Price'])
# target

# %%
# === index constants (match features.columns order) ===
# Example: if features.columns == ['MedInc','AveRooms','CHAS ','...','Latitude',...]
MEDINC_IDX = 0
AVGRM_IDX = 2
AVGBEDRM_IDX = 3
LATITUDE_IDX = 5
LONGITUDE_IDX = 6
# (define other indices if you need them explicitly)

# === default property template: population means, shaped 1*7 ===
property_stats = features.mean().values.reshape(1, -1)   # shape (1,7)

# %%
type(property_stats)
# property_stats

# %%
regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
# fitted_vals

rmse_log = np.sqrt(mean_squared_error(target, fitted_vals))


# %%
# === predict with a trained sklearn LinearRegression model ===

def get_estimated_log(AveRooms, AveBedrms, Latitude, Longitude,
                    MedInc= features['MedInc'].mean(), High_Conf= True):

    # override examples (user-specified values)
    property_stats[0, MEDINC_IDX] = MedInc
    property_stats[0, AVGRM_IDX] = AveRooms
    property_stats[0, AVGBEDRM_IDX] = AveBedrms
    property_stats[0, LATITUDE_IDX] = Latitude
    property_stats[0, LONGITUDE_IDX] = Longitude

    property_stats_df = pd.DataFrame(property_stats, columns=features.columns)
    log_est = regr.predict(property_stats_df)[0][0]

    # calc range
    if High_Conf:
        upper_bound= log_est + 2*rmse_log
        lower_bound= log_est - 2*rmse_log
        interval= 95
    else:
        upper_bound= log_est + rmse_log
        lower_bound= log_est - rmse_log
        interval= 68

    return float(log_est), float(upper_bound), float(lower_bound), interval


# %%
get_estimated_log(8, 4, Latitude= 37, Longitude= -120, High_Conf= False)

# %% [markdown]
# ### 3. Observations / takeaways
#
# - Feature order must match exactly the model’s training order (drop/column ordering mistakes break predictions).
#
# - Defaulting unknown fields to feature means yields a neutral baseline; user overrides (rooms, latitude, longitude) are the most impactful and intuitive.
#
# - The model predicts log(price); always reverse transform before presenting dollar amounts.
#
# - Prediction uncertainty comes from residuals → use RMSE (on log-scale) and reverse-transform bounds carefully (apply exp after adding/subtracting RMSE in log-space).
#
# - Limitations: some inputs (MedInc) are hard for users to pick; tool is “quick & dirty”—good for demos, not final valuation.
#
# - Always apply RMSE in log-space, add/subtract there, then exponentiate to get dollar bounds (do not exponentiate RMSE first).
#
# - Defaults = feature means give neutral baseline; expose only a few easy overrides (eg: rooms) to keep tool usable.
#
# **Summary:**
#
# Create a 1×7 property vector (defaults = feature means), give indices readable names, let users override a few intuitive fields (e.g. rooms), feed the row into the trained model to get y_hat (log-price), then exponentiate to get dollars. Add ±1/2·RMSE (in log space) and exponentiate to produce a 68%/95% prediction interval in dollars.

# %% [markdown]
# ## 1. Final Chapter
#
# - Inflation adjustment: 1990 dataset median price = $170,000; today’s California median ≈ $775,058 (Zillow). Compute scale factor, Multiply model outputs (after exponentiation and ×100000) by this factor.
#
# - Final function (get_dollar_estimate): wraps log-price prediction + inflation scaling, returns rounded estimates (nearest $100000).
#
# - Python extras learned: optional arguments with defaults, keyword vs. positional arguments, early returns for invalid inputs, docstrings for quick documentation, packaging notebook code into a reusable .py module (e.g., boston_valuation.py).

# %% [markdown]
# ### 2. Coding

# %%
# inflation scaling from zillow data
SCALE_FACTOR = 775058 / (np.median(housing.target) * 100000)


# %%
def get_dollar_estimate(rm, bedrm, lat, long, medinc= 3.8706710029069766, wide_range= True):
    """
    Estimate the price of a property in California.

    Keyword arguments:
    rm -- number of rooms
    bedrm -- number of bedrooms
    lat -- latitude
    long -- longitude
    medinc -- median income in block (default: 3.87)
    wide_range -- True: 95% CI, False: 68% CI
    """

    if rm < 1 or lat < 32 or lat > 42 or long < -125 or long > -114 or medinc <= 0:
        print("Unrealistic values. Try again.")
        return

    log_est, upper, lower, conf = get_estimated_log(rm, bedrm, lat, long, medinc, wide_range)

    dollar_est  = np.e**log_est * 100000 * SCALE_FACTOR
    dollar_hi   = np.e**upper   * 100000 * SCALE_FACTOR
    dollar_low  = np.e**lower   * 100000 * SCALE_FACTOR

    # round
    return (float(np.around(dollar_est, -3)),
            float(np.around(dollar_hi, -3)),
            float(np.around(dollar_low, -3)),
            float(conf))


# %%
get_dollar_estimate(6, 2, 39, -121, wide_range= False)

# %% [markdown]
# ### 3. Practical cautions
#
# - Predictions are sensitive to unrealistic inputs (e.g., rm=0). Add input checks to prevent nonsense results.
#
# - Always apply log transform & RMSE math in log-space, then exponentiate → scale → round.
#
# - Document functions clearly with docstrings so users understand arguments/limits.
#
# - For reuse, save as a module (california_valuation.py) and import california_valuation as c_val in new notebooks.
#
# **Summary:**
#
# The tool now adjusts 1990s regression outputs to modern dollar values by scaling with today’s median. It supports configurable inputs, validates them, provides confidence intervals, and can be packaged into a Python module for reuse.
