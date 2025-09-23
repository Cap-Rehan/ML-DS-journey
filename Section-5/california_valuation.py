import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

data = pd.DataFrame(data= housing.data, columns= housing.feature_names)
features = data.drop(columns= ['Population'], axis= 1)
# features
log_prices = np.log(housing.target)
target = pd.DataFrame(data= log_prices, columns= ['Price'])
# target

# === index constants (match features.columns order) ===
MEDINC_IDX = 0
AVGRM_IDX = 2
AVGBEDRM_IDX = 3
LATITUDE_IDX = 5
LONGITUDE_IDX = 6
property_stats = features.mean().values.reshape(1, -1)   # shape (1,7)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
# fitted_vals

rmse_log = np.sqrt(mean_squared_error(target, fitted_vals))

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
    # predicted_price = (np.e**log_est) * 100000   # dollars

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

# inflation scaling from zillow data
SCALE_FACTOR = 775058 / (np.median(housing.target) * 100000)

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