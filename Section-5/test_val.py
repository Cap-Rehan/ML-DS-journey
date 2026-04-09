# testing our valuation model on importing as a module...

import california_valuation as c_val

price, upper, lower, interval = c_val.get_dollar_estimate(6, 2, 39, -121, wide_range= False)

print(f"The price of the property is {price}")
print(f"The range is: $ {lower} to $ {upper}")
print(f"With a confidence of {interval}%")