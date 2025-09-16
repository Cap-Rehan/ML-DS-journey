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
# ## Everything is an object

# %% [markdown]
# numbers, strings, modules (math), DataFrames (pd.DataFrame), custom scripts (life), etc.

# %% [markdown]
# - Object state & attributes:
#
# Objects carry attributes that describe their current state (e.g., lightning_mcqueen.fuel_level, regr.intercept_, math.pi).
#
# Access via dot notation: object.attribute.
#
# - Methods = functions bound to objects:
#
# Invoke behaviors that may change state or produce side effects (e.g., lightning_mcqueen.drive(), pd.read_csv(), plt.show()).
#
# Syntax: object.method(args…).
#
# - Defining methods in your module:
#
# Add def quote_maximus(): … inside life.py.
#
# Escape special characters in strings with backslashes (\').

# %%
# import and call
import life as hitchhikersGuide
hitchhikersGuide.quote_maximus()

# %% [markdown]
# - Variables vs. objects:
#
# A variable is a name (“box”) referencing an object in memory.
#
# You can reassign the variable to different objects (e.g., myAge = 32 → ‘200’ → 20.53).
#
# The object’s type (and behavior) lives with the object, not the variable.
#
# - Dynamic typing & behavior:
#
# Operators/methods adapt to object types (3 * 4 → 12; 'Ni' * 4 → 'NiNiNiNi').
#
# Python emphasizes what an object does (its methods) over its static type.
#
# - Building on prior examples:
#
# Regression: X = pd.read_csv(... ), regression.fit(X, y), plt.scatter(X, y), slope = regression.coef_.
#
# Calculator challenge: implement square_root(x) in life.py using math.sqrt

# %% [markdown]
# **Key Terms / Questions**
# - Object
# - Attribute
# - Method
# - Dot notation
# - Variable vs. object
# - Dynamic typing
# - Escaping strings (\')
# - Q: How do you list an object’s methods/attributes?
# - Q: When should you encapsulate functionality in a method vs. a standalone function?

# %% [markdown]
# **Summary**
#
# In Python, all values are objects with attributes (state) and methods (behaviors), accessed via object.attribute and object.method(). Variables merely reference objects, whose types and available operations are determined at runtime—enabling flexible, expressive code design.
