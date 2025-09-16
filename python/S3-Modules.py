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
# ## Python Modules & Packages

# %% [markdown]
# A module = any .py file containing Python code.
#
# A package = a directory of modules (e.g., the entire pandas folder is a package).
#
# Importing loads that code into your namespace so you can reuse it.

# %%
import math
import life

# %%
# Brings in the entire module as an object.
# Access members via dot notation: 
print(math.pi)
print(life.name)

# %%
# Aliasing with 'as' :
import life as lf

# %%
lf.movie

# %%
# Selective import with 'from … import ...'
from life import age
age # no need for dot notation here

# %% [markdown]
# Copies the specified attribute into your namespace as a standalone variable/class.
#
# Use it directly: LinearRegression(), theAnswer.
#
# Unlike import, you don’t reference the module name again.
#
# - Why this matters
#
# Reusability: Leverage thousands of lines of community‑built code (e.g., scikit‑learn, pandas) rather than reinventing the wheel.
#
# Organization: Namespacing via modules/packages keeps code clean.
#
# Convention: Common aliases (pd, plt) speed up coding and readability across projects

# %% [markdown]
# **Key Terms / Questions**
# - Module (.py file)
# - Package (folder of modules)
# - import
# - as (alias)
# - from … import
# - Dot notation
# - Namespace
# - Q: When is from … import preferred over import … as?
# - Q: How do you inspect a module’s contents (e.g., list of functions)?

# %% [markdown]
# **Summary**
#
# Python’s import system lets you tap into both your own .py scripts and vast open‑source libraries. Use import module (and optionally as for an alias) to access everything under a module, or from module import name to pull specific items directly into your namespace—powering rapid, organized, and readable code reuse.
