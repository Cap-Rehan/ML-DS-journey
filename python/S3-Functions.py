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
# ## What we know about Functions 

# %% [markdown]
# - Anatomy:
#
# Header: def keyword + function name + () + :
#
# Body: indented block (4 spaces) of statements executed when called
#
# Indentation is syntactic: groups body under header; missing or extra spaces lead to errors.
# - Distinguishing
#
# Functions: parentheses after a name indicate a call; plt.show()
#
# Variable/accessor: math.pi, life.theAnswer (no ())
# - Benefits:
#
# Abstraction & Reuse: hide complex steps behind a simple name
#
# Maintainability: change logic in one place, call everywhere
#
# Readability: descriptive names convey intent

# %% [markdown] vscode={"languageId": "plaintext"}
# - Parameters vs. Arguments
#
# Parameter = placeholder name in function header (e.g., amount, destination).
#
# Argument = actual value passed when calling the function (e.g., 'five', 'store')

# %%
def fill_the_fridge(amount):
    print("Buy " + amount + " cartons")


# %% [markdown]
# Must supply exactly the parameters declared, or you get a TypeError (“missing required argument”).
#
# - Positional arguments:
#
# Passed by order: milk_mission('twenty', 'store') → amount='twenty', destination='store'.
#
# Swapping order leads to semantically wrong behavior.
#
# - Keyword (named) arguments:
#
# Passed by name=value: milk_mission(destination='store', amount='twenty').
#
# Order doesn’t matter when every argument is named.
#
# Mixing positional and keyword allowed—positional must come first.
#
# - Function nesting: you can call one function inside another (e.g., using print() within your own function).
#
# - Plus operator is overloaded:
#
# + on numbers performs addition.
#
# + on strings performs concatenation.

# %% [markdown]
# **Key Terms / Questions**
# - Parameter (function definition placeholder)
# - Argument (value passed in call)
# - Positional argument
# - Keyword argument
# - TypeError (missing/duplicate args)
# - Argument overloading (+)
# - Q: How do you provide default values for parameters?
# - Q: When should you prefer keyword over positional arguments?

# %% [markdown]
# ### Return Values: functions can output data via the return statement.

# %%
def get_milk(money):
    litres = money / 1.15
    return litres


# %%
# When Python executes return, it exits the function immediately and
# hands back the following value.

# Using a return: capture the output in a variable:
bought = get_milk(20.5)
# bought now holds the returned value (≈17.8).

# %% [markdown]
# You can inline the calculation directly in the return (no intermediate variable needed).
#
# - Dynamic Typing: Python’s operators work across types—e.g., times('Ni', 4) repeats the string ('NiNiNiNi').
#
# Implication: functions must handle or guard against unexpected types when necessary.
#
# - Function Flavors:
#
# No inputs, no outputs (e.g., show())
#
# Inputs only (e.g., fill_the_fridge(amount))
#
# Input + output (e.g., times(x, y) or get_milk(money))
#
# - Design & Reuse:
#
# Functions break complex tasks into manageable pieces.
#
# They enable code reuse across different inputs (e.g., running regressions on many X/Y pairs).
#
# At a larger scale, modules and packages group related functions/files to tame complexity.
#
# Pythonic philosophy: “Simple is better than complex; complex is better than complicated.” (import this)

# %% [markdown]
# **Key Terms / Questions**
# - return
# - Return value
# - Dynamic typing
# - Operator overloading
# - Code reuse
# - Taming complexity
# - Q: How can you enforce type checks inside a function?
# - Q: When should a function return None vs. a meaningful value?
