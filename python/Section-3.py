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
# ## Just revising some Python programming concepts that are gonna be used in this course

# %% [markdown]
# ### DataFrames from Pandas

# %%
import pandas as pd

# %%
data= pd.read_csv('lsd_math_score_data.csv')

# %%
type(data)

# %%
# accessing a column; a Series object is returned
onlyMathScores = data['Avg_Math_Test_Score']

# %%
type(onlyMathScores)

# %%
cleanData = data[['LSD_ppm', 'Avg_Math_Test_Score']]
# preserves 2D structure; a DataFrame

# %%
type(cleanData)

# %%
# Adding new columns
data['Test_Subject'] = 'Jennifer Lopez'            # broadcast string

# %%
data.head()

# %%
data['High_Score'] = 100                         # broadcast number
data.head()

# %%
data['High_Score'] -= data['Avg_Math_Test_Score'] # elementwise add
data.head()

# %%
data['High_Score'] = data['High_Score']**2      # elementwise square
data.head()

# %%
# deleting columns
del data['Test_Subject']
del data['High_Score']

# %%
data.head()

# %% [markdown]
# **Key Terms / Questions**
#
# - DataFrame vs. Series
# - Tabular structure (rows & columns)
# - pd.read_csv()
# - Column selection (data['col'] → Series, data[['col1','col2']] → DataFrame)
# - Broadcasting (scalar → all rows)
# - Elementwise operations on Series
# - del keyword for column removal
# - Q: When would you prefer a Series over a one‑column DataFrame?
# - Q: How do you handle missing values in a DataFrame before analysis?

# %% [markdown]
# **Summary**
#
# Pandas DataFrames store 2D data; Series are their single‑column slices. Use data['col'] to get a Series, or data[['col1','col2']] to subset a DataFrame. You can add, compute on, and delete columns with intuitive syntax, leveraging elementwise operations on Series for rapid table manipulation.
