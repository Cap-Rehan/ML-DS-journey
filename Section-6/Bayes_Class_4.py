# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: ml_env (3.14.3)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
PROB_TOKEN_SPAM = "SpamData/03_Testing/prob-spam.txt"
PROB_TOKEN_HAM = "SpamData/03_Testing/prob-ham.txt"
PROB_TOKEN_ALL = "SpamData/03_Testing/prob-tokens.txt"

TEST_FEATURES= "SpamData/03_Testing/X-test.txt"
TEST_TARGET= "SpamData/03_Testing/y-test.txt"

# %%
X_test = np.loadtxt(TEST_FEATURES, delimiter= ' ')
y_test = np.loadtxt(TEST_TARGET, delimiter= ' ')

prob_spam_token = np.loadtxt(PROB_TOKEN_SPAM, delimiter= ' ')
prob_ham_token = np.loadtxt(PROB_TOKEN_HAM, delimiter= ' ')
prob_all_tokens = np.loadtxt(PROB_TOKEN_ALL, delimiter= ' ')

# %%
X_test.shape

# %% [markdown]
# ## The Dot Product

# %%
dot_product = X_test.dot(prob_spam_token)
dot_product.shape

# %% [markdown]
# ## Set the Prior
#
# ### $$ P\left( Spam \mid X \right) = \frac{P\left( X \mid Spam \right) \,P\left( Spam \right)}{P\left( X \right)}$$

# %%
prob_spam = 0.322

# %%
prob_spam_token_log = np.log(prob_spam_token)

# %%
prob_spam_token_log.shape

# %% [markdown]
# ## Joint Probability in log format

# %%
joint_log_spam = X_test.dot(prob_spam_token_log - np.log(prob_all_tokens)) + np.log(prob_spam)
joint_log_ham = X_test.dot(np.log(prob_ham_token) - np.log(prob_all_tokens)) + np.log(1 - prob_spam)

# %%
print(joint_log_ham.size)
joint_log_ham

# %% [markdown]
# ## Making Predictions
# ### Checking which joint probability is higher
# $$
# P(\text{Spam} \mid X) > P(\text{Ham} \mid X)
# $$
#
# **OR**
#
# $$
# P(\text{Spam} \mid X) < P(\text{Ham} \mid X)
# $$

# %%
prediction = joint_log_spam > joint_log_ham

# %%
print(prediction[:5])
print(y_test[:5])

# %% [markdown]
# ### Simplify, we can our calculations
#
# $$ P\left( Spam \mid X \right) = \frac{P\left( X \mid Spam \right) \,P\left( Spam \right)}{P\left( X \right)}$$
# $$ P\left( Ham \mid X \right) = \frac{P\left( X \mid Ham \right) \,P\left( Ham \right)}{P\left( X \right)}$$
#
# since we are using these probabilities to reach our final result and both of them have the same denominator, we can simply cancel it out.
#
# **NOTE**: 
#
# $$ P\left( X \mid Spam \right) \,P\left( Spam \right) \neq \frac{P\left( X \mid Spam \right) \,P\left( Spam \right)}{P\left( X \right)}$$

# %%
# joint_log_spam = X_test.dot(prob_spam_token_log - np.log(prob_all_tokens)) + np.log(prob_spam)
# joint_log_ham = X_test.dot(np.log(prob_ham_token) - np.log(prob_all_tokens)) + np.log(1 - prob_spam)

joint_log_spam = X_test.dot(prob_spam_token_log) + np.log(prob_spam)
joint_log_ham = X_test.dot(np.log(prob_ham_token)) + np.log(1 - prob_spam)

# %% [markdown]
# ## Metrics & Evaluation
# ### Accuracy

# %%
correct_docs = (y_test == prediction).sum()
print("accuracy =", correct_docs / len(y_test))

# %% [markdown]
# ## Visualizing our Results

# %%
linedata = np.linspace(-12000, 1, 1000)
yaxis_label= "P(X | Spam)"
xaxis_label= "P(X | NonSpam)"

# %%
plt.style.use("ggplot")
plt.figure(figsize= (20, 8), dpi= 300)

plt.subplot(1, 2, 1)

plt.xlabel(xaxis_label, fontsize= 14)
plt.ylabel(yaxis_label, fontsize= 14)
plt.title("SPAM v NONSPAM", fontsize= 25)

plt.xlim([-12000, -2000])
plt.ylim([-12000, -2000])

plt.scatter(joint_log_ham, joint_log_spam, alpha= 0.65, color= "navy", s= 25)
plt.plot(linedata, linedata, color= "orange", alpha= 0.8)

plt.subplot(1, 2, 2)

plt.xlabel("P(X | NonSpam)", fontsize= 14)
plt.title("CROWDED AREA", fontsize= 25)

plt.xlim([-2000, 0])
plt.ylim([-2000, 0])

plt.scatter(joint_log_ham, joint_log_spam, alpha= 0.65, color= "navy", s= 10)
plt.plot(linedata, linedata, color= "orange", alpha= 0.8)

plt.show()

# %%
sns.set_theme(style="whitegrid", context="notebook")
labels = 'Actual Category'

summary_df = pd.DataFrame({yaxis_label: joint_log_spam, xaxis_label:joint_log_ham, labels: y_test})

# %%
sns.lmplot(summary_df, x= xaxis_label, y= yaxis_label, fit_reg= False, height= 12, aspect= 1, scatter_kws= {'alpha':0.6, 's':24}, hue= labels, markers= ['o', 's'], palette= 'colorblind', legend= False)

plt.xlim([-1000, 0])
plt.ylim([-1000, 0])
plt.plot(linedata, linedata, color= "#010101")
plt.legend(['NonSpam', 'Spam', 'Decision Boundary'], loc= 'lower right', fontsize= 14)

plt.show()

# %%
