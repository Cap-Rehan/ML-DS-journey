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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score, precision_score, f1_score

# %%
DATA_JSON_FILE = 'SpamData/01_Processing/email-text-data.json'

# %%
data = pd.read_json(DATA_JSON_FILE)
data

# %%
vectorizer = CountVectorizer(stop_words='english')

# %%
features = vectorizer.fit_transform(data.MESSAGE)

# %%
features.shape

# %%
vectorizer.vocabulary_

# %%
X_train, X_test, y_train, y_test = train_test_split(features, data.CATEGORY, train_size= 0.7)

# %%
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# %%
prediction = classifier.predict(X_test)
prediction.shape[0]

# %%
true_pos = (prediction == 1) & (y_test == 1)
false_pos = (prediction == 1) & (y_test == 0)
true_neg = (prediction == 0) & (y_test == 0)
false_neg = (prediction == 0) & (y_test == 1)

# %%
nr_correct = true_pos.sum() + true_neg.sum()
nr_incorrect = false_pos.sum() + false_neg.sum()

# %%
print('Docs classified correctly:', nr_correct)
print('Docs classified incorrectly:', nr_incorrect)

# %%
# accuracy = (nr_correct) / (prediction.shape[0])
# print(f"The accuracy is {accuracy:.2%}")
classifier.score(X_test, y_test)

# %%
print(round(recall_score(y_test, prediction)*100, 2))
print(round(precision_score(y_test, prediction)*100, 2))
print(round(f1_score(y_test, prediction)*100, 2))

# %%
emails = [
"""Hey,
Are we still meeting tomorrow for the project discussion?
Let me know what time works for you.""",

"""Congratulations!
You have won a $500 Amazon gift card.
Click the link below to claim your reward now.""",

"""Hi,
Please find the attached notes from today's lecture.
Let me know if anything is missing.""",

"""Limited time offer!
Buy 1 get 3 free on all supplements.
Order now before the deal expires tonight.""",

"""Reminder:
Your electricity bill is due tomorrow.
Please make the payment to avoid late fees.""",

"""URGENT!
Your account has been compromised.
Verify your details immediately to restore access.""",

"""Hey,
I uploaded the assignment solutions to the shared folder.
Check them before the quiz tomorrow.""",

"""You have been selected for an exclusive investment opportunity.
Earn up to 200% returns in just 7 days.
Register now to secure your spot.""",

"""Hello,
Can you send me the dataset you mentioned yesterday?
I need it for the analysis.""",

"""Act fast!
Lowest prices on premium watches.
Huge discount available only today."""
]

# %%
emails_matrix = vectorizer.transform(emails)

# %%
classifier.predict(emails_matrix)
