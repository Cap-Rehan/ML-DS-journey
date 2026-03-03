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

# %% [markdown]
# # Imports, Constants

# %%
import numpy as np
import pandas as pd

# %%
TRAINING_DATA_FILE = "SpamData/02_Training/train-data.txt"
TEST_DATA_FILE = "SpamData/02_Training/test-data.txt"
VOCAB_SIZE = 3000

FULL_TRAIN_MATRIX = "SpamData/02_Training/train-matrix.csv"
FULL_TEST_MATRIX = "SpamData/02_Training/test-matrix.csv"

PROB_TOKEN_SPAM = "SpamData/03_Testing/prob-spam.txt"
PROB_TOKEN_HAM = "SpamData/03_Testing/prob-ham.txt"
PROB_TOKEN_ALL = "SpamData/03_Testing/prob-tokens.txt"

TEST_FEATURES= "SpamData/03_Testing/X-test.txt"
TEST_TARGET= "SpamData/03_Testing/y-test.txt"

# %%
# Load Sparse Data from TXT

sparse_train = np.loadtxt(TRAINING_DATA_FILE, delimiter= " ", dtype= int)
sparse_test = np.loadtxt(TEST_DATA_FILE, delimiter= " ", dtype= int)

# %%
print(sparse_train[:5])
print(sparse_train[-5:])
print(sparse_test[:5])
print(sparse_test[-5:])

# %%
# Inspect Dataset Size
print(sparse_train.shape[0])
print(sparse_test.shape[0])

print("Unique training emails:\n", np.unique(sparse_train[:, 0]).size)
print("Unique testing emails:\n", np.unique(sparse_test[:, 0]).size)

# %% [markdown]
# **Summary**
#
# You initialized a new training notebook, imported essential libraries, and loaded the preprocessed sparse matrices from text files into NumPy arrays using np.loadtxt. Each row represents a word occurrence defined by DOC_ID, WORD_ID, LABEL, and OCCURENCE. You verified successful loading by inspecting slices and analyzing dataset dimensions. Using .shape and np.unique, you confirmed both total row counts and the number of unique emails present in the training and test sets. At this stage, the data exists purely in numeric form and is ready to be transformed from sparse format into the full matrix representation required for Naive Bayes training.

# %% [markdown]
# ## Creating DataFrame

# %%
column_names = ['DOC_ID', 'CATEGORY'] + list(range(VOCAB_SIZE))
print(column_names[-5:])
column_names[:5]

# %%
index_name = np.unique(sparse_train[:, 0])
print(index_name[-5:])
index_name[:5]

# %%
full_train_data = pd.DataFrame(columns= column_names, index= index_name)

# %%
full_train_data.fillna(value= 0, inplace= True)
full_train_data


# %% [markdown]
# ## Create a full matrix from a Sparse Matrix

# %%
def make_full_matrix(sparse_matrix, no_words, doc_idx= 0, word_idx= 1, label_idx= 2, freq_idx= 3):
    COLUMNS= ['DOC_ID', 'CATEGORY'] + list(range(no_words))
    INDEX= np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(columns= COLUMNS, index= INDEX)
    full_matrix.fillna(value= 0, inplace= True)

    for i in range(sparse_matrix.shape[0]):
        doc_no= sparse_matrix[i][doc_idx]
        word_no= sparse_matrix[i][word_idx]
        label= sparse_matrix[i][label_idx]
        freq= sparse_matrix[i][freq_idx]

        full_matrix.at[doc_no, 'DOC_ID']= doc_no
        full_matrix.at[doc_no, 'CATEGORY']= label
        full_matrix.at[doc_no, word_no]= freq

    full_matrix.set_index('DOC_ID', inplace= True)
    
    return full_matrix


# %%
full_train_df = make_full_matrix(sparse_train, VOCAB_SIZE)

# %%
full_test_df = make_full_matrix(sparse_test, VOCAB_SIZE)

# %%
full_train_df.to_csv(FULL_TRAIN_MATRIX)
full_test_df.to_csv(FULL_TEST_MATRIX)

# %% [markdown]
# ## Training the Naive Bayes Model

# %% [markdown]
# ### Calculating Probabilities

# %%
print(full_train_df['CATEGORY'].value_counts())
print(full_train_df.CATEGORY.size)
print(len(full_train_df))

# %%
# probability of spam
prob_spam = full_train_df['CATEGORY'].value_counts()[1]/len(full_train_df)
prob_spam

# %%
# number of words / tokens
full_train_features = full_train_df.loc[:, 0:]
full_train_features.head()

# %%
words_per_email = full_train_features.sum(axis= 1)
words_per_email.tail()

# %%
total_wc = words_per_email.sum()
print(total_wc)

# %%
# number of tokens in spam and ham
word_per_spam = (full_train_df.loc[full_train_df.CATEGORY == 1, 0:]).sum(axis= 1)
word_per_ham = (full_train_df.loc[full_train_df.CATEGORY == 0, 0:]).sum(axis= 1)

# %%
spam_wc = word_per_spam.sum()
print(spam_wc)
ham_wc = word_per_ham.sum()
print(ham_wc)

# %%
print(total_wc - spam_wc - ham_wc)

# %%
# which emails tend to be longer
print(word_per_spam.mean())
print(word_per_ham.mean())

# %%
# Summing all the tokens occuring in spam
full_train_df

# %%
spam_train_tokens = full_train_features.loc[full_train_df.CATEGORY == 1]
spam_train_tokens.tail()

# %%
spam_train_tokens.shape

# %%
summed_spam_tokens= spam_train_tokens.sum(axis= 0) + 1 # laplace smoothing
summed_spam_tokens

# %%
ham_train_tokens= full_train_features.loc[full_train_df.CATEGORY == 0]
summed_ham_tokens= ham_train_tokens.sum(axis= 0) + 1
summed_ham_tokens

# %%
(summed_spam_tokens == 0).any()

# %% [markdown]
# ### $ P \left( Token \mid Spam \right) $

# %%
prob_tokens_spam = summed_spam_tokens / (spam_wc + VOCAB_SIZE)
prob_tokens_spam

# %%
print(prob_tokens_spam.sum())

# %% [markdown]
# ### $ P \left( Token \mid Ham \right) $

# %%
prob_tokens_ham = summed_ham_tokens / (ham_wc + VOCAB_SIZE)
print(prob_tokens_ham.sum())

# %% [markdown]
# ### $ P\left( Token \right) $

# %%
prob_tokens_all = full_train_features.sum(axis= 0) / total_wc
print(prob_tokens_all.sum())

# %% [markdown]
# ### Saving the trained model

# %%
np.savetxt(PROB_TOKEN_SPAM, prob_tokens_spam)
np.savetxt(PROB_TOKEN_HAM, prob_tokens_ham)
np.savetxt(PROB_TOKEN_ALL, prob_tokens_all)

# %% [markdown]
# ### Preparing Testing Files

# %%
X_test= full_test_df.loc[:, full_test_df.columns != 'CATEGORY']
y_test= full_test_df.CATEGORY

# %%
y_test

# %%
np.savetxt(TEST_FEATURES, X_test)
np.savetxt(TEST_TARGET, y_test)
