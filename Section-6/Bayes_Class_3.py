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
