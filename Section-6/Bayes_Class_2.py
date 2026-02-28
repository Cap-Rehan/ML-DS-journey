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
# # THIS IS PT 2

# %%
from os import walk
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


# %%
def email_body_generator(path):

    for root, dirnames, filenames in walk(path):
        for file_name in filenames:

            file_path = join(root, file_name)
            stream = open(file_path, encoding= 'latin-1')
            is_body = False
            lines = []

            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == "\n":
                    is_body = True

            stream.close()
            email_body = "\n".join(lines)

            yield file_name, email_body


# %%
def df_from_directory(path, classification):
    rows = []
    row_names = []

    for file_name, email_body in email_body_generator(path):
        rows.append({
            'MESSAGE': email_body,
            'CATEGORY': classification
        })
        row_names.append(file_name)

    return pd.DataFrame(rows, index=row_names)


# %%
SPAM_1_PATH = "SpamData/01_Processing/spam_assassin_corpus/spam_1"
SPAM_2_PATH = "SpamData/01_Processing/spam_assassin_corpus/spam_2"
EASY_NONSPAM_1_PATH = "SpamData/01_Processing/spam_assassin_corpus/easy_ham_1"
EASY_NONSPAM_2_PATH = "SpamData/01_Processing/spam_assassin_corpus/easy_ham_2"
SPAM_CAT = 1
HAM_CAT = 0

spam_emails = df_from_directory(SPAM_1_PATH, 1)
spam_emails = pd.concat([spam_emails, df_from_directory(SPAM_2_PATH, SPAM_CAT)])
ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = pd.concat([ham_emails, df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT)])

# %%
data = pd.concat([spam_emails, ham_emails])

# %%
data.drop('cmds', inplace=True)
data['MAIL_ID'] = range(1, 5797)
data['FILE_NAME'] = data.index
data.set_index('MAIL_ID', inplace=True)

# %% [markdown]
# ## 1. Build Vocabulary (Top 5000 Words)
#
# Key terms: stemming, flattening, value_counts, VOCAB_SIZE

# %%
VOCAB_SIZE= 3000


# %%
# def clean_msg_no_html(message):
#     soup = BeautifulSoup(message, 'lxml')
#     cleaned_text = soup.get_text()
#     print(cleaned_text[:500])
#     return []

def clean_msg_no_html(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):

    soup = BeautifulSoup(message, 'lxml')
    cleaned_text = soup.get_text()
    
    words = word_tokenize(cleaned_text.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
#             filtered_words.append(word) 
    
    return filtered_words


# %%
# there were errors due to html.parser so we shifted to lxml

# for idx, msg in data.MESSAGE.items():
#     try:
#         clean_msg_no_html(msg)
#     except Exception as e:
#         print("FAILED AT MAIL_ID:", idx)
#         print("ERROR:", e)
#         break

# %%
stemmed_nested_list = data.MESSAGE.apply(clean_msg_no_html)

# %%
stemmed_nested_list

# %%
flattened_list = [item for sublist in stemmed_nested_list for item in sublist]

# %%
len(flattened_list)

# %%
frequencies = pd.Series(flattened_list).value_counts()
frequencies

# %%
frequencies[:VOCAB_SIZE]

# %% [markdown]
# ### 2. Create Vocabulary DataFrame with WORD_ID
#
# Key terms: index, WORD_ID, dataframe construction

# %%
word_ids = list(range(VOCAB_SIZE))
vocab = pd.DataFrame({"VOCAB_WORD": frequencies[:VOCAB_SIZE].index.values})

# %%
vocab.index = word_ids
vocab.index.name = "WORD_ID"

# %%
vocab.head()

# %% [markdown]
# ### 3. Save Vocabulary to CSV
#
# Key terms: to_csv, index_label, header

# %%
WORD_ID_FILE = 'word-by-ids.csv'

# vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header= vocab.VOCAB_WORD.name)

# %% [markdown]
# **Summary**
#
# In this lesson, you generated the vocabulary used by the Naive Bayes classifier by extracting stemmed words from all email bodies, counting their frequencies, and selecting the top 2500 most common terms. You converted this vocabulary into a structured pandas DataFrame, assigned each word a stable WORD_ID, and saved the result as a CSV file for reuse. This vocabulary now serves as the foundation for mapping emails into numerical feature vectors in the upcoming classifier training stage.

# %% [markdown]
# ## 1. Full Matrix vs Sparse Matrix
#
# Key terms: full matrix, sparse matrix, occurrence, label

# %%
# Sparse matrix: Remove rows where OCCURRENCE = 0 → only keep words that actually appear.

# %% [markdown]
# ### 2. Convert Nested Lists → Word Columns DataFrame
#
# Key terms: series of lists, to_list(), from_records()

# %%
list_of_lists = stemmed_nested_list.tolist()
word_columns_df = pd.DataFrame.from_records(list_of_lists)
print(word_columns_df.shape)
word_columns_df.head()

# %% [markdown]
# ### 3. Shuffle & Split (Train/Test)
#
# Key terms: train_test_split, test_size, random_state

# %%
X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY, train_size= 0.75, random_state=41)

X_train.index.name = y_train.index.name = "DOC_ID"

# %%
(X_train.shape[0])/(word_columns_df.shape[0])

# %% [markdown]
# **Summary**
#
# In this stage, you moved from conceptual understanding (full vs sparse matrix) toward preparing data for feature engineering. A full matrix would contain 2500 entries per email, most of which would be zeros. A sparse matrix removes these zero-occurrence rows to drastically reduce size and improve efficiency. You then converted your stemmed nested list into a structured DataFrame, shuffled the dataset, and split it into training and testing sets using train_test_split with a fixed seed for reproducibility.

# %% [markdown]
# ## 1. Word Lookup via Index
#
# Key terms: pd.Index, get_loc(), reverse lookup

# %%
vocab.VOCAB_WORD

# %%
word_index = pd.Index(vocab.VOCAB_WORD)

# %%
print(word_index[77])
word_index.get_loc("line")


# %% [markdown]
# ### 2. Create Sparse Matrix Function
#
# Key terms: nested loops, set lookup, dictionary rows

# %%
def make_sparse_matrix(dframe, indexed_words, labels):

    rows = dframe.shape[0]
    cols = dframe.shape[1]

    words_set = set(indexed_words)
    dict_list = []

    for i in range(rows):
        for j in range(cols):
            word = dframe.iat[i, j]

            if word in words_set:
                doc_id = dframe.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.iloc[i]

                item = {'DOC_ID': doc_id, 'WORD_ID': word_id, 'CATEGORY': category, 'OCCURENCE': 1}
                dict_list.append(item)
    return pd.DataFrame(dict_list)


# %% [markdown]
# ### 3. Generate Sparse Training Matrix
#
# Key terms: runtime cost, vocabulary size impact

# %%
import time

start = time.perf_counter()

sparse_train_dframe = make_sparse_matrix(X_train, word_index, y_train)

end = time.perf_counter()
print(f"Time taken: {end - start:.6f} seconds")

# %%
print(sparse_train_dframe.shape)
sparse_train_dframe.head()

# %% [markdown]
# **Summary**
#
# You built a sparse matrix by iterating over every word in the training dataset, checking whether it belongs to the top 2500 vocabulary words, and mapping it to its corresponding WORD_ID. Each valid word occurrence became one row containing DOC_ID, WORD_ID, LABEL, and OCCURENCE = 1. Because every occurrence is stored separately, the resulting matrix contains hundreds of thousands of rows. The computational cost is high due to nested loops and large vocabulary size.

# %% [markdown]
# ## 1. Combine Duplicate Word Occurrences
#
# Key terms: groupby, sum, aggregation

# %%
train_grouped = sparse_train_dframe.groupby(['DOC_ID', 'WORD_ID', 'CATEGORY', 'OCCURENCE']).sum().reset_index()

print(train_grouped.shape)
train_grouped.head()

# %% [markdown]
# ### 2. Save Training Sparse Matrix (.txt)
#
# Key terms: np.savetxt, integer format

# %%
TRAINING_DATA_FILE = 'SpamData/02_Training/train-data.txt'

# np.savetxt(TRAINING_DATA_FILE, train_grouped, fmt= "%d")

# %% [markdown]
# ### 3. Create & Save Test Sparse Matrix
#
# Key terms: reuse function, identical pipeline

# %%
sparse_test_dframe = make_sparse_matrix(X_test, word_index, y_test)
test_grouped = sparse_test_dframe.groupby(['DOC_ID', 'WORD_ID', 'CATEGORY', 'OCCURENCE']).sum().reset_index()

# %%
TEST_DATA_FILE = "SpamData/02_Training/test-data.txt"
# np.savetxt(TEST_DATA_FILE, test_grouped, fmt= "%d")

# %%
test_grouped.shape

# %% [markdown]
# **Summary**
#
# You reduced redundancy in the sparse matrix by grouping identical (DOC_ID, WORD_ID, LABEL) combinations and summing their occurrences. This significantly shrank the dataset while preserving word frequency information. You then exported both training and test sparse matrices as .txt files using NumPy’s savetxt, producing clean numeric datasets ready for Naive Bayes training. At this stage, the data pipeline is complete: raw emails → cleaned tokens → vocabulary mapping → sparse feature matrix → persisted training/test data.

# %%
train_doc_ids = set(train_grouped.DOC_ID)
test_doc_ids = set(test_grouped.DOC_ID)

# %%
# unique IDs saved vs Original dataset

print(len(train_doc_ids))
print(len(test_doc_ids))

print(X_train.shape[0])
print(X_test.shape[0])

# %% [markdown]
# ### 2. Which Emails Were Excluded?
#
# Key terms: set difference, membership

# %%
original_train_ids = set(X_train.index)
missing_train_ids = original_train_ids - train_doc_ids

original_test_ids = set(X_test.index)
missing_test_ids = original_test_ids - test_doc_ids

# %%
print(missing_train_ids)
print(missing_test_ids)

# %%
print(data.MESSAGE[325])
print(clean_msg_no_html(data.at[325, "MESSAGE"]))

# %% [markdown]
# ### 3. Why This Happens
#
# Key terms: vocabulary filter, zero-feature document
#
# Core filter in sparse builder:
# -> " _if_ word _in_ word_set: "
#
# If:
# - Email contains no top-2500 vocabulary words
# - OR cleaning removes all content
#
# → No sparse rows created
# → Email excluded
#
# These are effectively zero-feature documents.

# %% [markdown]
# **Summary**
#
# You verified how many emails actually survived the sparse matrix transformation and discovered that some emails were excluded. By comparing document IDs using Python sets, you identified missing emails and traced the cause. Some messages consisted entirely of encrypted blocks or malformed text, while others were dominated by HTML content that was stripped away during preprocessing. Because the sparse matrix only records words from the top 2500 vocabulary, any email with zero valid tokens produced no entries and therefore vanished from the dataset. This highlights an important preprocessing subtlety: aggressive filtering and vocabulary limits can silently remove documents. With preprocessing complete and text fully converted into numerical representations, the dataset is now ready for Naive Bayes training.
