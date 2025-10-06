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
# # Starting some real work!

# %%
from os import walk
from os.path import join

import pandas as pd

# %% [markdown]
# ## 1. Notebook Setup & Constants
#
# Key terms: Markup, Constants, Relative Path, Naming Convention
#
# - Add headings in Jupyter notebook for organization.
#
# - Create a Constants section to store file paths (values that don’t change).
#
# - Naming convention: constants written in ALL_CAPS_WITH_UNDERSCORES.
#
# - Example constant for practice email file:

# %%
EXAMPLE_FILE = "SpamData/01_Processing/practice_email.txt"

# %% [markdown]
# ### 2. Opening & Reading Files
#
# Key terms: Stream/File Object, open(), read(), close()
#
# - Use open() to create a stream → returns a file object.
#
# - Read contents using .read() and store in variable:

# %%
with open(EXAMPLE_FILE, encoding="latin-1") as f:
    message = f.read()
print(message)

# %% [markdown]
# File reading workflow:
#
# - Open file → create stream
#
# - Read contents → save to variable
#
# - Close file → release resource (no need if using 'with' statement)
#
# - The .read() method returns a string (<class 'str'>).

# %% [markdown]
# ### 3. Encodings & Email Structure
#
# Key terms: Encoding, ASCII, UTF-8, Latin-1, Email Header, Email Body
#
# - Encoding: way computers translate characters into 0s and 1s.
#
# - ASCII = old system (127 characters, no accents/foreign alphabets).
#
# - UTF-8 = standard in Python 3.
#
# - Use latin-1 for consistency across systems in this project.

# %%
# checking the default encoding 
import sys
print(sys.getfilesystemencoding())

# %% [markdown]
# Email file format:
#
# - Header: sender, recipient, subject, cc, timestamps, routing info.
#
# - Body: main content of email (focus for spam classification).
#
# - Headers often hidden in email clients but still stored in raw files.

# %% [markdown]
# **Summary**
#
# - Organized notebook with headings & constants makes code cleaner.
#
# - Open–read–close cycle is the standard way to handle files in Python.
#
# - Always pay attention to encoding to avoid errors across systems.
#
# - Emails contain two parts: header (metadata) + body (content). For spam detection, we’ll ignore the header and work with the body text.

# %% [markdown]
# ## 1. Goal: Extract only the email body
#
# Key terms: Email body, header, boolean flag, list of lines
#
# - We want to modify the existing code so that it prints only the body of an email (ignoring the header).
#
# - To do this, we’ll:
#
# - Introduce a boolean variable is_body = False to track when we’ve entered the body.
#
# - Create a list lines = [] to collect body lines.

# %% [markdown]
# ### 2. Logic: Detect start of the body
#
# Key terms: Blank line, \n, condition checks
#
# - We know the email body starts after a blank line that separates the header and the body.
#
# So inside our loop:
#
# - If is_body is True, append the line to lines.
#
# - If the line is exactly "\n", switch is_body = True (that means we’ve reached the body).

# %% [markdown]
# ### 3. Output: Combine and print body
#
# Key terms: join(), readability, formatting
#
# - After looping, close the file and join the list into a single string.
#
# - Using '\n'.join(lines) makes the text clean and readable.
#
# - If we just print the list directly, it shows \n characters and commas (messy).

# %%
# coding:

is_body = False
lines = []

with open(EXAMPLE_FILE, encoding="latin-1") as f:
    for line in f:
        if is_body:
            lines.append(line)
        elif line == "\n":
            is_body = True

email_body = "\n".join(lines)
print(email_body)


# %% [markdown]
# **Summary**
#
# We built Python logic to extract the body of an email by:
#
# - Tracking whether we’re inside the body with a boolean.
#
# - Detecting the blank line that marks the start of the body.
#
# - Collecting all following lines into a list and joining them neatly for output.
#
# This works because all emails in the dataset follow the same header → blank line → body structure.

# %% [markdown]
# ## 1. Generator Functions
#
# Key terms: generator function, yield, memory efficiency, looping
#
# - A generator function is like a normal function but instead of return, it uses yield.
#
# - yield allows the function to pause and remember its state before continuing.
#
# - This is useful when working with large datasets (like 5000 emails) since it avoids loading everything into memory at once.

# %%
# code for a sample generator function

def generate_squares(N):
    for my_number in range(N):
        yield my_number**2


# %%
for i in generate_squares(11):
    print(i, end='\t')


# %% [markdown]
# ### 2. Email Body Extraction with a Generator
#
# Key terms: os.walk(), join(), nested loops, yield file_name, email_body
#
# - We need a function to loop over all emails in a folder and extract each body one at a time.
#
# The os.walk(path) function gives us:
#
# - root (directory path)
#
# - dirnames (subfolders — unused here)
#
# - filenames (all files inside the folder)
#
# For each file, we open it, extract the body (same logic as before), then yield a tuple (file_name, email_body).

# %%
def email_body_generator(path):
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            file_path = join(root, file_name)
            stream = open(file_path, encoding="latin-1")

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


# %% [markdown]
# ### 3. Build a DataFrame from Directory
#
# Key terms: pd.DataFrame, list of dicts, row names, classification labels
#
# - Next, wrap the generator into a function that builds a Pandas DataFrame.
#
# Steps:
#
# - Initialize empty lists rows and row_names.
#
# - Loop over the generator, appending email data to lists.
#
# - Convert to a DataFrame with MESSAGE and CATEGORY columns.

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


# %% [markdown]
# Example Usage with Paths
#
# Key terms: relative paths, constants, spam vs ham classification
#
# - Define relative paths for all email folders.
#
# - Spam = 1, Non-spam = 0.

# %%
SPAM_1_PATH = "SpamData/01_Processing/spam_assassin_corpus/spam_1"
SPAM_2_PATH = "SpamData/01_Processing/spam_assassin_corpus/spam_2"
EASY_NONSPAM_1_PATH = "SpamData/01_Processing/spam_assassin_corpus/easy_ham_1"
EASY_NONSPAM_2_PATH = "SpamData/01_Processing/spam_assassin_corpus/easy_ham_2"

# Example: build DataFrame from SPAM_1 folder
spam_emails = df_from_directory(SPAM_1_PATH, 1)

# Check dimensions
print(spam_emails.shape)

# Check first 5 rows
spam_emails.head()

# %% [markdown]
# **Summary**
#
# - We introduced generator functions to process data one email at a time without overloading memory.
#
# - Practiced with generate_squares(N) to learn yield.
#
# - Built email_body_generator(path) to loop through files and extract (file_name, email_body).
#
# - Wrapped results into a Pandas DataFrame with df_from_directory().
#
# - Defined relative paths for spam and ham datasets, then built spam_emails DataFrame.
#
# This pipeline lets us efficiently load thousands of emails into a clean, labeled dataset.

# %% [markdown]
#
