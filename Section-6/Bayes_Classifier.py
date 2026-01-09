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
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

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
SPAM_CAT = 1
HAM_CAT = 0

# Example: build DataFrame from SPAM_1 folder
# classification = 1 for spam and 0 for nonspam/ham
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
# ## 1. Loading all spam emails into one DataFrame
#
# Key terms: append(), constants, category labels
# - Previously, we loaded emails only from spam_1.
# - Now we extend this by appending emails from spam_2.
# - To improve readability, we replace raw numbers (1, 0) with semantic constants.
#
# Convention:
# - Spam = 1
# - Ham (non-spam) = 0

# %%
spam_emails = pd.concat([spam_emails, df_from_directory(SPAM_2_PATH, SPAM_CAT)])

print(spam_emails.shape)
spam_emails.tail()

# %% [markdown]
# ### 2. Creating a DataFrame for non-spam (ham) emails
#
# Key terms: ham emails, symmetry, reuse of logic
# - Same logic as spam → reuse the same function.
# - Load both easy_ham_1 and easy_ham_2.
# - Category label = HAM_CAT (0).

# %%
ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = pd.concat([ham_emails, df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT)])

print(ham_emails.shape)
ham_emails.tail()

# %% [markdown]
# ### 3. Combining spam and non-spam into one dataset
#
# Key terms: pd.concat(), full dataset, inspection
# - Final step: merge spam + ham into one DataFrame.
# - This is the dataset we’ll use for training the Naive Bayes model.

# %%
data = pd.concat([spam_emails, ham_emails])

print("Shape of entire dataframe is: ", data.shape)
data.tail()

# %% [markdown]
# **Summary**
#
# We loaded all email files from disk and converted them into a single, clean Pandas DataFrame.
# Spam and non-spam emails were labeled consistently using constants and merged into one dataset with 5800 samples.
# This marks the transition from raw files → structured data, which is exactly what we need before feature extraction and model training.

# %% [markdown]
# ## 1. Checking for missing (null) values
#
# Key terms: isnull(), None, missing values vs empty strings
# - We first check whether any email bodies are missing (i.e., None / null).
#
# Accessing a column can be done in two equivalent ways:
# - data.MESSAGE
# - data['MESSAGE']
#
# isnull() returns a Boolean series (True/False for each row).
# - To check if any missing values exist, chain .values.any().
# - In Python, missing values are represented by None, not null.

# %%
print(data.MESSAGE.isnull().any())
print(data.MESSAGE.isnull().sum())

# %% [markdown]
# Important distinction:
# - None → missing value
# - "" → empty string (length = 0, but still a valid string)

# %% [markdown]
# ### 2. Detecting empty emails (strings of length zero)
#
# Key terms: empty strings, str.len(), Boolean masking
# - Even if there are no null values, emails may still be empty strings.
#
# Strategy:
# 1. Convert column to string
# 2. Measure string length
# 3. Check if length equals zero

# %%
(data.MESSAGE.str.len() == 0).any()

# %%
print((data.MESSAGE.str.len() == 0).sum()) # counting
data[data.MESSAGE.str.len() == 0] # displaying

# %% [markdown]
# ### 3. Locating and identifying the problematic rows
#
# Key terms: Boolean indexing, system files, .index, get_loc()
#
# - To find which rows are empty, we filter the dataframe using the Boolean condition.

# %%
print(data[data.MESSAGE.str.len() == 0].index)

# %% [markdown]
# Why this happened:
# - os.walk() blindly reads all files in a directory.
# - System files (cmds, .DS_Store) are not emails but got included anyway.
# - .DS_Store is a macOS hidden file.
# - cmds files are system artifacts from extracted archives.
#

# %%
# locating the entry

data.index.get_loc('cmds') # not working for some reason...

# %% [markdown]
# **Summary**
#
# We performed the first serious data-cleaning audit on our dataset.
# - Verified there are no null (None) values.
# - Detected 4 empty emails caused by system files, not real data.
# - Traced the issue back to os.walk() indiscriminately reading all files.
#

# %% [markdown]
# ## 1. Removing system files from the DataFrame
#
# Key terms: drop(), index labels, inplace
#
# - Some rows in the DataFrame are not real emails (cmds, .DS_Store).
#
# - These came from os.walk() picking up system files.
#
# - We remove them using DataFrame.drop() by index name.
#
# - Setting inplace=True modifies the DataFrame directly (no reassignment needed).

# %%
data.drop('cmds', inplace=True)

# %%
data.shape

# %% [markdown]
# ### 2. Creating document IDs for emails
#
# Key terms: document ID, sequential indexing, range()
#
# - File names as index are hard to track and meaningless for analysis.
#
# - We create a sequential document ID for every email.
#
# - IDs run from 0 to len(data) - 1.

# %%
data['MAIL_ID'] = range(1, 5797)

# %%
data

# %% [markdown]
# ### 3. Replacing index & preserving filenames
#
# Key terms: set_index(), index reassignment, metadata preservation
#
# - We don’t want to lose the original filenames → store them in a column.
#
# - Then replace the DataFrame index with DOC_ID.

# %%
data['FILE_NAME'] = data.index
data.head()

# %%
data.set_index('MAIL_ID', inplace=True)

# %%
data.tail()

# %% [markdown]
# **Summary**
#
# - We finalized a critical data-cleaning phase by:
#
# - Removing non-email system files from the dataset.
#
# - Introducing sequential document IDs for reliable tracking.
#
# - Replacing messy filename-based indexing with clean numeric IDs while preserving filenames as metadata.

# %% [markdown]
# ## 1. Why save the cleaned dataset
#
# Key terms: persistence, preprocessing cost, reproducibility
#
# We’ve already done expensive work:
#
# - Read ~5800 files from disk
#
# - Extracted email bodies
#
# - Cleaned system files
#
# - Built a structured DataFrame
#
# Re-running all of this every time is wasteful and slow.
#
# Saving the DataFrame lets us reload instantly and continue from a clean checkpoint.

# %% [markdown]
# ### 2. Saving the DataFrame using Pandas (to_json)
#
# Key terms: to_json(), relative path, file extension
#
# - Pandas supports multiple output formats (CSV, TXT, JSON, etc.).
#
# - We use JSON because it preserves structure and is widely used in ML & web systems.
#
# - First, define a constant for the output file path.

# %%
DATA_JSON_FILE = 'SpamData/01_Processing/emails.json'

# %%
data.to_json(DATA_JSON_FILE)

# %% [markdown]
# ### 3. Understanding the JSON output
#
# Key terms: JSON structure, keys, document IDs, serialization
#
# Raw JSON looks ugly and dense when opened directly.
#
# When beautified, the structure becomes clear:
#
# - Top-level keys = DataFrame columns
# - - CATEGORY
# - - MESSAGE
# - - FILE_NAME
# - Nested keys = document IDs (0 → 5795)
#
# Example interpretation:
#
# - CATEGORY["60"] = 1 → document 60 is spam
# - MESSAGE["60"] = "email body text..."
#
# JSON preserves:
#
# - Column names
# - Document IDs
# - Full message content
#
# This makes it ideal for:
#
# - Reloading into Python
# - Sending over APIs
# - Long-term storage

# %% [markdown]
# **Summary**
#
# We finalized data cleaning and persisted our work by saving the dataset to a JSON file.
# This allows us to skip file parsing and cleaning in future sessions while preserving the full DataFrame structure.
# With clean, saved data in place, the next step is data exploration and visualization, where we start learning from the dataset instead of just preparing it.

# %% [markdown]
# ## 1. Counting spam vs non-spam messages
#
# Key terms: value_counts(), categories, aggregation
#
# - Before visualizing anything, we need the actual counts.
# - Use value_counts() on the CATEGORY column.
#
# Category meaning:
#
# - 1 → spam
# - 0 → ham (legitimate mail)

# %%
data.CATEGORY.value_counts()

# %%
SPAM_COUNT = data.CATEGORY.value_counts()[1]
HAM_COUNT = data.CATEGORY.value_counts()[0]

# %%
category_counts = [SPAM_COUNT, HAM_COUNT]
label_names = ['Spam mails', 'Legit mails']

# %% [markdown]
# ### 2. Creating a basic pie chart with Matplotlib
#
# Key terms: plt.pie(), labels, %matplotlib inline
#
# - We use matplotlib for visualization.
#
# - %matplotlib inline ensures plots render correctly in Jupyter and export with the notebook.

# %% [markdown]
# ### 3. Customizing pie charts (quality + aesthetics)
#
# Key terms: figsize, dpi, startangle, autopct, colors, explode

# %%
plt.style.use('fivethirtyeight')
plt.figure(figsize= (4,4), dpi= 150)
plt.pie(category_counts, labels= label_names, explode= [0, 0.05], autopct='%1.0f%%')
plt.show()

# %%
print(plt.style.available) # all themes available

# %% [markdown]
# **Summary**
#
# We visualized the spam vs ham distribution using pie charts, starting from raw counts and progressively improving the design.
#
# Key takeaways:
# - Always compute counts first (value_counts).
# - Default plots are rarely presentation-ready.
# - DPI, font size, colors, percentages, and explode effects dramatically improve clarity.

# %% [markdown]
# ## 1. Turning a pie chart into a donut chart
#
# Key terms: plt.Circle(), radius, add_artist, axes
# - A donut chart is just a pie chart with a circle drawn on top.
# - We draw a circle at the center (0, 0) and place it over the pie.
# - The circle must be added to the current axes using plt.gca().add_artist().

# %% [markdown]
# ### 2. Improving donut chart readability
#
# Key terms: pctdistance, label positioning
# - Percent labels default to the center of the wedge (bad for donuts).
# - Use pctdistance to push labels outward into the donut ring.

# %%
plt.figure(figsize= (4,4), dpi= 150)

plt.pie(category_counts, labels= label_names, explode= [0.01, 0.01],
        autopct='%1.0f%%', pctdistance= 0.7)

circle= plt.Circle((0, 0), radius= 0.5, fc='#F0F0F0')
plt.gca().add_artist(circle)

plt.show()

# %% [markdown]
# ### 3. Donut charts with multiple categories
#
# Key terms: multiple classes, color palettes, explode offsets
# - Donut charts shine with 3–5 categories.
# - Add categories, sizes, and matching colors.
# - Use small, equal offsets to create subtle spacing between segments.

# %%
plt.figure(figsize= (4,4), dpi= 150)

plt.pie([26, 60, 17, 55], labels= ['A', 'B', 'C', 'D'], explode= [0.01, 0.01, 0.01, 0.01],
        autopct='%1.0f%%', pctdistance= 0.8)

circle= plt.Circle((0, 0), radius= 0.6, fc='#F0F0F0')
plt.gca().add_artist(circle)

plt.show()

# %% [markdown]
# **Summary**
#
# A donut chart is a pie chart with a centered circle drawn on top.
# Key techniques include controlling radius, label distance, DPI, and color palettes.
# Donut charts are especially effective for multi-category comparisons in reports and dashboards.
#

# %% [markdown]
# ## 1. What NLP is & why we need it
#
# Key terms: Natural Language Processing (NLP), text preprocessing, Naive Bayes
# - NLP is the field that deals with understanding and processing human language.
# - It used to be part of classical AI but now sits firmly in machine learning.
# - Real-world NLP applications include:
#     - Search engines
#     - Sentiment analysis
#     - Ads (Google AdWords)
#     - Translation
#     - Spellcheck & autocorrect
#     - Voice assistants (Siri, Alexa)
# - For our Naive Bayes spam classifier, NLP is required to convert raw email text into a machine-readable form.
# - Raw text cannot be fed directly into ML algorithms → preprocessing is mandatory.
#
#
# ### 2. Text preprocessing pipeline (high-level)
#
# Key terms: normalization, tokenization, stop words, stemming, punctuation removal
# - Planned preprocessing steps:
#
#     1.	Lowercasing – ignore capitalization differences
#     2.	Tokenization – split sentences into individual words
#     3.	Stop-word removal – remove common grammar words like the, is, and
#     4.	HTML tag removal – strip formatting noise
#     5.	Stemming – reduce words to their root form
#         - go, goes, going → go
#     6.	Punctuation removal – grammar doesn’t matter for Naive Bayes
#
# This pipeline reduces noise and helps the model focus on meaningful word frequencies.

# %% [markdown]
# ### 3. Lowercasing & introducing NLTK
#
# Key terms: string normalization, NLTK, Porter Stemmer
# - Letter casing usually carries no semantic meaning for classification.
# - Search engines already ignore casing → we should too.
#

# %% [markdown]
# From here on, we’ll rely on NLTK (Natural Language Toolkit):
# - Industry-standard Python library for NLP
# - Used by researchers and professionals
# - Provides tools for tokenization, stop words, stemming, etc.

# %%
msg = "All work and no play makes Jack a dull boy. # The Shining 1980 reference"
msg.lower()

# %% [markdown]
# ### Imports:
#
# import nltk
#
# from nltk.stem import PorterStemmer
#
# from nltk.corpus import stopwords
#
# from nltk.tokenize import word_tokenize

# %% [markdown]
# **Summary**
#
# We’re entering the NLP phase of the project, where raw email text is transformed into structured data suitable for machine learning.
# We defined the full preprocessing pipeline and started with lowercasing, the simplest but essential normalization step.

# %% [markdown]
# ## 1. Tokenization with NLTK
#
# Key terms: tokenization, punkt, word_tokenize
# - Tokenization = splitting a sentence into individual words (tokens).
# - NLTK handles this for us, but it requires a tokenizer model called punkt.
# - The tokenizer is downloaded once and stored locally under nltk_data/tokenizers/.
#

# %%
# download tokenizer...
nltk.download('punkt')

# %%
nltk.download('punkt_tab') # LookUpError solution

# %%
tokens = word_tokenize(msg.lower())
print(tokens)

# %% [markdown]
# ### 2. Stop words & why we remove them
#
# Key terms: stop words, bag of words, Naive Bayes
# - Stop words are extremely common words that carry little meaning: the, a, and, to, of, is, on…
# - Since Naive Bayes treats words independently, stop words add noise but no signal.
#
# Example:
# - “Flights to London” → “to” is useless for classification
# - Modern search engines keep stop words; we intentionally remove them.

# %%
nltk.download('stopwords')

# %%
print(type(stopwords.words('english')))
stopwords.words('english') # a list 

# %% [markdown]
# ### 3. Using sets to remove stop words efficiently
#
# Key terms: set, membership testing, in, not in
# - A Python set is ideal for fast membership checks.
# - Sets are unordered and contain unique values only.
# - Much faster than checking against a list for large datasets.
#

# %%
stop_words = set(stopwords.words('english'))
type(stop_words)

# %%
if 'that' in stop_words:
    print("a stopword...")
else: print("not a stopword...")

# %%
words = set(word_tokenize(msg.lower()))

filtered= []
junk= []
for word in words:
    if word not in stop_words:
        filtered.append(word)
    else: junk.append(word)

print(filtered, "\n\n")
print(junk)

# %% [markdown]
# **Summary**
#
# We introduced tokenization using NLTK, downloaded the required resources, and learned how to remove stop words efficiently using sets.
# At this point, our text is lowercase, tokenized, and partially cleaned—but it still contains punctuation and unstemmed words.

# %% [markdown]
# ## 1. Word stems & stemming
#
# Key terms: stemming, word stem, Porter Stemmer
# - Stemming reduces words to their base/root form so related words are treated the same.
# - Example:
# 	- fishing, fished, fisher, fishlike → fish
#
# - Stemming does not guarantee real words:
# 	- argue, argued, argues, arguing → argu
# 	- This is intentional: the goal is grouping variants, not linguistic correctness.
# - The standard English stemmer is the Porter Stemmer, created by Martin Porter (1980s).
# - NLTK also provides the SnowballStemmer, useful for non-English languages.

# %%
lyrics= """This love is ablaze, I saw flames from the side of the stage
And the fire brigade comes in a couple of days
Until then, we've got nothin' to say and nothin' to know
But somethin' to drink and maybe somethin' to smoke
Let it go until our roads are changed
Singin' "We Found Love" in a local rave
No, I don't really know what I'm supposed to say
But I can just figure it out and then just hope and pray
I told her my name and said, "It's nice to meet ya"
Then she handed me a bottle of water filled with tequila
I already know she's a keeper just from this one small act of kindness
I'm in deep shit if anybody finds out
I meant to drive home but I've drunk all of it now
Not soberin' up, we just sit on the couch
One thing led to another, now, she's kissin' my mouth"""

words = word_tokenize(lyrics.lower())
stemmer = PorterStemmer()

filtered= []
junk= []

for word in words:
    stemmed_word= stemmer.stem(word)

    if word not in stop_words:
        filtered.append(stemmed_word)
    else:
        junk.append(stemmed_word)
print(words, "\n\n")
print(filtered, "\n\n")
print(junk)

# %% [markdown]
# ### 2. Removing punctuation
#
# Key terms: punctuation filtering, isalpha()
# - Our output still contains punctuation (., !, ?).
# - Naive Bayes ignores grammar → punctuation adds noise.
# - Python strings provide a simple solution: isalpha().

# %%
words = word_tokenize(msg.lower())

filtered= []
junk= []

for word in words:
    stemmed_word= stemmer.stem(word)

    if word not in stop_words and word.isalpha():
        filtered.append(stemmed_word)
    else:
        junk.append(stemmed_word)

print(filtered, "\n\n")
print(junk)

# %% [markdown]
# **Summary**
#
# We added the final two preprocessing steps: stemming and punctuation removal.
# Stemming collapses word variants into a single root, and isalpha() cleanly filters punctuation.
# At this point, our text is lowercased, tokenized, stop-word filtered, stemmed, and punctuation-free.

# %% [markdown]
# ## 1. What HTML tags are & why we remove them
#
# Key terms: HTML, tags, rich text, plain text
# - HTML adds structure and formatting to emails (bold text, paragraphs, images, links).
# - Email clients render HTML nicely, but underneath it’s just raw text with tags.
# - HTML uses opening and closing tags (\<b>...\</b>, \<p>...\</p>, \<h1>...\</h1>).
#
# For Naive Bayes + Bag of Words, formatting is irrelevant → HTML tags are noise.
# We only care about actual words, not presentation.

# %% [markdown]
# ### 2. Seeing HTML inside real emails
#
# Key terms: HTML email body, raw email text, pandas .at
# - Some emails in the dataset contain large blocks of HTML.
# - We can inspect a specific email efficiently using .at[index, column].

# %%
data.at[7, 'MESSAGE']

# %% [markdown]
# 3. Removing HTML tags with BeautifulSoup
#
# Key terms: BeautifulSoup, HTML parser, get_text()
# - Python provides an excellent library for HTML parsing: BeautifulSoup.
# - It can parse, prettify, and strip HTML with minimal code.
# - Python ships with a built-in HTML parser → no extra setup needed.

# %%
soup = BeautifulSoup(data.at[7, 'MESSAGE'], 'html.parser')

# %%
print(soup.prettify())

# %%
clean_text= soup.get_text()
print(clean_text)

# %% [markdown]
# **Summary**
#
# HTML tags exist to make emails look good, not to add meaning.
# For our spam classifier, HTML is pure noise and must be stripped.
# Using BeautifulSoup, we cleanly remove all tags and extract only the text.
#
# Next step: combine all preprocessing steps into reusable Python functions and scale this to every email in the dataset.

# %% [markdown]
#
