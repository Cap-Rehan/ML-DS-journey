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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
from PIL import Image

from bs4 import BeautifulSoup

# %% [markdown]
# ## 1. Notebook Setup & Constants
# Key terms: Markup, Constants, Relative Path, Naming Convention

# %%
EXAMPLE_FILE = "SpamData/01_Processing/practice_email.txt"

# %% [markdown]
# ### 2. Opening & Reading Files
# Key terms: Stream/File Object, open(), read(), close()

# %%
with open(EXAMPLE_FILE, encoding="latin-1") as f:
    message = f.read()
print(message)

# %% [markdown]
# ### 3. Encodings & Email Structure
# Key terms: Encoding, ASCII, UTF-8, Latin-1, Email Header, Email Body

# %%
# checking the default encoding 
import sys
print(sys.getfilesystemencoding())

# %% [markdown]
# **Summary**
# Organized notebook with constants and standard open-read-close file handling for robust email processing.

# %% [markdown]
# ## 1. Goal: Extract only the email body
# Key terms: Email body, header, boolean flag, list of lines

# %% [markdown]
# ### 2. Logic: Detect start of the body
# Key terms: Blank line, \n, condition checks

# %% [markdown]
# ### 3. Output: Combine and print body
# Key terms: join(), readability, formatting

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
# Implemented logic to extract email bodies by detecting the header-separating blank line and joining subsequent lines.

# %% [markdown]
# ## 1. Generator Functions
# Key terms: generator function, yield, memory efficiency, looping

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
# Key terms: os.walk(), join(), nested loops, yield file_name, email_body

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
# Key terms: pd.DataFrame, list of dicts, row names, classification labels

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
# Key terms: relative paths, constants, spam vs ham classification

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
# Used generator functions (`yield`) to process emails one-by-one, avoiding memory overload, and built a labeled DataFrame.

# %% [markdown]
# ## 1. Loading all spam emails into one DataFrame
# Key terms: append(), constants, category labels

# %%
spam_emails = pd.concat([spam_emails, df_from_directory(SPAM_2_PATH, SPAM_CAT)])

print(spam_emails.shape)
spam_emails.tail()

# %% [markdown]
# ### 2. Creating a DataFrame for non-spam (ham) emails
# Key terms: ham emails, symmetry, reuse of logic

# %%
ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = pd.concat([ham_emails, df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT)])

print(ham_emails.shape)
ham_emails.tail()

# %% [markdown]
# ### 3. Combining spam and non-spam into one dataset
# Key terms: pd.concat(), full dataset, inspection

# %%
data = pd.concat([spam_emails, ham_emails])

print("Shape of entire dataframe is: ", data.shape)
data.tail()

# %% [markdown]
# **Summary**
# Loaded and merged all spam and ham emails into a single 5800-sample DataFrame with consistent labeling.

# %% [markdown]
# ## 1. Checking for missing (null) values
# Key terms: isnull(), None, missing values vs empty strings

# %%
print(data.MESSAGE.isnull().any())
print(data.MESSAGE.isnull().sum())

# %% [markdown]
# Important distinction:
# - None → missing value
# - "" → empty string (length = 0, but still a valid string)

# %% [markdown]
# ### 2. Detecting empty emails (strings of length zero)
# Key terms: empty strings, str.len(), Boolean masking

# %%
(data.MESSAGE.str.len() == 0).any()

# %%
print((data.MESSAGE.str.len() == 0).sum()) # counting
data[data.MESSAGE.str.len() == 0] # displaying

# %% [markdown]
# ### 3. Locating and identifying the problematic rows
# Key terms: Boolean indexing, system files, .index, get_loc()

# %%
print(data[data.MESSAGE.str.len() == 0].index)

# %% [markdown]
# Why this happened:
# - os.walk() blindly reads all files in a directory.
# - System files (cmds, .DS_Store) are not emails but got included anyway.

# %%
# locating the entry

data.index.get_loc('cmds') # not working for some reason...

# %% [markdown]
# **Summary**
# Audited data to remove nulls and empty strings caused by system files like `.DS_Store`.

# %% [markdown]
# ## 1. Removing system files from the DataFrame
# Key terms: drop(), index labels, inplace

# %%
data.drop('cmds', inplace=True)

# %%
data.shape

# %% [markdown]
# ### 2. Creating document IDs for emails
# Key terms: document ID, sequential indexing, range()

# %%
data['MAIL_ID'] = range(1, 5797)

# %%
data

# %% [markdown]
# ### 3. Replacing index & preserving filenames
# Key terms: set_index(), index reassignment, metadata preservation

# %%
data['FILE_NAME'] = data.index
data.head()

# %%
data.set_index('MAIL_ID', inplace=True)

# %%
data.tail()

# %% [markdown]
# **Summary**
# Removed non-email artifacts and established a sequential `MAIL_ID` index while preserving original filenames.

# %% [markdown]
# ## 1. Why save the cleaned dataset
# Key terms: persistence, preprocessing cost, reproducibility

# %% [markdown]
# ### 2. Saving the DataFrame using Pandas (to_json)
# Key terms: to_json(), relative path, file extension

# %%
DATA_JSON_FILE = 'SpamData/01_Processing/emails.json'

# %%
data.to_json(DATA_JSON_FILE)

# %% [markdown]
# ### 3. Understanding the JSON output
# Key terms: JSON structure, keys, document IDs, serialization

# %% [markdown]
# **Summary**
# Persisted the cleaned DataFrame to JSON to avoid re-processing in future sessions.

# %% [markdown]
# ## 1. Counting spam vs non-spam messages
# Key terms: value_counts(), categories, aggregation

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
# Key terms: plt.pie(), labels, %matplotlib inline

# %% [markdown]
# ### 3. Customizing pie charts (quality + aesthetics)
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
# Visualized class distribution (Spam vs. Legit) using styled pie charts with custom colors and percentage labels.

# %% [markdown]
# ## 1. Turning a pie chart into a donut chart
# Key terms: plt.Circle(), radius, add_artist, axes

# %% [markdown]
# ### 2. Improving donut chart readability
# Key terms: pctdistance, label positioning

# %%
plt.figure(figsize= (4,4), dpi= 150)

plt.pie(category_counts, labels= label_names, explode= [0.01, 0.01],
        autopct='%1.0f%%', pctdistance= 0.7)

circle= plt.Circle((0, 0), radius= 0.5, fc='#F0F0F0')
plt.gca().add_artist(circle)

plt.show()

# %% [markdown]
# ### 3. Donut charts with multiple categories
# Key terms: multiple classes, color palettes, explode offsets

# %%
plt.figure(figsize= (4,4), dpi= 150)

plt.pie([26, 60, 17, 55], labels= ['A', 'B', 'C', 'D'], explode= [0.01, 0.01, 0.01, 0.01],
        autopct='%1.0f%%', pctdistance= 0.8)

circle= plt.Circle((0, 0), radius= 0.6, fc='#F0F0F0')
plt.gca().add_artist(circle)

plt.show()

# %% [markdown]
# **Summary**
# Refined visualization into donut charts using a central circle artist for better aesthetics in multi-category data.

# %% [markdown]
# ## 1. What NLP is & why we need it
# Key terms: Natural Language Processing (NLP), text preprocessing, Naive Bayes
#
# ### 2. Text preprocessing pipeline (high-level)
# Key terms: normalization, tokenization, stop words, stemming, punctuation removal

# %% [markdown]
# ### 3. Lowercasing & introducing NLTK
# Key terms: string normalization, NLTK, Porter Stemmer

# %% [markdown]
# ### Imports:

# %%
msg = "All work and no play makes Jack a dull boy. # The Shining 1980 reference"
msg.lower()

# %% [markdown]
# **Summary**
# Defined the NLP preprocessing pipeline: lowercasing, tokenization, stop-word removal, stemming, and punctuation stripping.

# %% [markdown]
# ## 1. Tokenization with NLTK
# Key terms: tokenization, punkt, word_tokenize

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
# Key terms: stop words, bag of words, Naive Bayes

# %%
nltk.download('stopwords')

# %%
print(type(stopwords.words('english')))
stopwords.words('english') # a list 

# %% [markdown]
# ### 3. Using sets to remove stop words efficiently
# Key terms: set, membership testing, in, not in

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
# Implemented NLTK tokenization and efficient stop-word removal using Python sets to reduce noise.

# %% [markdown]
# ## 1. Word stems & stemming
# Key terms: stemming, word stem, Porter Stemmer

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
# Key terms: punctuation filtering, isalpha()

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
# Applied Porter Stemmer and `isalpha()` to normalize words to roots and strip punctuation.

# %% [markdown]
# ## 1. What HTML tags are & why we remove them
# Key terms: HTML, tags, rich text, plain text

# %% [markdown]
# ### 2. Seeing HTML inside real emails
# Key terms: HTML email body, raw email text, pandas .at

# %%
data.at[7, 'MESSAGE']

# %% [markdown]
# 3. Removing HTML tags with BeautifulSoup
# Key terms: BeautifulSoup, HTML parser, get_text()

# %%
soup = BeautifulSoup(data.at[7, 'MESSAGE'], 'html.parser')

# %%
print(soup.prettify())

# %%
clean_text= soup.get_text()
print(clean_text)


# %% [markdown]
# **Summary**
# Used BeautifulSoup to strip HTML tags, ensuring only raw text remains for classification.

# %% [markdown]
# ## 1. Designing the email cleaning function
# Key terms: preprocessing function, modularity, reusability
#
# ### 2. Cleaning text: lowercase → tokenize → filter → stem
# Key terms: normalization, tokenization, stop-word removal, stemming

# %%
def clean_msg(message, stemmer= PorterStemmer(),
                stop_words= set(stopwords.words('english'))):

    words= word_tokenize(message.lower())
    filtered_words= []

    for word in words:
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))

    return filtered_words


# %%
r1= clean_msg(data.at[3, 'MESSAGE'])


# %% [markdown]
# ### 3. Extending the function: removing HTML tags
# Key terms: BeautifulSoup, HTML stripping, pipeline completion

# %%
def clean_msg_no_html(message, stemmer= PorterStemmer(),
                    stop_words= set(stopwords.words('english'))):
    
    soup= BeautifulSoup(message, 'html.parser')
    clean_message = soup.get_text()

    words= word_tokenize(clean_message.lower())
    filtered_words= []

    for word in words:
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))

    return filtered_words


# %%
r2= clean_msg_no_html(data.at[3, 'MESSAGE'])

# %%
# exporting both to compare

# import json

# with open("r1data.json", "w") as f:
#     json.dump(r1, f)

# with open("r2data.json", "w") as f:
#     json.dump(r2, f)

# %% [markdown]
# **Summary**
# Encapsulated all preprocessing steps (HTML stripping, tokenization, stemming) into a single reusable function.

# %% [markdown]
# ## 1. Reviewing dataframe slicing & selection
# Key terms: at, iat, iloc, labels vs positions

# %%
data.at[50, 'MESSAGE']

# %%
data.iat[49, 0]

# %%
data.iloc[48:52]

# %% [markdown]
# ### 2. Applying a function to multiple emails
# Key terms: apply, vectorization, series transformation

# %%
# Select a subset of messages:
first_emails = data.MESSAGE.iloc[0:5]
print(type(first_emails))
first_emails

# %%
# Apply cleaning function to each email:
first_emails.apply(clean_msg_no_html)

# %% [markdown]
# ### 3. Flattening nested lists & scaling to all messages
# Key terms: nested lists, flattening, list comprehension, benchmarking

# %%
# %%time

nested_list= data.MESSAGE.apply(clean_msg_no_html)

# %%
print(type(nested_list))
nested_list

# %%
# Flatten using loops (classic approach):

flat_list= []

# for sublist in nested_list:
#     for item in sublist:
#         flat_list.append(item)

# Flatten using list comprehension (preferred):

flat_list= [item for sublist in nested_list for item in sublist]

len(flat_list)

# %%
flat_list

# %% [markdown]
# **Summary**
# Applied the cleaning function to the entire dataset using `apply()`, flattened the results, and benchmarked performance.

# %% [markdown]
# ## 1. Using logic to slice dataframes
# Key terms: boolean indexing, conditional slicing, dataframe subsets

# %%
print(type(data[data.CATEGORY == 0]))
data[data.CATEGORY == 0]

# %% [markdown]
# ### 2. Extracting indices for spam & ham emails
# Key terms: index extraction, label tracking

# %%
ids_spam= data[data.CATEGORY == 1].index
ids_ham= data[data.CATEGORY == 0].index

type(ids_spam) # not lists or series or DataFrame

# %%
ids_ham

# %% [markdown]
# ### 3. Subsetting a series using an index
# Key terms: loc, index-based selection, aligned slicing

# %%
nested_list_spam= nested_list[ids_spam]
nested_list_ham= nested_list[ids_ham]

# %%
print(nested_list_spam.shape)
nested_list_spam

# %%
list_spam= [item for sublist in nested_list_spam for item in sublist]
len(list_spam)

# %%
list_ham= [item for sublist in nested_list_ham for item in sublist]
len(list_ham)

# %%
# storing count values in a dictionary (side quest)

words_count= {}

for the_word in list_spam:
    if (the_word not in words_count):
        words_count[the_word] = 1
    else: words_count[the_word] += 1

words_count

# %% [markdown]
# ### 4. Counting total & unique words
# Key terms: pd.Series, value_counts, vocabulary size

# %%
spammy_words= pd.Series(list_spam)
spammy_words.shape[0] # number of all words

# %%
spammy_words.value_counts().shape # number of unique words

# %%
spammy_words.value_counts()[:10] # frequency

# %%
normal_words= pd.Series(list_ham)
print(normal_words.value_counts().shape)
normal_words.value_counts()[:10]

# %% [markdown]
# **Summary**
# Sliced the dataset by category to compute vocabulary size and identify the most frequent words for spam vs. ham.

# %% [markdown]
# ## A little note:
# Word clouds are fast, visual tools for highlighting frequent words in large text datasets.

# %% [markdown]
# ## 1. Creating a Basic Word Cloud
# Key terms: WordCloud, generate, imshow, interpolation

# %%
word_cloud= WordCloud().generate(email_body)

plt.figure(figsize=(8,8), dpi=227)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show() # we'll learn customization later

# %% [markdown]
# ### 2. Using NLTK Corpora (Entire Books)
# Key terms: nltk.corpus, gutenberg, words()

# %%
nltk.download('brown')
nltk.download('shakespeare')
nltk.download('gutenberg')

# %%
corpus1= nltk.corpus.gutenberg.words('whitman-leaves.txt')

# %%
type(corpus1)

# %% [markdown]
# ### 3. Preparing Corpus for WordCloud
# Key terms: tokens, join, string conversion

# %%
word_list= [word for word in corpus1]
corpus1_string= ' '.join(word_list)

word_cloud = WordCloud().generate(corpus1_string)

plt.figure(figsize=(6,6), dpi=150, facecolor='black')
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
print(corpus1_string)

# %% [markdown]
# **Summary**
# Generated basic word clouds from email text and standard NLTK literary corpora.

# %% [markdown]
# ## 1. Mask Preparation (Image → Array)
# Key terms: PIL (Pillow), RGB, NumPy array, mask

# %%
PNG_PATH= 'SpamData/01_Processing/wordcloud_resources/skull-icon.png'

icon = Image.open(PNG_PATH)

image_mask= Image.new(mode= "RGB", size= icon.size, color=(255,255,255))
image_mask.paste(icon, box= icon)

rgb_array= np.array(image_mask)

# %% [markdown]
# ### 2. WordCloud with Mask & Styling
# Key terms: WordCloud, mask, colormap, max_words

# %%
word_cloud= WordCloud(mask= rgb_array, background_color= 'white', max_words= 500) # colormap?
word_cloud.generate(corpus1_string)

plt.figure(figsize=(6,6), dpi=150)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ### 3. Understanding How the Mask Works
# Key terms: RGB values, pixel-level control

# %%
print(rgb_array.shape)

# side quest started
pixels = rgb_array.reshape(-1, 3)
print(pixels.shape)

# %%
values, counts = np.unique(pixels, axis=0, return_counts=True)

# %%
for v, c in zip(values, counts): # side quest ends
    print(v, '=', c)

# %% [markdown]
# **Summary**
# Created masked word clouds using NumPy arrays derived from images to control word placement.

# %% [markdown]
# ## 1. Ham Word Cloud (Thumbs-Up Mask)
# Key terms: mask, join, colormap, WordCloud

# %%
THUMBS_UP = 'SpamData/01_Processing/wordcloud_resources/thumbs-up.png'
THUMBS_DOWN = 'SpamData/01_Processing/wordcloud_resources/thumbs-down.png'

FONT_PATH = 'SpamData/01_Processing/wordcloud_resources/OpenSansCondensed-Bold.ttf'

# %%
# prepare string
ham_string= ' '.join(list_ham)

# prepare mask
icon= Image.open(THUMBS_UP)
new_mask= Image.new(mode= "RGB", size= icon.size, color= (255, 255, 255))
new_mask.paste(icon, box= icon)

# creating the array
array_up= np.array(new_mask)

# %%
# generating wordcloud
ham_cloud= WordCloud(mask= array_up, font_path= FONT_PATH, max_words= 500,
                    min_font_size= 6, colormap= 'autumn') # try relative scaling
ham_cloud.generate(ham_string.upper())

# plotting
plt.figure(figsize= (12, 12), dpi= 227, facecolor= 'black')
plt.imshow(ham_cloud, interpolation= 'bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ### 2. Spam Word Cloud (Thumbs-Down Mask + Custom Font)
# Key terms: font_path, uppercase text, max_font_size

# %%
# prepare string
spam_string= ' '.join(list_spam)

# prepare mask
icon= Image.open(THUMBS_DOWN)
new_mask= Image.new(mode= "RGB", size= icon.size, color= (255, 255, 255))
new_mask.paste(icon, box= icon)

# creating the array
array_down= np.array(new_mask)

# %%
# generating wordcloud
spam_cloud= WordCloud(mask= array_down, font_path= FONT_PATH, max_words= 2000,
                    min_font_size= 6, colormap= 'cool') # try relative scaling
spam_cloud.generate(spam_string.upper())

# plotting
plt.figure(figsize= (12, 12), dpi= 227, facecolor= 'black')
plt.imshow(spam_cloud, interpolation= 'bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ### 3. Stemming vs Visualization
# Key terms: stemming trade-off, model vs visualization

# %% [markdown]
# **Summary**
# Generated distinct 'Thumbs Up' (Ham) and 'Thumbs Down' (Spam) word clouds using custom masks and fonts.
