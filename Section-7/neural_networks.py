# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ml-ds (3.11.15)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import os
import numpy as np
import tensorflow as tf
import keras

from keras.preprocessing.image import img_to_array, array_to_img, load_img
from IPython.display import display

import matplotlib.pyplot as plt

from keras.layers import Dense, Input, Activation
from keras.models import Sequential
from keras.callbacks import TensorBoard

from time import strftime

# %%
# Constants

CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

TOTAL_PIXELS = 32*32
CHANNELS = 3
TOTAL_INPUTS = TOTAL_PIXELS * CHANNELS

VALIDATION_SIZE = 10000
SMALL_DATASET_SIZE = 1000

LOG_DIR = 'tensorboard_cifar_logs/'

# %% [markdown]
# # Importing Data

# %%
from keras.datasets import cifar10

# %%
(X_train_all, y_train_all), (X_test, y_test) = cifar10.load_data()

# %%
print(X_test.shape, y_train_all.shape)

# %% [markdown]
# # Explore the Data

# %%
display(array_to_img(X_train_all[0]))

# %%
plt.imshow(X_train_all[0])
plt.xlabel(CIFAR_LABELS[y_train_all[0][0]], fontsize= 18)
plt.show()

# %%
plt.figure(figsize=(20, 10), dpi= 227)
plt.tight_layout()

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train_all[i])
    plt.xlabel(CIFAR_LABELS[y_train_all[i][0]], fontsize= 16)
    plt.xticks([])
    plt.yticks([])

# %%
nr_images, x, y, c = X_train_all.shape
print(f'Number of images: {nr_images} ,\tWidth: {x} ,\tHeight: {y} ,\tChannels: {c}')

# %% [markdown]
# # Preprocessing

# %%
X_train_all = X_train_all / 255.0
X_train_all = X_train_all.reshape(nr_images, x*y*c)

# %%
X_test = X_test / 255.0
X_test = X_test.reshape(X_test.shape[0], x*y*c)

# %% [markdown]
# ## Creating Validation DataSet

# %%
x_val, y_val = X_train_all[:VALIDATION_SIZE], y_train_all[:VALIDATION_SIZE]
X_train, y_train = X_train_all[VALIDATION_SIZE:], y_train_all[VALIDATION_SIZE:]

# %% [markdown]
# ## Small Dataset to work initially

# %%
X_train_small = X_train[:SMALL_DATASET_SIZE]
y_train_small = y_train[:SMALL_DATASET_SIZE]

# %% [markdown]
# # Define the Neural Network using Keras

# %%
model_1 = Sequential([
    Input((TOTAL_INPUTS,)),
    Dense(units= 128, activation= 'relu'),
    Dense(units= 64, activation= 'relu'),
    Dense(32, activation= 'relu'),
    Dense(16, activation= 'relu'),
    Dense(10, activation= 'softmax')
])

model_1.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy',
                metrics= ['accuracy'])

# %%
type(model_1)

# %%
model_1.summary()


# %% [markdown]
# # Tensorboard (visualizing learning)

# %%
def get_tensorboard(model_name):
    folder = f'{model_name} at {strftime("%d %b, %H:%M")}'
    dir_path = os.path.join(LOG_DIR, folder)
    
    try:
        os.makedirs(dir_path)
    except OSError as err:
        print(err.strerror)
    else:
        print("directory created successfully")

    return TensorBoard(log_dir= dir_path)


# %% [markdown]
# # Fit the model

# %%
print(np.unique(y_train_small))

# %%
BATCH_SIZE = 1000
NR_EPOCHS = 50

# %%
import time
start = time.time()

model_1.fit(x= X_train_small, y= y_train_small, batch_size= BATCH_SIZE,
            epochs= NR_EPOCHS, callbacks= [get_tensorboard('Model 1')],
            verbose= 0, validation_data= (x_val, y_val))

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# %%
