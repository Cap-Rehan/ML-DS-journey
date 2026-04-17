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

from keras.preprocessing.image import array_to_img
from IPython.display import display

import matplotlib.pyplot as plt

from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix

from time import strftime
import time

import itertools

# %%
# Constants

CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

TOTAL_PIXELS = 32*32
CHANNELS = 3
TOTAL_INPUTS = TOTAL_PIXELS * CHANNELS

VALIDATION_SIZE = 10000
SMALL_DATASET_SIZE = 5000

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
model_2 = Sequential()
model_2.add(Input((TOTAL_INPUTS,)))
model_2.add(Dropout(rate= 0.2))
model_2.add(Dense(units= 128, activation= 'relu'))
model_2.add(Dense(units= 64, activation= 'relu'))
model_2.add(Dense(32, activation= 'relu'))
model_2.add(Dense(16, activation= 'relu'))
model_2.add(Dense(16, 'softmax'))

model_2.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy',
                metrics= ['accuracy'])

# %%
type(model_1)

# %%
model_2.summary()

# %%
model_3 = Sequential()
model_3.add(Input((TOTAL_INPUTS,)))
model_3.add(Dropout(rate= 0.2))
model_3.add(Dense(units= 128, activation= 'relu'))
model_3.add(Dropout(rate= 0.2))
model_3.add(Dense(units= 64, activation= 'relu'))
model_3.add(Dense(32, activation= 'relu'))
model_3.add(Dense(16, activation= 'relu'))
model_3.add(Dense(16, 'softmax'))

model_3.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy',
                metrics= ['accuracy'])

# %% [markdown]
# # Fit the model

# %%
BATCH_SIZE = 1000
NR_EPOCHS = 50

# %%
start = time.time()

model_1.fit(x= X_train, y= y_train, batch_size= BATCH_SIZE,
            epochs= NR_EPOCHS, callbacks= [get_tensorboard('Model 1 L')],
            verbose= 0, validation_data= (x_val, y_val))

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# %%
start = time.time()

model_2.fit(x= X_train, y= y_train, batch_size= BATCH_SIZE,
            epochs= NR_EPOCHS, callbacks= [get_tensorboard('Model 2 L')],
            verbose= 0, validation_data= (x_val, y_val))

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# %%
start = time.time()

model_3.fit(x= X_train, y= y_train, batch_size= BATCH_SIZE,
            epochs= NR_EPOCHS, callbacks= [get_tensorboard('Model 3 L')],
            verbose= 0, validation_data= (x_val, y_val))

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# %%
import this

# %% [markdown]
# # Predictions on individual images

# %%
x_val[0].shape

# %%
expanded_img = np.expand_dims(x_val[0], axis = 0)
expanded_img.shape

# %%
np.set_printoptions(precision=2)

# %%
model_1.predict(expanded_img)[0]

# %%
prediction = model_1.predict(expanded_img)
print(np.argmax(prediction, axis=1)[0])

# %%
prediction2 = model_2.predict(expanded_img)
print(np.argmax(prediction2, axis=1)[0])

# %%
print(y_val[0][0])

# %%
for i in range(10):
    pred = model_3.predict(np.expand_dims(x_val[i], axis= 0))
    print('True') if (y_val[i][0] == np.argmax(pred)) else print('False')
    # print(f'Actual: {y_val[i][0]}, Predicted: {np.argmax(pred)}')

# %% [markdown]
# # Evaluation

# %%
model_2.metrics_names

# %%
test_loss, test_accuracy = model_2.evaluate(X_test, y_test)
print(f'Model 2:\nLoss: {test_loss:.3}\tAccuracy: {test_accuracy:.2%}')

# %% [markdown]
# # Confusion Matrix

# %%
pred = model_2.predict(X_test)
conf_matrix = confusion_matrix(y_true= y_test, y_pred= np.argmax(pred, axis= 1))

# %%
conf_matrix.shape

# %%
print("maximum:", conf_matrix.max())
print("minimum:", conf_matrix.min())

# %%
conf_matrix[0]

# %%
plt.figure(figsize= (12, 12), dpi= 120)

plt.imshow(conf_matrix, cmap= 'Purples')
plt.title("Confusion Matrix", fontsize= 20)
plt.ylabel("Actual Labels", fontsize= 14)
plt.xlabel("Predicted Labels", fontsize= 14)

plt.yticks(np.arange(10), CIFAR_LABELS)
plt.xticks(np.arange(10), CIFAR_LABELS)

for i, j in itertools.product(range(10), range(10)):
    plt.text(j, i, conf_matrix[i, j], horizontalalignment= 'center',
             color= 'white' if conf_matrix[j, i] > conf_matrix.max()/1.4 else 'black')

plt.colorbar()
plt.show()

# %%
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
recall

# %%
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
precision

# %%
avg_recall, avg_precision = np.mean(recall), np.mean(precision)
print(f'Recall Score: {avg_recall:.2%}\nPrecision Score: {avg_precision:.2%}')

# %%
f_score = avg_precision*avg_recall*2 / (avg_recall + avg_precision)
print(f'F Score = {f_score:.2%}')
