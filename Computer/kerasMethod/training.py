import csv
import glob
import json
import numpy as np
import pandas as pd
import time

from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


print('Loading training data...')
time_load_start = time.time()         # Returns the number of ticks after a certain event.


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Load training data, unpacking what's in the saved .npz files.
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 4), 'float')

training_data = glob.glob('training_data/*.npz')
print(image_array.shape)
for single_npz in training_data:
    with np.load(single_npz) as data:
        print(data.files)
        train_temp = data['train']
        train_labels_temp = data['train_labels']
        train_temp, train_labels_temp = unison_shuffled_copies(train_temp, train_labels_temp)
        print(train_temp.shape)
        print(train_labels_temp.shape)

    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))


X = image_array[1:, :]
y = label_array[1:, :]
print('Shape of feature array: ', X.shape)
print('Shape of label array: ', y.shape)

# # Normalize with l2 (not gonna use this...)
# X = preprocessing.normalize(X, norm='l2')

# Normalize from 0 to 1
X = X / 255.

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20)
model = Sequential()

time_load_end = time.time()
time_load_total = time_load_end - time_load_start
print('Total time taken to load image data:', time_load_total, 'seconds')


# Get start time of Training
time_training_start = time.time()

print('Training...')

# Dense(n) is a fully-connected layer with n hidden units in the first layer.
# You must specify the expected input data shape (e.g. input_dim=20 for 20-dimensional input vector).
model.add(Dense(30, input_dim=38400, kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Activation('relu'))


model.add(Dense(4, kernel_initializer='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train,
    epochs=10,
    batch_size=1000,
    validation_data=(X_test, Y_test))

# Get end time of Training
time_training_end = time.time()
time_training_total = time_training_end - time_training_start
print('')
print('Total time taken to train model:', time_training_total, 'seconds')

# Evalute trained model on TEST set
print('')
print('Evaluation of model on test holdout set:')
score = model.evaluate(X_test, Y_test, batch_size=1000)
loss = score[0]
accuracy = score[1]
print('')
print('Loss score: ', loss)
print('Accuracy score: ', accuracy)

# Save model as h5
timestr = time.strftime('%Y%m%d_%H%M%S')
filename_timestr = 'nn_{}.h5'.format(timestr)
model.save('nn_h5/nn_{}.h5'.format(timestr))

# Save parameters to json file
json_string = model.to_json()
with open('./logs/nn_params_json/nn_{}.json'.format(timestr), 'w') as new_json:
    json.dump(json_string, new_json)

# Save training results to csv log
row_params = [str(training_data)[-33:-2], filename_timestr, loss, accuracy]
with open('./logs/log_nn_training.csv','a') as log:
    log_writer = csv.writer(log)
    log_writer.writerow(row_params)