import cv2
import numpy as np
import glob
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB

from sklearn.externals import joblib

print('Loading training data...')
e0 = cv2.getTickCount()
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
# load training data
image_array = np.zeros((1, 2400))
label_array = np.zeros((1, 4), 'float')
training_data = glob.glob('training_data_updated/*.npz')
print(image_array.shape)
for single_npz in training_data:
    with np.load(single_npz) as data:
        print(data.files)
        train_temp = data['train']
        train_labels_temp = data['train_labels']
        #train_temp, train_labels_temp = unison_shuffled_copies(train_temp, train_labels_temp)
        print(train_temp.shape)
        print(train_labels_temp.shape)
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))


train = np.array(image_array[1:, :], dtype=np.float32)
train_labels = np.array(label_array[1:, :], dtype=np.float32)
print(train[0].shape)
#print(train_labels)
true_labels = train_labels.argmax(-1)
print('True labels:', true_labels)
e00 = cv2.getTickCount()
time = (e00 - e0)/ cv2.getTickFrequency()
print('Loading image duration:', time)


log_model = joblib.load('result/logistic_regression.pkl')

#get prediction scores
print(log_model.score(train, true_labels))

#print confusion matrix
print(confusion_matrix(true_labels, log_model.predict(train)))
print(log_model.predict(train))