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
        train_temp, train_labels_temp = unison_shuffled_copies(train_temp, train_labels_temp)
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

#split dataset
X_train, X_test, y_train, y_test = train_test_split(train, true_labels, test_size=0.2)


#setup models
log_model = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

random_forest = ensemble.RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

gnb = GaussianNB()

#fit models to dataset
log_model.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
gnb.fit(X_train, y_train)

#get prediction scores
print(log_model.score(X_test, y_test))
print(random_forest.score(X_test, y_test))
print(gnb.score(X_test, y_test))

#print confusion matrix
print(confusion_matrix(y_test, log_model.predict(X_test)))
print(confusion_matrix(y_test, random_forest.predict(X_test)))
print(confusion_matrix(y_test, gnb.predict(X_test)))

#save model to load more easily
joblib.dump(log_model, 'result/logistic_regression.pkl')