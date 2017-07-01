import numpy as np
import glob
import cv2

print("Loading training data")
load_time_start = cv2.getTickCount()

image_array = np.zeros((1,38400))
label_array = np.zeros((1,4),'float')
training_data = glob.glob('training_data_temp/*.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
        print(data.files)
        train_temp = data['train']
        train_labels_temp = data['train_labels']
        print(train_temp.shape)
        print(train_labels_temp.shape)
    image_array = np.vstack((image_array,train_temp))
    label_array = np.vstack((label_array,train_labels_temp))

train = image_array[1:,:]
train_labels = label_array[1:,:]
print(train.shape)
print(train_labels.shape)

load_time_end = cv2.getTickCount()
load_time = (load_time_end - load_time_start)/ cv2.getTickFrequency()

print('Loading Time: ',load_time)

print("Training Started")
train_time_start = cv2.getTickCount()

# creating Neural Net
layer_sizes = np.int32([38400, 72, 4])
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(layer_sizes)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,500,0.0001)
criteria2 = (cv2.TERM_CRITERIA_COUNT, 100,0.001)
model.setTermCriteria(criteria)
model.setBackpropWeightScale(0.001)
model.setBackpropMomentumScale(0.0)
params = dict(term_crit = criteria,
              train_method = cv2.ml.ANN_MLP_BACKPROP,
              bp_dw_scale = 0.001,
              bp_moment_scale = 0.0)

print("Training MLP")

num_iter = model.train(train.astype(np.float32), cv2.ml.ROW_SAMPLE, train_labels.astype(np.float32))
train_time_end = cv2.getTickCount()
train_time = (train_time_end - train_time_start) / cv2.getTickFrequency()
print("Training Completed in: ", train_time)

model.save('mlp_xml/mlp.xml')

print("Ran for %d iterations" %num_iter)



for single_npz in testing_data:
    with np.load(single_npz) as data:
        test_temp = data['train']
        test_labels_temp = data['train_labels']
        print(test_temp.shape)
        print(test_labels_temp.shape)
    image_array = np.vstack((image_array, test_temp))
    label_array = np.vstack((label_array, test_labels_temp))

test = image_array[1:, :]
test_labels = label_array[1:, :]


ret, resp = model.predict(test)
prediction = resp.argmax(-1)
print("Prediction: ", prediction)

true_labels = test_labels.argmax(-1)
print("True labels: ", true_labels)

num_correct = np.sum(true_labels == prediction)
print("Correct predictions: ", num_correct)
test_rate = np.mean(true_labels == prediction)
print("Test rate: %f" % (test_rate*100))