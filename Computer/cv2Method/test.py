import numpy as np
import glob
import cv2

image_array = np.zeros((1,38400))
label_array = np.zeros((1,4)).astype(np.float32)
testing_data = glob.glob('testing_data_temp/*.npz')

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
print(test.shape)
print(test_labels.shape)
print("Loaded Testing Data")

layer_sizes = np.int32([38400 , 32, 4])
model = cv2.ml.ANN_MLP_create()
model.setLayerSizes(layer_sizes)
model.load('mlp_xml/mlp.xml')

print("Testing started")
test_start_time = cv2.getTickCount()
ret, resp = model.predict(test)
prediction = resp.argmax(-1)
print("Prediction: ", prediction)

true_labels = test_labels.argmax(-1)
print("True labels: ", true_labels)

num_correct = np.sum(true_labels == prediction)
print("Correct predictions: ", num_correct)
test_rate = np.mean(true_labels == prediction)
print("Test rate: %f" % (test_rate*100))