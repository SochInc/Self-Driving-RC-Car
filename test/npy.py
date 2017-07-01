import numpy as np

np.save('tmp/123.npy', np.array([[1, 2, 3], [4, 5, 6]]))
x = np.load('../Computer/training_data_temp/training_data-1.npy')

print(x)