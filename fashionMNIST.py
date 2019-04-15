import pandas as pd 
import numpy as np 
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Dropout, Conv2D
from keras import backend as kerasBackend
from keras.utils import np_utils


import loadData 
import plot

train_data_path = "dataset/fashion-mnist_train.csv"
test_data_path = "dataset/fashion-mnist_test.csv"

# initialize the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]

image_size = [28,28]
image_depth = 1
num_classes = 10
EPOCHS = 5
BATCH_SIZE = 32

image_height = image_size[0]
print("image_height = ",image_height )
image_width = image_size[1]
print("image_width = ",image_width )


#load training data
load_data = loadData.LoadData(image_size)

train_data = pd.read_csv(train_data_path)
x_train, y_train = load_data.load_formatted_data(data = train_data)

test_data = pd.read_csv(test_data_path)
x_test, y_test = load_data.load_formatted_data(data = test_data)

num_train_samples = x_train.shape[0]
num_test_samples = x_test.shape[0]



x_train = x_train.reshape((num_train_samples, image_height, image_width, image_depth))
x_test = x_test.reshape((num_test_samples, image_height, image_width, image_depth))
input_shape = (image_height, image_width, image_depth)
 

print("\nx_train.shape  = ",x_train.shape)
print("x_test.shape  = ",x_test.shape)

def data_normalize(x, max_value):
	return x.astype("float32") / max_value

# scale data to the range of [0, 1]
x_train = data_normalize(x_train, 255.0)
x_test = data_normalize(x_test, 255.0)


# one-hot encode the training and testing labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

History = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

print(History.history.keys())

plot = plot.Plot(History)
plot.accuracy()
plot.loss()
