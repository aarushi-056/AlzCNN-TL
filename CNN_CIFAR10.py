import matplotlib.pyplot as plt
import pickle
from keras.layers import Input, Conv2D, Activation, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import cifar10
# Import ImageDataGenerator from tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_val, y_val) = cifar10.load_data()
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

input = Input(shape=(32, 32, 3))
X = Conv2D(64, (1, 1))(input)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Conv2D(64, (3, 3))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Conv2D(64, (5, 5))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Dropout(0.25)(X)
X = MaxPool2D((2,2))(X)

X = Conv2D(128, (1, 1))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Conv2D(128, (3, 3))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Conv2D(128, (5, 5))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Dropout(0.25)(X)
X = Conv2D(256, (1, 1))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Conv2D(256, (3, 3))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Conv2D(256, (5, 5))(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Dropout(0.25)(X)
X = Flatten()(X)
output = Dense(10, activation="softmax")(X)

model = Model(input, output)


model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=50,
    horizontal_flip=True)
validationgen = ImageDataGenerator(
    rescale=1./255)


# Replace fit_generator with fit and pass the generator directly
datagen.fit(X_train)
validationgen.fit(X_val)
history = model.fit(datagen.flow(X_train, y_train, batch_size=128),
                    steps_per_epoch=len(X_train) // 128, # Use floor division to ensure an integer value
                    validation_data=validationgen.flow(X_val, y_val), epochs=700).history

with open("history.dat", "wb") as fp:
    pickle.dump(history, fp)
