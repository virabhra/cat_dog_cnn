from keras.models import Sequential
from keras.layers import  Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# add 2nd convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# add 3rd convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# add 4th layer with max_pooling
classifier.add(Conv2D(32,(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening
classifier.add(Flatten())

#Developing ANN artict
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/DS/Practice/DL/cnn_1/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/DS/Practice/DL/cnn_1/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

classifier.fit(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = len(test_set), callbacks=[es])

classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/DS/Practice/DL/cnn_1/dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'