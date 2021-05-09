from __future__ import print_function
import keras
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.metrics import precision_score, recall_score
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
#from sklearn.model_selection import train_test_split

batch = 18
epoch = 6
l_rate = 0.001
train_dir = "E://kaggle_malaria_detection//cell_images"
test_dir = "E://kaggle_malaria_detection//crops//res_120_test"
#Image dimensions
img_width, img_height = 64,64

#models

print(img_width,img_height)
model = Sequential()

model.add(Conv2D(16,(3,3), input_shape=(img_width, img_height, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128,(3,3), input_shape=(img_width, img_height, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation ='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1024, activation ='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

SGD = optimizers.sgd(lr = l_rate, decay= 1e-6 , momentum=0.8, nesterov=True)
model.compile(
loss='binary_crossentropy',
optimizer=SGD,
metrics=['accuracy']
)

# datagen = ImageDataGenerator(
# rescale= 1./255,
# validation_split=0.2
# )
#
# train_generator = datagen.flow_from_directory(
# directory=train_dir,
# target_size=(img_width, img_height),
# classes=['Parasitized','Uninfected'],
# class_mode='binary',
# batch_size=batch,
# subset='training'
# )
#
# validation_generator = datagen.flow_from_directory(
# directory=train_dir,
# target_size=(img_width, img_height),
# classes=['Parasitized','Uninfected'],
# class_mode='binary',
# batch_size=batch,
# subset='validation'
# )
# print('Classes : ',train_generator.class_indices)
#
# training = model.fit_generator(
# generator=train_generator,
# steps_per_epoch=22048//batch,
# epochs= epoch,
# validation_data=validation_generator,
# validation_steps=5510//batch,
# )
# print('training done')
# print('saving model...')
# model.save_weights('E://kaggle_malaria_detection//models//Malaria_cnn.h5')
# print('model saved !')

print("Classes :  {'Parasitized': 0, 'Uninfected': 1}")
model.load_weights('E://kaggle_malaria_detection//models//Malaria_cnn.h5')
src = "E://kaggle_malaria_detection//crops//res_120_test//Parasitized//180_50.png"
#src = "E://kaggle_malaria_detection//crops//res_120_test//Uninfected//90_101.png"
img = Image.open(src)
img = np.asarray(img.resize((64,64)))
img = img.reshape(1,64,64,3)
pred = model.predict_classes(img)
x= np.argmax(pred)
print(pred)
print(x)
