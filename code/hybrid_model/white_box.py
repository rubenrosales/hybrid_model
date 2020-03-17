import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np


def whiteBoxModel(image_dataset_path):
    '''
    function: whiteBoxModel
    Purpose: Main function that generates CNN model and evaluates on image dataset
    '''

    #define paramaters
    training_data_size = 490
    val_data_size = 210
    batch_size = 32
    epochs = 50
    train_data_path = image_dataset_path + '/train'
    val_data_path = image_dataset_path + '/val'
    test_data_path = image_dataset_path + '/test'
    img_size=224
    
    model = getCNN()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=keras.applications.inception_v3.preprocess_input,
        validation_split=0.3,
    )

    train_gen = generator.flow_from_directory(
        train_data_path, 
        batch_size=batch_size,    
        subset='training', 
        class_mode='categorical', 
        target_size=(img_size, img_size)
    )

    val_gen = generator.flow_from_directory(
        val_data_path,     
        batch_size=batch_size,
        subset='validation', 
        class_mode='categorical', 
        target_size=(img_size,img_size)
    )

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch= training_data_size//batch_size ,
        epochs= epochs,
        validation_data=val_gen,
        validation_steps= val_data_size // batch_size
    )

    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
                test_data_path,
                target_size=(img_size, img_size),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True
            ) 

    return model.predict_generator(generator)

def getCNN():
    '''
    function: getCNN
    Purpose: Create 2d cnn for classifying images
    '''
    
    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation('relu'))
     model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    return model