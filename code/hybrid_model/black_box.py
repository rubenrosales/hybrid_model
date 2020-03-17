from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers import Input, Dense, Concatenate
from keras.layers import LSTM
from keras.callbacks import CSVLogger
from keras.layers import merge

from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
import os
from keras.callbacks import ModelCheckpoint

import os, glob, shutil
from scipy import signal
from pywt import wavedec
import pywt

import scipy.ndimage

def resize_(before, after_size=300):
    '''
    Function: resize_
    Purpose: Resizes data using spline interpolation
    '''
    before_size = before.shape[0]
    ratio = after_size / before_size
    return scipy.ndimage.zoom(before, ratio)

def processData(data1, data2, data3, train_split = .7, random=True):
    '''
        Function: processData
        Purpose: Returns training and test data with the option of having them randomized
    '''

    #Generate array with numbers 0 - size of training data
    #then shuffle the array
    random_range = np.arange(data1.shape[0])
    if random:
        np.random.shuffle(random_range)

    labels = np.asarray([0,1,2]* (data1.shape[0] // 3))
    train_range = int(random_range.shape[0] * train_split)

    x_train = data1[random_range[:train_range]]
    x_train1 = data2[random_range[:train_range]]
    x_train2 = data3[random_range[:train_range]]

    y_train = labels[random_range[:train_range]]

    x_test =  data1[random_range[train_range:]]
    x_test1 =  data2[random_range[train_range:]]
    x_test2 =  data3[random_range[train_range:]]

    y_test = labels[random_range[train_range:]]

    shuffleImageData()
    return x_train, x_train1, x_train2, y_train, x_test, x_test1, x_test2, y_test

def shuffleImageData(random_range, train_range):
    '''
    function: shuffleImageData
    Purpose: Split data into training/val/test
    '''
    
    image_path = './images_full'

    image_files = [f for f in listdir(test_data_path) if isfile(join(test_data_path, f))]

    train_split_range = int(train_range * .7)

    train_set = image_files[random_range[:train_split_range]]
    val_set = image_files[random_range[train_split_range:train_range]]
    test_set = image_files[random_range[train_range:]]

    image_dataset_path = "../data_whitebox"

    copyData(image_path, image_dataset_path + '/train', train_set)
    copyData(image_path, image_dataset_path + '/val', val_set)
    copyData(image_path, image_dataset_path + '/test', test_set)

def copyData(current_dir, new_dir, files)
    '''
    function: copyData
    Purpose: Given a list of files and new directory, copy files into that directory
    '''
    for file in files:
        name = os.path.join(current_dir, file)
        if os.path.isfile( name ) :
            shutil.copy( name, new_dir)


def getCombinedModel(input_shape_1, input_shape_2, input_shape_3, num_classes):
    '''
    function: getCNN
    Purpose: Create model composed of 2 1-D CNN's and 1 LSTM
    '''

    kernel_size = 3

    input1 = Conv1D(32, (kernel_size), activation = "relu")(input_shape_1)
    input1 = BatchNormalization()(input1)
    input1 = Conv1D(filters = 64, kernel_size = (kernel_size),padding = 'Same', activation ='relu')(input1)
    input1 = BatchNormalization()(input1)

    input1 = Conv1D(filters = 128, kernel_size = (kernel_size),padding = 'Same', activation ='relu')(input1)
    input1 = BatchNormalization()(input1)
    input1 = Flatten()(input1)
    input1 = Dense(256, activation='relu')(input1)

    input2 = Conv1D(32, (kernel_size), activation = "relu")(input_shape_2)
    input2 = BatchNormalization()(input2)
    input2 = Conv1D(filters = 64, kernel_size = (kernel_size),padding = 'Same', activation ='relu')(input2)
    input2 = BatchNormalization()(input2)

    input2 = Conv1D(filters = 128, kernel_size = (kernel_size),padding = 'Same', activation ='relu')(input2)
    input2 = BatchNormalization()(input2)
    input2 = Flatten()(input2)
    input2 = Dense(256, activation='relu')(input2)


    input3 = Conv1D(32, (kernel_size), activation = "relu")(input_shape_3)
    input3 = BatchNormalization()(input3)
    input3 = Conv1D(filters = 64, kernel_size = (kernel_size),padding = 'Same', activation ='relu')(input3)
    input3 = BatchNormalization()(input3)

    input3 = Conv1D(filters = 128, kernel_size = (kernel_size),padding = 'Same', activation ='relu')(input3)
    input3 = BatchNormalization()(input3)
    input3 = Flatten()(input3)
    input3 = Dense(256, activation='relu')(input3)

    encode_combined = Concatenate()([input1, input2, input3])
    FC1 = Dropout(0.2)(encode_combined)
    predictions = Dense(num_classes, activation='softmax')(FC1) 

    return Model(inputs=[input_shape_1,input_shape_2, input_shape_3], outputs=[predictions])


def generateBlackBoxData(path):
    '''
    function: generateBlackBoxData
    Purpose: Process signal data bu performing wavelet decomposition on it. Then, returning data already split intro train/test sets.
    '''

    max_lev = 3

    class_names = ['h_HV_SBR_1_discrete', 'h_HV_SBR_2_discrete', 'h_HV_SBR_3_discrete']
    class_names_vv = ['h_VV_SBR_1_discrete', 'h_VV_SBR_2_discrete', 'h_VV_SBR_3_discrete']

    data1=[]
    data2=[]
    data3 = []
    num_classes = 3

    for _file in glob.glob(path):
        filename = _file
        file_contents = loadmat(filename)

        for idx in range(num_classes):
            fc = file_contents[class_names[idx]]
            fc_vv = file_contents[class_names_vv[idx]]
            
            #normalize data
            data = fc / np.linalg.norm(fc)
            data_vv = fc_vv /np.linalg.norm(fc_vv)
            
            #Assign data to a variable so we can overwrite it after every level of decomposition
            hv = data[:22]
            vv = data_vv[:22]

            for level in range(0, max_lev):

                cA_hv, cD_hv = pywt.dwt(hv, 'db4', axis=0)
                cA1_vv, cD_vv = pywt.dwt(vv, 'db4', axis=0)

                hv = cD_hv
                vv = cD_vv

                if level == 0:
                    data1.append(np.abs(cA_hv))

                if level == 1:
                    data2.append(np.abs(cA_hv))

                if level == 2:
                    data3.append(np.abs(cA_hv))  

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    data3 = np.asarray(data3)

    train_test_split = .7

    return processData(data1, data2, data3, train_test_split)

def blackBoxModel(dataset_path):
    '''
    function: blackBoxModel
    Purpose: Main function that proceses signal data and returns model with predictions
    '''

    batch_size = 8
    epochs = 20
    verbose = 0
    iterations = 3
    n_fold = 3

    dataset_path = dataset_path + '/*.mat'

    x_train, x_train_1, x_train_2, y_train, x_test, x_test1, x_test2, y_test = generateBlackBoxData(dataset_path)

    num_classes = to_categorical(y_train).shape[1]

    input_shape = Input(shape=(x_train.shape[1], x_train.shape[2]))
    input_shape_ = Input(shape=(x_train_1.shape[1], x_train_1.shape[2]))
    input_shape1 = Input(shape=(x_train_2.shape[1], x_train_2.shape[2]))

    for _ in range(n_fold):
    skf = StratifiedKFold( n_splits=n_fold, shuffle=True)
    for train, val in  skf.split(x_train, y_train):
        model = getCombinedModel(input_shape, input_shape_, input_shape1, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
                            [x_train[train], x_train_vv[train], x_train_angle_hv[train]], 
                            to_categorical(y_train[train]), 
                            batch_size=batch_size, 
                            shuffle=True,
                            epochs=epochs, 
                            verbose=verbose, 
                            validation_data = ([x_train[val], 
                                x_train_vv[val], x_train_angle_hv[val]], 
                            to_categorical(y_train[val]))
                        ) 
        del model

    return model.predict([x_test, x_test1, x_test2], y_test)