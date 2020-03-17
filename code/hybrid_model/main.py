import numpy as np
import tensorflow as tf

from black_box import blackBoxModel
from white_box import whiteBoxModel

from os import listdir
from os.path import isfile, join

def getListOfClassifiedImages(matched_predictions):
    '''
    Function: getListOfClassifiedImages
    Purpose: Given indices of images, return a list of names corresponding to those indices
    '''
    test_data_path = './cnn_data/test'
    image_files = [f for f in listdir(test_data_path) if isfile(join(test_data_path, f))]

    selected_files = []
    for i in range(len(image_files)):
        if i in matched_predictions:
            selected_files.append(image_files[i])

    return selected_files
def main():
    '''
    Function: main
    Purpose: Call white-box & black-box model and get intersection of correctly classified signals
    '''

    signals_dataset_path = "../Channel_Impulse_Response"
    image_dataset_path = "../data_whitebox"

    test_results_black_box = blackBoxModel(signals_dataset_path)
    test_results_white_box = whiteBoxModel(image_dataset_path)

    # get index of predicted class
    black_box_indexes = tf.argmax(test_results_black_box, axis=1)
    black_box_indexes = black_box_indexes.eval()
    white_box_indexes = tf.argmax(test_results_white_box, axis=1)
    white_box_indexes = white_box_indexes.eval()

    #getting the intersection of correctly classified items from black-box and white-box model
    matched_predictions = np.where(white_box_indexes == black_box_indexes)

    images = getListOfClassifiedImages(matched_predictions)

if __name__== "__main__":
  main()