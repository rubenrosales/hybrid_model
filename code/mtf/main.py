import util
import numpy as np
from complex_signal import ComplexSignal
import random
from scipy.io import loadmat
import os, glob
from multiprocessing import Pool
from keras.preprocessing.image import array_to_img
import scipy.ndimage as sc
from scipy.stats import entropy

def resize_(before, after_size):
    '''
    Function: resize_
    Purpose: Resizes data using spline interpolation
    '''
    before_size = before.shape[0]
    ratio = after_size / before_size
    return sc.zoom(before, ratio)

def scalar_wrapper(arg):
    '''
    Function: scalar_wrapper
    Purpose: Helper function to parallelize MTF
    '''

    arg.generateMTF()
    return arg

def main():

    #define parameters for MTF
    _IMAGE_SIZE = 300
    N = 300
    _CLASSES = 3
    _TIME_BINS = 100
    _THETA_BINS = 100
    _R_BINS = 100
    _ORDER = 1
    max_level = 1
    
    data = ['h_HV_SBR_1_discrete', 'h_HV_SBR_2_discrete', 'h_HV_SBR_3_discrete']
    data_vv = ['h_VV_SBR_1_discrete', 'h_VV_SBR_2_discrete', 'h_VV_SBR_3_discrete']
    path_name = "./dataset"

    for _file in glob.glob(path_name):

      _data = []

      filename = _file
      file_contents = loadmat(filename)

      data = np.asarray(data)
      _CLASSES = data.shape[0]
      i = 0

      for _class in range(len(data)):
          r, theta, time = [], [], []

          #Read HV and VV property from data and normalize using L2 norm
          _data_hv = file_contents[data[_class]][:,1]
          _data_vv = file_contents[data_vv[_class]][:,1]
          _data_hv /= np.linalg.norm(_data_hv)
          _data_vv /= np.linalg.norm(_data_vv)

          #Define features that will have MTF applied to them
          r  = np.abs(_data_hv)
          theta = np.abs(_data_vv)
          time = ( np.abs(_data_hv) - np.abs(_data_vv) ) **2

          #Resize data to allow for larger images to be created
          r = resize_(np.abs(r), N)
          theta = resize_(np.abs(theta), N)  
          time  = resize_(np.abs(time), N)

          time, theta, r = util.processRawMatrix(time, theta, r)

          ts = ComplexSignal(time, theta, r, _TIME_BINS, _THETA_BINS, _R_BINS, _IMAGE_SIZE, _ORDER, i)

          _data.append(ts)

          i = i + 1 % _CLASSES
          if i == 0:
            i = _CLASSES

      #use multiprocessing library to run all MTF for all classes of an individual signal in parallel
      p = Pool(6)
      res = p.map( scalar_wrapper, _data)
      p.close()
      p.join()

      #Save images
      for item in res:
        pred_img = array_to_img(item.mtf)
        pred_img.save('../mtf/images/{0}/{1}_{2}.png'.format(item._class, filename, item._class))
        
if __name__== "__main__":
  main()
