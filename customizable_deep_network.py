import keras
import numpy as np
from keras.layers import Dense, Activation, AveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.optimizers import RMSprop, Adadelta, Adam, SGD, Adamax, Adagrad

def convolution_input (node1, node2, k1_size, k2_size, input_shape,pool1):
  model = Sequential()
  model.add(Conv2D(8, kernel_size=(k1_size, k1_size), activation='relu',input_shape = input_shape, padding='same'))
  model.add(Conv2D(8, kernel_size=(k2_size, k2_size),padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((pool1, pool1), strides=2,padding='same'))
  return (model)
def convolution_block (model, node1, node2, k1_size, k2_size, pool):
  lnumber = len(model.layers) - 1
  output = model.layers[lnumber].output
  input_shape=output.shape[1::]
  model.add(Conv2D(node1, kernel_size=(k1_size, k1_size), activation='relu',input_shape = input_shape, padding='same'))
  model.add(Conv2D(node2, kernel_size=(k2_size, k2_size),padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((pool,pool), strides=2,padding='same'))
  return (model)

def av_pool_block (model, node1, node2, k1_size, k2_size,av_pool):
  lnumber = len(model.layers) - 1
  output = model.layers[lnumber].output
  input_shape=output.shape[1::]
  model.add(Conv2D(node1, kernel_size=(k1_size, k1_size), activation='relu',padding='same'))
  model.add(Conv2D(node2, kernel_size=(k2_size, k2_size),padding='same'))
  model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(AveragePooling2D(pool_size=(av_pool, av_pool), strides=2,padding='same'))
  return (model)

def fully_connected_block (model, val_dense, drop, n_classes):
  model.add(Flatten())
  model.add(Dense(val_dense, activation='relu')) 
  model.add(Dropout(rate=drop)) 
  #n_classes = np.unique(o_y_train.tolist()).size
  model.add(Dense(n_classes, activation='softmax'))
  return (model)


def build_network (blocks,input_shape, inputl_node,inputl_kernel,hiddenl_node,
                     hiddenl_kernel, pooling_size, n1_ave_pool, n2_ave_pool, 
                     k1_ave_pool, k2_ave_pool, av_pool, val_dense, drop, n_classes):
  x = 0
  p = 0
  y = 0
  model = Sequential

  for layer in range (blocks):
    if layer == 0:
      n1 = inputl_node[x]
      n2 = inputl_node[x+1]
      k1_size = inputl_kernel[x]
      k2_size = inputl_kernel[x+1]
      max_pool = pooling_size[p]
      
      model = convolution_input(n1,n2,k1_size, k2_size, input_shape,max_pool)
      
    if layer > 0 and layer < (blocks -2):
        k1_size = hiddenl_kernel[x]
        k2_size = hiddenl_kernel[x+1]
        n1 = hiddenl_node[x]
        n2 = hiddenl_node[x+1]
        max_pool = pooling_size[p+1]
        model = convolution_block (model, n1, n2, k1_size, k2_size, max_pool)
        x = x+2
        p = p+1
     
    if layer == blocks-2:
         
      model = av_pool_block (model, n1_ave_pool, n2_ave_pool, k1_ave_pool, k2_ave_pool, av_pool)
    #else :
    if layer == blocks-1:
       model = fully_connected_block (model, val_dense, drop, n_classes)

  return(model)


