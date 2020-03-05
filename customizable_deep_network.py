import keras
import numpy as np
from keras.layers import Dense, Activation, AveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.optimizers import RMSprop, Adadelta, Adam, SGD, Adamax, Adagrad

# TODO: only square kernels
# refactor out conv2d conv2d activation batch 
# TODO: option for strides
def convolution_input (node1, node2, k1_size, k2_size, input_shape,pool1):
  model = Sequential()
  # TODO: remove hard coded 8 and give option for number of filters 
  model.add(Conv2D(8, kernel_size=(k1_size, k1_size), activation='relu',input_shape = input_shape, padding='same'))
  model.add(Conv2D(8, kernel_size=(k2_size, k2_size),padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((pool1, pool1), strides=2,padding='same'))
  return (model)

def convolution_block (model, node1, node2, k1_size, k2_size, pool):
  lnumber = len(model.layers) - 1
  output = model.layers[lnumber].output
  input_shape=output.shape[1::]
  model.add(Conv2D(node1, kernel_size=(k1_size, k1_size), activation='relu',input_shape = input_shape, padding='same'))
  model.add(Conv2D(node2, kernel_size=(k2_size, k2_size),padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((pool,pool), strides=2,padding='same'))
  return (model)

def av_pool_block (model, node1, node2, k1_size, k2_size,av_pool):
  lnumber = len(model.layers) - 1
  output = model.layers[lnumber].output
  input_shape=output.shape[1::]
  model.add(Conv2D(node1, kernel_size=(k1_size, k1_size), activation='relu',padding='same'))
  model.add(Conv2D(node2, kernel_size=(k2_size, k2_size),padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
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

  for layer in range (blocks):
    if layer == 0:
      n1 = inputl_node[x]
      n2 = inputl_node[x+1]
      k1_size = inputl_kernel[x]
      k2_size = inputl_kernel[x+1]
      max_pool = pooling_size[p]
      
      model = convolution_input(n1,n2,k1_size, k2_size, input_shape,max_pool)
      
    elif layer > 0 and layer < (blocks -2):
      k1_size = hiddenl_kernel[x]
      k2_size = hiddenl_kernel[x+1]
      n1 = hiddenl_node[x]
      n2 = hiddenl_node[x+1]
      max_pool = pooling_size[p+1]
      model = convolution_block (model, n1, n2, k1_size, k2_size, max_pool)
      x = x+2
      p = p+1
     
    elif layer == blocks-2:
         
      model = av_pool_block (model, n1_ave_pool, n2_ave_pool, k1_ave_pool, k2_ave_pool, av_pool)
    
    elif layer == blocks-1:
       model = fully_connected_block (model, val_dense, drop, n_classes)

  return(model)

def architecture_params_wrapper(blocks=5,
                                inputl_kernel=15,
                                inputl_node=8,
                                hiddenl_kernels=(7,5),
                                hiddenl_nodes=((8,16),(32,32)),
                                pooling_sizes = (7,5,3),
                                n1_ave_pool=64, 
                                n2_ave_pool=64, 
                                k1_ave_pool=3, 
                                k2_ave_pool=3, 
                                av_pool=4,
                                val_dense=512,
                                drop=0.5
                               ):
  params={}
  params['blocks'] = blocks # Each block has 5 layers = 2 Conv2D + 1 Activation + 1 Normalization + 1 Pooling


  """ 
  Helper function to make architecture inspired by deep consensus
  param blocks: Each block has 5 layers = 2 Conv2D + 1 Activation + 1 Normalization + 1 Pooling
  param k1_size:  Kernel size of the first Conv2D layer
  param K2_size: Kernel size of the second Conv2D layer
  param pool1: max pooling size
  returns: param dictionary
  """

  #inputl_kernel = [k1_size,k2_size]
  """Represents the kernel size of the first layer of the network or call input layer"""
  assert type(inputl_kernel) == int
  params['inputl_kernel'] = [inputl_kernel,inputl_kernel]

  #inputl_node =  [node1_number, node2_number]
  """Represents the number of node that each CONV2D layer has in the first layer"""
  assert type(inputl_node) == int
  params['inputl_node'] = [inputl_node,inputl_node]

  #hiddenl_kernel = [k3_size, k4_size, k5_size,k6_size,...] # Depends on the number of layers
  """Represents the kernel size of the hidden layers of the network"""
  
  params['hiddenl_kernel']=[]
  assert len(hiddenl_kernels) == blocks - 3
  for i in hiddenl_kernels:
    assert type(i) == int                         
    params['hiddenl_kernel'].extend((i,i)) # [11, 11,7, 7,5,5,3,3]

  # hiddenl_node = [node3_number, node4_number, node5_number, node6_number,...] # Depends on the number of layers
  """Represents the number of node that each CONV2D layer has in the hidden layers"""
  params['hiddenl_node']=[]
  assert len(hiddenl_nodes) == blocks - 3
  for i in hiddenl_nodes:
    assert type(i) == tuple
    assert len(i) == 2                         
    params['hiddenl_node'].extend(i) # [8,16,32,32,32,32,32,32]
  for i in params['hiddenl_node']:
    assert type(i) == int

  # pooling_size = [pool1_size, pool2_size, pool3_size, ...] # Depends on the number of layers
  """Represents the size of pooling layer in the input and hidden layer"""
  assert len(pooling_sizes) == blocks - 3 + 1
  params['pooling_size']=pooling_sizes

  # Penultimate Block
  # av_pool = [node1_number, node2_number, k1_size, k2_size, av_pool_size]
  #average_pool= [64, 64, 3, 3, 4]
  params['n1_ave_pool'] = n1_ave_pool
  params['n2_ave_pool'] = n2_ave_pool
  params['k1_ave_pool'] = k1_ave_pool
  params['k2_ave_pool'] = k2_ave_pool
  params['av_pool'] = av_pool

  # Output Layer = Full connected network
  params['val_dense'] = val_dense
  params['drop'] = drop
  
  return(params)

