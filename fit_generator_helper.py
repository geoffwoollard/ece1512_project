#import keras
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#from keras.layers import Dense, Activation, AveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
#from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
#from keras.optimizers import RMSprop, Adadelta, Adam, SGD
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator

import mrc

def parse_file(fname,label,method='all',idx_list=None):
    #fname = '/Users/gw/Documents/education/2018w/ece1512/project/P11/J85/simulated_particles.mrcs'
    #label = 0
    if method == 'all':
        header = mrc.read_header(fname)
        df = pd.DataFrame({'fname':fname,'idx':range(header['nz']),'class':label})
    elif method == 'idx_list':
        assert idx_list is not None, 'supply particle indices'
        df = pd.DataFrame({'fname':fname,'idx':idx_list,'class':label})
    return(df)

def parse_files(fname_list,label_list):
    df_list = []
    for fname,label in zip(fname_list,label_list):
        df_list.append(parse_file(fname,label))
    
    return(pd.concat(df_list,axis=0))

def read_particle_from_stack(fname,idx,nx,ny):
    particles = mrc.read_imgs(fname,idx=idx,num=1).reshape(1,nx,ny)
    return(particles)

def crop(x,n_crop,nx,ny):
    x = x[:,int(nx/2-n_crop/2):int(nx/2+n_crop/2),int(nx/2-n_crop/2):int(nx/2+n_crop/2),:]
    return(x)

def read_particles(dict_list,nx,ny):
    
    particle_n=0
    particles=np.zeros((len(dict_list),nx,ny))
    for row in dict_list:

        particle = read_particle_from_stack(row['fname'],row['idx'],nx,ny)
        particles[particle_n,:,:] = particle
        particle_n+=1
    return(particles)   

def XY_from_df_batch(df_batch,nx,ny,crop_n=None, num_classes=2, do_1to3channel=False):

    dict_list = df_batch.to_dict('records')
    x_train = read_particles(dict_list,nx,ny)
    #x_train = crop(x_train,128)
    if do_1to3channel:
        X = np.stack([x_train]*3, -1)
        Y = df_batch['class'].values # https://stackoverflow.com/questions/49392972/error-when-checking-target-expected-dense-3-to-have-shape-3-but-got-array-wi
    else:
        X = x_train[:,:,:,np.newaxis]
        Y = to_categorical(df_batch['class'].values, num_classes=num_classes,dtype='int') #https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array

    if crop_n is not None: X = crop(X,crop_n,nx,ny) # match 128x128 in Deep Consensus
    return(X,Y)

def batch_image_augment(X,Y,batch_size,rotation_range=180):
    datagen = ImageDataGenerator(rotation_range=rotation_range)
    it = datagen.flow(X, Y, batch_size=batch_size)
    X,Y = it.next()
    return(X,Y)

def XY_to_3channel(X,Y):
  X3 = np.stack([X[:,:,:,0]]*3, -1)
  Y3 = np.argmax(Y,axis=1)
  return(X3,Y3)

def image_loader(df,batch_size,do_augment=False,**kwargs):
    
    L = df.shape[0]
    while True: #this line is just to make the generator infinite, keras needs that
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end,L)
            X, Y = XY_from_df_batch(df.iloc[batch_start:limit],**kwargs)
            if do_augment: X,Y = batch_image_augment(X,Y,batch_size=limit-batch_start)
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples 
            batch_start += batch_size   
            batch_end += batch_size





def main():
	fname_list = ['/Users/gw/Documents/education/2018w/ece1512/project/P11/J75/simulated_particles.mrcs',
              '/Users/gw/Documents/education/2018w/ece1512/project/P11/J103/simulated_particles.mrcs']
	label_list = [0,1]
	df = parse_files(fname_list,label_list)
	df = df.sample(df.shape[0])
	header = mrc.read_header(fname_list[0])
	nx,ny=header['nx'],header['nx']
	val_n = 2000
	df = df.iloc[:-val_n]
	df_val = df.iloc[-val_n:]