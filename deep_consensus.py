import keras
from keras.layers import Dense, Activation, AveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam

def deep_consensus_wrapper(input_shape, # 128,128,1
	n_classes=2,
	loss='categorical_crossentropy',
	metrics=None,
	conv2d1_k=15,
	conv2d1_n=8,
	conv2d2_k=15,
	conv2d2_n=8,
	
	mp3_k=7,
	mp3_strides=2,
	conv2d4_k=7,
	conv2d4_n=8,
	conv2d5_k=7,
	conv2d5_n=16,
	
	mp6_k=5,
	mp6_strides=2,
	conv2d7_k=3,
	conv2d7_n=32,
	conv2d8_k=3,
	conv2d8_n=32,
	
	mp9_k=3,
	mp9_strides=2,
	conv2d10_k=3,
	conv2d10_n=64,
	conv2d11_k=3,
	conv2d11_n=64,
	
	ap12_k=4,
	ap12_strides=2,
	dense13_n=512,
	dropout13_rate=0.5,

	optimizer=SGD()

	):

	if metrics is None: metrics = ['categorical_accuracy']

	model = Sequential()
	#2
	# 1808 = (15*15+1)*8
	# plus one for bias
	model.add(Conv2D(conv2d1_n, kernel_size=(conv2d1_k,conv2d1_k), activation='relu',input_shape=input_shape, padding='same'))
	#3
	model.add(Conv2D(conv2d2_n, kernel_size=(conv2d2_k,conv2d2_k),padding='same'))
	# see https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
	# not sure if batch normaliztion + relu or relu + batch normaliztion
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#4
	model.add(MaxPooling2D((mp3_k,mp3_k), strides=mp3_strides,padding='same'))
	#5 
	model.add(Conv2D(conv2d4_n, kernel_size=(conv2d4_k,conv2d4_k), activation='relu',padding='same'))
	#6
	model.add(Conv2D(conv2d5_n, kernel_size=(conv2d5_k,conv2d5_k),padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#7
	model.add(MaxPooling2D((mp6_k,mp6_k), strides=mp6_strides,padding='same'))
	#8 
	model.add(Conv2D(conv2d7_n, kernel_size=(conv2d7_k,conv2d7_k), activation='relu',padding='same'))
	#9
	model.add(Conv2D(conv2d8_n, kernel_size=(conv2d8_k,conv2d8_k),padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#10
	model.add(MaxPooling2D((mp9_k,mp9_k), strides=mp9_strides,padding='same'))
	#11
	model.add(Conv2D(conv2d10_n, kernel_size=(conv2d10_k,conv2d10_k), activation='relu',padding='same'))
	#12
	model.add(Conv2D(conv2d11_n, kernel_size=(conv2d11_k,conv2d11_k),padding='same'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#13
	model.add(AveragePooling2D(pool_size=(ap12_k, ap12_k), strides=ap12_strides,padding='same'))
	###
	model.add(Flatten())
	model.add(Dense(dense13_n, activation='relu')) # 2097664 = 512*(8*8*64) ; AveragePooling2D size is 8*8*64=4096
	model.add(Dropout(rate=dropout13_rate)) # note large dropout rate

	model.add(Dense(n_classes, activation='softmax'))

	#model.summary()

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return(model)

