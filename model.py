'''
Model.py
Steering Angles
Evan Solty

'''

import os
os.environ['KERAS_BACKEND'] = "tensorflow"
import cv2
import tensorflow
import pickle
from os import listdir
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy import random
import pandas as pd
import math
import json

from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, BatchNormalization,Input, ELU

def last_preprocess(image):
	''' Crops, resizes, converts to YUV color scale and scales values to be between 
	-0.5 and 0.5. Input is cv2.imread(path_to_image)'''
	shape = image.shape
	# cropping image to remove horizon and dash of car
	image = image[math.floor(shape[0]/3):shape[0]-20, 0:shape[1]]
	#scale/resizing image, inter_area used for shrinking, 66,200 used in nvidea paper
	image = cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)
	#switch to YUV so color and brightness are separated
	image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	# normalize by constant getting range of -.5 to .5 for faster sgd
	image = (image-128.)/256

	return image
	
def generate_train_data(data, batch_size):
	''' Reads data from pandas dataframe. Reduce chance of generating mostly images
	with zero steering values. Process, transform, and flip images and steering values
	'''
	images = np.zeros((batch_size, 66, 200, 3))
	steering = np.zeros(batch_size)
	while 1:
		
		for row in range(batch_size):
			line = random.randint(len(data)-1) 
			y = data.iloc[line]['steering']
			
			# get random number to determine center, left, right
			image_pos = random.randint(2)
			# Reduce the probability of image with steering equals zero as there are many of these
			reduce_prob = random.randint(100)
			
			if image_pos == 2:
				if y > 0:
					image = cv2.imread(data.iloc[line]['center'].strip())
				elif y == 0 and reduce_prob <= 50:
					image = cv2.imread(data.iloc[line]['center'].strip())
				else:
					image_pos = random.randint(1)

			if image_pos == 0:
				image = cv2.imread(data.iloc[line]['left'].strip())
				y += 0.25

			if image_pos >= 1:
				image = cv2.imread(data.iloc[line]['right'].strip())
				y -= 0.25
				
			image = last_preprocess(image)
			image,y = image_transforms(image,y)
			image,y = flip_image(image,y)
			
			images[row] = image
			steering[row] = y

		yield images, steering

def flip_image(image,y):
	'''Horizontally flip images'''
	chance = random.randint(1)
	if chance == 1:
		image = cv2.flip(image,1)
		if y > 0: y = -y
		elif y < 0: y = abs(y)
		return image,y
	else:
		return image,y
        
def image_transforms(image,y):
	'''Shifts images vertically and horizontally with a %50 with random number of pixels 
	shifted by. Rotates	some images'''
	chance = random.randint(10)
	# vertical and horizonal shifts for images with low steering values
	if y < 0.3 and chance <= 5:
		rows, col, ch = image.shape
		x_shift = random.randint(80)
		y_shift = random.randint(19)
		pts1 = np.float32([[100,10],[100,50],[120,10]])
		pts2 = np.float32([[100+x_shift,y_shift],[100+x_shift,40+y_shift],[120+x_shift,y_shift]])
		M = cv2.getAffineTransform(pts1,pts2)
		image = cv2.warpAffine(image,M,(col,rows))
		
		# each pixel shifted adds 0.008 to steering
		y += x_shift*0.008
		
		# steering value needs to be between -1 and 1
		if y > .9:
			y = .9
			

	# rotate    
	if y == 0 and chance >= 5 and chance <= 6:
		angle = random.randint(20)
		rot = cv2.getRotationMatrix2D((100,50), 360-angle, 1.0)
		rotated = cv2.warpAffine(image, rot, (200, 66))
		y = angle/60
		
		return image,y
	else:
		return image,y

def steering_model():
	''' 9 layer model based on Nvidia paper with dropouts added'''		
	model = Sequential()
	model.add(Convolution2D(24, 5, 5, subsample= (2, 2), name='cv2d1', input_shape=(66, 200, 3)))
	model.add(Dropout(0.25))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5, subsample= (2, 2), name='cn2d2'))
	model.add(Dropout(0.25))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5, subsample= (2, 2), name='cn2d3'))
	model.add(Dropout(0.25))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='cn2d4'))
	model.add(Dropout(0.25))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='cn2d5'))
	model.add(Dropout(0.25))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(1164, name = "dense_0"))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(100, name = "dense_1"))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(50, name = "dense_2"))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))
	model.add(Dense(10, name = "dense_3"))
	model.add(Activation('relu'))
	model.add(Dense(1, name = "dense_4"))
	model.compile(loss = 'mse', optimizer = 'Adam')
	return model


if __name__ == '__main__':
	csv_file = 'driving_log.csv'
	data_df = pd.read_csv(csv_file)
	
	# batch size of 250 selected due to cpu
	batch_size = 250
	train_generator = generate_train_data(data_df,batch_size)
	steering_model().fit_generator(train_generator,samples_per_epoch=25000, nb_epoch=5, verbose=1)

	json_file = model.to_json()
	with open('model.json', 'w') as f:
		json.dump(json_file,f,ensure_ascii=False)

	model.save_weights('model.h5')