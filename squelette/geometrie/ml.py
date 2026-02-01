from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from pathlib import PosixPath
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from pathlib import Path

ACCURACY_THRESHOLD = 0.98
AUTOTUNE = tf.data.experimental.AUTOTUNE

class stopOnAccurancy(tf.keras.callbacks.Callback): 
	def on_epoch_end(self, epoch, logs={}): 
		if(logs.get('accuracy') >= ACCURACY_THRESHOLD):
			print("Niveau atteint, arrêt du training")
			self.model.stop_training = True


class ML():
  def __init__(self, knowledgeRepository):
  	image_count = len(list(PosixPath(knowledgeRepository).glob('*/*.png')))
  	self.BATCH_SIZE = 32
  	self.IMG_HEIGHT = 28
  	self.IMG_WIDTH = 28
  	self.STEPS_PER_EPOCH = np.ceil(image_count/self.BATCH_SIZE)
  	self.train_ds = tf.keras.utils.image_dataset_from_directory(
  		Path(knowledgeRepository),
  		validation_split=0.2,
  		subset="training",
  		seed=123,
  		image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
  		batch_size=self.BATCH_SIZE)
  	self.val_ds = tf.keras.utils.image_dataset_from_directory(
  		Path(knowledgeRepository),
  		validation_split=0.2,
  		subset="validation",
  		seed=123,
  		image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
  		batch_size=self.BATCH_SIZE)
  	print(self.train_ds.class_names)
  	self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
  	self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)  	

  def learn(self):
  	print("model creation")
  	self.model = keras.Sequential([
  			tf.keras.layers.Rescaling(1./255),		# si besoin de /255.0, ie int to float	
  			tf.keras.layers.Flatten(input_shape=(28,28,3)), 
			tf.keras.layers.Dense(300, activation='relu'), 
			tf.keras.layers. Dense(100, activation='relu'), 
			tf.keras.layers.Dense(3, activation='softmax'),])
  	print("compilation")
  	self.model.compile(optimizer='sgd',
  		loss='sparse_categorical_crossentropy',
  		metrics=['accuracy'])
  	print('fit')
  	self.model.fit(self.train_ds,validation_data=self.val_ds,epochs=1000,callbacks=[stopOnAccurancy()])
    
  
  def predict(self, urlImage, numClass):
  	img = image.load_img(urlImage)
  	img  = image.img_to_array(img)
  	img  = img.reshape((1,) + img.shape)
  	prediction = self.model.predict(img)
  	return prediction[0][numClass]

if __name__ == '__main__':
	ml = ML('/Users/ludivine/Desktop/Cours/NfS21-22/5MI/ML/geometrie/knowledge')
	ml.learn()
	file = '/Users/ludivine/Desktop/Cours/NfS21-22/5MI/ML/geometrie/knowledge/cercle/circle0.png'
	print('ok',ml.predict(file,0))
  	