import tensorflow as tf
import numpy as np
from pathlib import Path
from keras.preprocessing import image
import keras

ACCURACY_THRESHOLD = 0.98
class stopOnAccurancy(tf.keras.callbacks.Callback): 
	def on_epoch_end(self, epoch, logs={}): 
		if(logs.get('accuracy') >= ACCURACY_THRESHOLD):
			print("Niveau atteint, arrêt du training")
			self.model.stop_training = True

# toutes les images doivent avoir la même taille physique
IMAGE_HEIGTH = 28
IMAGE_WIDTH = 28
BATCH_SIZE = 32

dir = '\\Users\\paul_\\OneDrive\\Documents\\ESIEA\\TP_OPS_Monitoring\\geometrie\\knowledge'
def initML():
	x_train = tf.keras.utils.image_dataset_from_directory(
		Path(dir),
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(IMAGE_HEIGTH, IMAGE_WIDTH),
		batch_size=BATCH_SIZE)
	y_train = tf.keras.utils.image_dataset_from_directory(
		Path(dir),
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(IMAGE_HEIGTH, IMAGE_WIDTH),
		batch_size=BATCH_SIZE)

	class_names = ["cercle", "rectangle", "triangle"]
	class_names = x_train.class_names
	print(class_names)
	model = tf.keras.models.Sequential([ 
		tf.keras.layers.Rescaling(1/255.0),
		tf.keras.layers.Flatten(input_shape=(IMAGE_HEIGTH, IMAGE_WIDTH,3)), 
		tf.keras.layers.Dense(300, activation='relu'), 
		tf.keras.layers. Dense(100, activation='relu'), 
		tf.keras.layers.Dense(len(class_names), activation='softmax'),
	])

	model.compile(optimizer='sgd', 
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])
	model.fit(x_train, validation_data=y_train, epochs=500, callbacks=[stopOnAccurancy()])

	#imgFile = '/Users/ludivine/Desktop/Cours/NfS21-22/5MI/ML/geometrie/knowledge/cercle/circle0.png'
	imgFile = '\\Users\\paul_\\OneDrive\\Documents\\ESIEA\\TP_OPS_Monitoring\\geometrie\\knowledge\\triangle\\triangle200.png'
	img = image.load_img(imgFile)
	img = image.img_to_array(img)
	img = img.reshape((1,) + img.shape)

	prediction = model.predict(img)
	print("prediciton da la nouvelle data", prediction)
	print('classes de la noouvelle data')
	classes = np.argmax(prediction, axis=1)
	print(classes, "ce qui veut dire", class_names[classes[0]])

	for i in range(60):
		imgFile = '\\Users\\paul_\\OneDrive\\Documents\\ESIEA\\TP_OPS_Monitoring\\geometrie\\test'+str(i)+'.png'
		img = image.load_img(imgFile)
		img = image.img_to_array(img)
		img = img.reshape((1,) + img.shape)

		prediction = model.predict(img)
		classes = np.argmax(prediction, axis=1)
		print(i, "=", class_names[classes[0]],'avec une presicion de',np.amax(prediction))

	model.save("myNetwork.h5")

if __name__ == '__main__':
	#initML()
	model = keras.models.load_model("myNetwork.h5")