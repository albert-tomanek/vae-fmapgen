from typing import Tuple, List, Dict
import random
import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras import Model
from keras.preprocessing import image as kp_image
from keras.layers import Layer, Input, Lambda, concatenate, Dense, Dropout, LSTM, GaussianNoise, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, DepthwiseConv2D, Flatten
from keras.layers.core import Reshape, Activation
from keras.callbacks import LambdaCallback
from keras.losses import mean_squared_error

import tensorflow as tf

from keras.datasets import fashion_mnist, mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import wandb
wandb.init(project="vae-maskgen")

BATCH_SIZE = 32

class MNISTAutoEncoder:
	def __init__(self, repr_size=8):
		self.encoder = self.create_encoder(repr_size)
		self.decoder = self.create_decoder(repr_size)

		inp  = out = Input(shape=(28, 28))
		repr = self.encoder(out)
		qtz  = Lambda(lambda x: tf.one_hot(K.argmax(x), repr_size))(repr)
		out  = self.decoder(qtz)

		self.model = Model(inp, [out, repr])
		self.model.compile('rmsprop', loss=lambda true, out: mean_squared_error(true, out[0]))	# out = (pred, repr)

	@staticmethod
	def create_encoder(repr_size):
		inp = out = Input(shape=(28, 28))
		out = Reshape((28, 28, 1))(out)

		out = Conv2D(64, (4, 4), strides=1, padding='same')(out)	# -> 28x28x64
		out = Conv2D(repr_size, 1, padding='same')(out)	# -> 28x28x1
		out = BatchNormalization()(out)
		out = Activation('sigmoid')(out)

		m=Model(inp, out)
		# m.summary()
		return m

	@staticmethod
	def create_decoder(repr_size):
		inp = out = Input(shape=(28, 28, repr_size,))					# -> 28x28x8
		out = Conv2DTranspose(64, 1, strides=1, padding='same')(out)	# -> 28x28x64
		out = Conv2DTranspose(1, 1, strides=1, padding='same')(out)		# -> 28x28x1
		# out = Dense(28*28)(out)
		out = Reshape((28, 28))(out)

		m=Model(inp, out)
		m.summary()
		return m

	def train(self, gen):
		test_batch = next(gen)[0]

		self.model.fit_generator(gen, epochs=120, steps_per_epoch=50, callbacks=[LambdaCallback(
			on_batch_end=self.on_batch_end,
			on_epoch_end=lambda *args: self.on_epoch_end(*args, random.choice(test_batch)),
		)])

	def on_epoch_end(self, epoch, logs, sample_img):
		out, repr = self.model.predict(sample_img.reshape((1, *sample_img.shape)))

		wandb.log({
			"img_x": wandb.Image(sample_img),
			"img_map": wandb.Image(fmap_palette[np.argmax(repr, axis=-1)]),
			"img_y": wandb.Image(out),
		})

	def on_batch_end(self, batch, logs):
		wandb.log({
			"loss": logs['loss'],
		})

	@staticmethod
	def datagen():
		while True:
			imgs = x_train[np.array([random.randint(0, len(x_train)-1) for i in range(BATCH_SIZE)])]
			yield imgs, [imgs, np.zeros((BATCH_SIZE, 28, 28, 8))]	# The repr is a placeholder and isn't actually used to calculate loss.

fmap_palette = np.array([
	[  0,   0,   0],
	[128,   0,   0],
	[  0, 128,   0],
	[128, 128,   0],
	[  0,   0, 128],
	[128,   0, 128],
	[  0, 128, 128],
	[192, 192, 192],
	[128, 128, 128],
	[255,   0,   0],
	[  0, 255,   0],
	[255, 255,   0],
	[  0,   0, 255],
	[255,   0, 255],
	[  0, 255, 255],
	[255, 255, 255]
])

if __name__ == "__main__":
	model = MNISTAutoEncoder()

	if input('Load ENcoder? [y/N]: ') == 'y':
		model.encoder.load_weights('weights_enc.h5')
	if input('Load DEcoder? [y/N]: ') == 'y':
		model.decoder.load_weights('weights_dec.h5')

	model.encoder.summary()
	model.decoder.summary()

	try:
		model.train(model.datagen())
	finally:
		model.encoder.save_weights('weights_enc.h5')
		model.decoder.save_weights('weights_dec.h5')
