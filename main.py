from typing import Tuple, List, Dict
import random, os
from math import log10
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import keras
import keras.backend as K
from keras import Model
from keras.preprocessing import image as kp_image
from keras.layers import Layer, Input, Lambda, concatenate, Dense, Dropout, LSTM, GaussianNoise, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, DepthwiseConv2D, Flatten
from keras.layers.core import Reshape, Activation
from keras.callbacks import LambdaCallback
from keras.losses import mean_squared_error, binary_crossentropy
from keras.optimizers import Adam

import tensorflow as tf

BATCH_SIZE = 32
EPOCHS = 240
repr_size = 8
imgdims = (96, 96)

def identical_shuffle(a: np.array, b: np.array):
	order = np.arange(len(a))
	np.random.shuffle(order)
	return a[order], b[order]

def shift(a: np.array, n):
	b = a.copy()
	b[n:] = b[:-n]
	b[:n] = a[-n:]
	return b

@tf.custom_gradient
def quantize_fn(x):
	def grad(dy):
		return dy * 0	# Gradient is *probably* 0
	return tf.one_hot(tf.argmax(x, axis=-1), repr_size), grad

def img_variety(x):
	a = tf.reshape(x, [-1])					# Make it a 1D array
	b = tf.concat([tf.zeros((1), dtype=x.dtype), a[:-1]], axis=-1)	# Shift by 1
	return tf.reduce_sum(tf.square(tf.math.tanh(tf.cast(a - b, tf.float32))))	# The tanh is there to make all differences greater than 0 the same size.

class AutoEncoder:
	def __init__(self):
		self.encoder = self.create_encoder(repr_size)
		self.decoder = self.create_decoder(repr_size)

		inp  = out = Input(shape=(*imgdims, 1))
		repr = self.encoder(out)
		qtz  = Lambda(quantize_fn)(repr)		# Puts a 1 at only those positions with the highest value. The rest is 0 so when multiplied by the original will be blank.
		out  = self.decoder(qtz)
		out  = Lambda(lambda x: x * 255)(out)

		self.model = Model(inp, [out, repr], name='vae')
		self.model.compile(Adam(lr=10e-4), loss=lambda true, out: mean_squared_error(true, out[0]))	# out = (pred, repr)

	@staticmethod
	def create_encoder(repr_size):
		inp = out = Input(shape=(*imgdims, 1))
		out = Reshape((*imgdims, 1))(out)

		out = Conv2D(64, (8, 8), strides=1, padding='same')(out)	# -> 28x28x64
		out = Conv2D(repr_size, 1, padding='same')(out)				# -> 28x28x1
		out = BatchNormalization()(out)
		# out = Activation('sigmoid')(out)

		return Model(inp, out, name='encoder')

	@staticmethod
	def create_decoder(repr_size):
		inp = out = Input(shape=(*imgdims, repr_size,))				# -> 28x28x8
		out = Conv2D(64, (3, 3), strides=1, padding='same')(out)	# -> 28x28x64
		out = Conv2D(1, (8, 8), strides=1, padding='same')(out)		# -> 28x28x1
		# out = Dense(28*28)(out)
		out = BatchNormalization()(out)
		out = Activation('sigmoid')(out)
		out = Reshape((*imgdims, 1))(out)

		return Model(inp, out, name='decoder')

	def train(self, gen):
		test_batch = next(gen)[0]

		self.model.fit_generator(gen, epochs=EPOCHS, steps_per_epoch=50, callbacks=[LambdaCallback(
			on_batch_end=self.on_batch_end,
			on_epoch_end=lambda *args: self.log_example(gen)
		)])

	def log_example(self, gen):
		sample_img = random.choice(next(gen)[0])
		out, repr = self.model.predict(sample_img.reshape((1, *sample_img.shape)))

		wandb.log({
			"img_x": wandb.Image(sample_img),
			"img_map": wandb.Image(fmap_palette[np.argmax(repr, axis=-1)]),
			"img_y": wandb.Image(out),
			"repr_variance": float(img_variety(repr))
		})

	def on_batch_end(self, batch, logs):
		wandb.log({
			"loss": logs['loss'],
		})

	@staticmethod
	def datagen():
		textures = [Image.open('textures/' + file) for file in os.listdir('textures') if file.endswith('.jpg')]
		while True:
			batch = np.zeros((BATCH_SIZE, *imgdims, 3))
			for i in range(BATCH_SIZE):
				img = np.array(random.choice(textures).copy())
				for _ in range(4):
					polygon = AutoEncoder.triangle(*imgdims)
					mask = Image.new('1', imgdims, 0)
					ImageDraw.Draw(mask).polygon(polygon, fill=1, outline=1)
					mask = np.array(mask, dtype=np.uint8).repeat(3, axis=-1).reshape((*imgdims, 3))
					overlay = np.array(random.choice(textures))
					img = img * (1 - mask) + overlay * mask

				batch[i] = img
			mean = np.mean(batch, -1)
			mean = mean.repeat(1, -1).reshape(BATCH_SIZE, *imgdims, 1)
			batch = mean
			yield batch, [batch, np.zeros((BATCH_SIZE, *imgdims, repr_size))]	# The repr is a placeholder and isn't actually used to calculate loss.

	@staticmethod
	def triangle(max_x, max_y):
		min_sz=max_x // 3
		t = []
		t.append((random.randint(0, max_x - min_sz), random.randint(0, max_y - min_sz)))
		t.append((t[0][0] + random.randint(min_sz, max_x - t[0][0]), t[0][1] + random.randint(min_sz, max_y - t[0][1])))
		t.append((random.randint(t[0][1], t[1][1]), random.randint(t[0][0], t[1][0])))
		return t

class GANModel:
	def __init__(self):
		self.vae = AutoEncoder()
		self.discriminator = self.create_discriminator()
		self.discriminator.compile('adam', loss='binary_crossentropy')

		inp = Input(shape=(*imgdims, 1))
		fake, repr = self.vae.model(inp)
		self.discriminator.trainable = False
		disc = self.discriminator(fake)

		self.complete_model = Model(inp, disc, name='vae_gan')
		self.complete_model.compile('adam', loss=lambda true, out: binary_crossentropy(true, out) + img_variety(tf.argmax(out[1])))

	@staticmethod
	def create_discriminator():
		inp = Input(shape=(*imgdims, 1))
		out = Conv2D(2*repr_size, kernel_size=(8, 8), strides=4)(inp)
		out = Conv2D(4*repr_size, kernel_size=(4, 4), strides=2)(out)
		out = Flatten()(out)
		out = Dense(1, activation='sigmoid')(out)

		return Model(inp, out, name='disc')

	def train(self, datagen):				# The generater is expected to output in the format of AutoEncoder.datagen
		for i in range(EPOCHS):
			imgs: np.ndarray = next(datagen)[0]
			HALF_BATCH = BATCH_SIZE // 2

			# Train discriminator to discriminate
			x_jumbled = np.zeros((BATCH_SIZE, *imgdims, 1))
			x_jumbled[HALF_BATCH:] = self.vae.model.predict(imgs[HALF_BATCH:])[0]
			x_jumbled[:HALF_BATCH] = imgs[HALF_BATCH:]

			y_jumbled = np.zeros((BATCH_SIZE))
			y_jumbled[:HALF_BATCH] = 1
			x_jumbled, y_jumbled = identical_shuffle(x_jumbled, y_jumbled)
			# disc_batch_size = BATCH_SIZE / log10(vae_loss) - log10(disc_loss)

			history = self.discriminator.fit(x_jumbled, y_jumbled)
			wandb.log({"disc_loss": history.history['loss'][0]})

			# Train VAE (generator) to fake
			history = self.complete_model.fit(imgs[:BATCH_SIZE], np.ones(BATCH_SIZE))
			wandb.log({"gen_loss": history.history['loss'][0]})

			if i % 8 == 0:
				self.vae.log_example(datagen)

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

def main():
	opts = input('Load Encoder? Decoder? Discriminator? [nnn]: ')
	opts = opts if opts != '' else 'nnn'
	if opts[0] == 'y':
		model.vae.encoder.load_weights('weights_enc.h5')
	if opts[1] == 'y':
		model.vae.decoder.load_weights('weights_dec.h5')
	if opts[2] == 'y':
		model.discriminator.load_weights('weights_disc.h5')

	model.discriminator.summary()
	model.complete_model.summary()

	try:
		model.train(model.vae.datagen())
	finally:
		model.vae.encoder.save('weights_enc.h5')
		model.vae.decoder.save('weights_dec.h5')
		model.discriminator.save('weights_disc.h5')

if __name__ == "__main__":
	import wandb
	wandb.init(project="vae-maskgen")
	main()
