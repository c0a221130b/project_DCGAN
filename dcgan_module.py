import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import optuna
from datetime import datetime
from tqdm import tqdm


class Generator(Model):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            Input(shape=(self.noise_dim,)),
            Dense(4096, use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((4, 4, 256)),
            Conv2DTranspose(128, kernel_size=8, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, kernel_size=16, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(32, kernel_size=32, strides=2, padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(1, kernel_size=64, strides=2, padding='same', use_bias=False),
        ])
        
        return model
    
    def call(self, inputs):
        return self.model(inputs)
    
class Discriminator(Model):
    def __init__(self, dropout_rate):
        super(Discriminator, self).__init__()
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            Input(shape=(64, 64, 1)),
            Conv2D(64, kernel_size=64, strides=2, padding='same'),
            LeakyReLU(),
            Dropout(self.dropout_rate),
            Conv2D(64, kernel_size=32, strides=2, padding='same'),
            LeakyReLU(),
            Dropout(self.dropout_rate),
            Conv2D(128, kernel_size=16, strides=2, padding='same'),
            LeakyReLU(),
            Dropout(self.dropout_rate),
            Flatten(),
            Dense(1024),
            Dense(1)
        ])
        
        return model
    
    def call(self, inputs):
        return self.model(inputs)

class DCGAN(Model):
    def __init__(self, noise_dim, dropout_rate):
        super(DCGAN, self).__init__()
        self.noise_dim = noise_dim
        self.generator = Generator(noise_dim)
        self.discriminator = Discriminator(dropout_rate)
        
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, images, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}
    
    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=1)
    image = tf.image.resize(image, [64, 64])
    image = (image - 255.0) / 255.0
    return image

def save_ds(image_paths, filename):
    images = []
    for path in image_paths:
        image = preprocess_image(path)
        images.append(image.numpy())
    images = np.array(images)
    np.save(filename, images)
    
def load_ds(filename, batch_size):
    dataset_images = np.load(filename)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_images)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    return dataset