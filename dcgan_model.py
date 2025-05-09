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

def save_model(model, model_path):
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(model_path)
    print(f"Model saved at {model_path}")

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

def generate_and_save_images(model, epoch, noise_dim, output_dir="generated_images", num_examples=16):
    """
    Generates and saves images using the generator model.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    noise = tf.random.normal([num_examples, noise_dim])
    generated_images = model(noise, training=False)
    generated_images = (generated_images + 1) / 2.0  # Scale to [0, 1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(num_examples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    image_path = os.path.join(output_dir, f"image_at_epoch_{epoch:04d}.png")
    plt.savefig(image_path)
    plt.close(fig)
    print(f"Generated images saved at {image_path}")

def objective(trial):
    '''
    Objective function for Optuna
    '''
    # パラメータのチューニング
    noise_dim = trial.suggest_int("noise_dim", 50, 150)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) 
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
    g_learning_rate = trial.suggest_float("d_learning_rate", 1e-5, 1e-3, log=True)
    d_learning_rate = trial.suggest_float("d_learning_rate", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)
    # kernel_size = trial.suggest_categorical("kernel_size", [2, 3])
    
    # エポック数の固定
    epochs = 1000
    
    # DCGANモデルの構築とコンパイル
    dcgan = DCGAN(noise_dim=noise_dim, dropout_rate=dropout_rate)
    dcgan.compile(
        g_optimizer=Adam(learning_rate=g_learning_rate),
        d_optimizer=Adam(learning_rate=d_learning_rate),
        loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
    
    # データセットの作成
    image_paths = glob.glob(os.path.join('train_data/*', '*.jpg'))
    save_ds(image_paths, 'dataset_images.npy')
    train_dataset = load_ds('dataset_images.npy', batch_size=batch_size)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = f'models/epoch_{timestamp}'
    
    # トレーニングプロセス
    for epoch in range(epochs):
        for image_batch in train_dataset:
            metrics = dcgan.train_step(image_batch, batch_size)
        print(f"Epoch {epoch+1}/{epochs} - Generator loss: {metrics['gen_loss']}, Discriminator loss: {metrics['disc_loss']}")
        
        if (epoch + 1) % 1 == 0:
            save_model(dcgan.generator, os.path.join(model_save_dir, f'generator_epoch_{epoch+1}.h5'))
            save_model(dcgan.discriminator, os.path.join(model_save_dir, f'discriminator_epoch_{epoch+1}.h5'))
            
        generate_and_save_images(dcgan.generator, epoch + 1, noise_dim)

    
    # 評価
    noise = tf.random.normal([batch_size, noise_dim])
    generated_images = dcgan.generator(noise, training=False)
    real_output = dcgan.discriminator(image_batch, training=False)
    fake_output = dcgan.discriminator(generated_images, training=False)
    gen_loss = dcgan.generator_loss(fake_output)
    disc_loss = dcgan.discriminator_loss(real_output, fake_output)
    
    # 評価基準
    total_loss = abs(gen_loss / disc_loss - 0.5) + (gen_loss + disc_loss)
    
    save_model(dcgan.generator, os.path.join(model_save_dir, 'generator_model_final.h5'))
    save_model(dcgan.discriminator, os.path.join(model_save_dir, 'discriminator_model_final.h5'))
    
    return total_loss




study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10, n_jobs=1, show_progress_bar=True)

# Display best parameters
print("Best trial:")
trial = study.best_trial
print("  Loss: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))




