import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, load_model
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dcgan_module import Generator, Discriminator, DCGAN, save_ds, load_ds
import dcgan_module
import glob
import optuna
from datetime import datetime

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

def generate_and_save_images(model, epoch, noise_dim, output_dir="fine_images", num_examples=16):
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

# Load pretrained generator
pretrained_generator_path = "models/epoch_20250107_125113/generator_epoch_888.h5"

# generator = Generator(noise_dim=100)
# print(generator.shape)
# generator.load_weights(pretrained_generator_path)
# Adjust noise_dim to match the pretrained model
# Initialize model variables by calling it with dummy input

print("Pretrained generator loaded.")

# Generate and visualize new images
def generate_and_visualize_images(generator, noise_dim, num_examples=16):
    noise = tf.random.normal([num_examples, noise_dim])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images + 1) / 2.0  # Scale to [0, 1]

    fig = plt.figure(figsize=(4, 4))
    for i in range(num_examples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig("fine_images/final/final.png")

def objective(trial):
    noise_dim = 104
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
    g_learning_rate = trial.suggest_float("d_learning_rate", 1e-5, 1e-3, log=True)
    d_learning_rate = trial.suggest_float("d_learning_rate", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.7)
    
    generator = Generator(noise_dim=104)
    generator(tf.random.normal([1, 104]))
    generator.load_weights(pretrained_generator_path)
    print(generator.summary())
    
    
    # Define a simple discriminator for the new task
    discriminator = Discriminator(dropout_rate=0.4)

    image_paths = glob.glob("igarashi_tenni/*.jpg")
    save_ds(image_paths, 'tuning_images.npy')
    new_dataset = load_ds('tuning_images.npy', batch_size=batch_size)

    model_save_dir = f'models/tuned/'
    
    # Recompile DCGAN model
    dcgan = DCGAN(noise_dim=noise_dim, dropout_rate=0.4)
    for layer in generator.layers[:-2]:
        layer.trainable = False
    dcgan.generator = generator  # Use pretrained generator
    
    
    dcgan.discriminator = discriminator
    dcgan.compile(
        g_optimizer=Adam(learning_rate=1e-4),
        d_optimizer=Adam(learning_rate=1e-4),
        loss_fn=BinaryCrossentropy(from_logits=True)
    )

    # Fine-tune the model
    epochs = 10000
    for epoch in range(epochs):
        for image_batch in tqdm(new_dataset, desc=f"Epoch {epoch+1}/{epochs}"):
            metrics = dcgan.train_step(image_batch, batch_size=batch_size)
        print(f"Epoch {epoch+1}: Generator loss: {metrics['gen_loss']:.4f}, Discriminator loss: {metrics['disc_loss']:.4f}")
        
        # if (epoch + 1) % 1 == 0:
        #     save_model(dcgan.generator, os.path.join(model_save_dir, f'generator_epoch_{epoch+1}.h5'))
        #     save_model(dcgan.discriminator, os.path.join(model_save_dir, f'discriminator_epoch_{epoch+1}.h5'))
            
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
    # Save fine-tuned generator
    # save_model(dcgan.generator, os.path.join(model_save_dir, 'tuned_generator.h5'))
    save_model(dcgan.generator, os.path.join(model_save_dir, f'generator/tuned_generator_final_epoch_{datetime.now().strftime("%y%m%d_%H%M%S")}.h5'))
    save_model(dcgan.discriminator, os.path.join(model_save_dir, f'discriminator/tuned_discriminator_final_epoch_{datetime.now().strftime("%y%m%d_%H%M%S")}.h5'))
    print(f"Tuned generator saved at {model_save_dir}")
    
    # Visualize fine-tuned generator results
    generate_and_visualize_images(generator, noise_dim=noise_dim)
    
    return total_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5, n_jobs=1, show_progress_bar=False)

# Display best parameters
print("Best trial:")
trial = study.best_trial
print("  Loss: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))




