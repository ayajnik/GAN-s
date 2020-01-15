import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ganmodel.GAN import Generator
from ganmodel.GAN import Discriminator
from ganmodel.GAN import train_step

import argparse

import os

def load_data():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    return x_train


def plot_sample(x, n_rows=5, n_cols=5, outputdir="images", filename="fake_sample.png", figsize=(5, 5)):
    index = 0
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].imshow(x[index], cmap="gray")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            index += 1
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    plt.savefig(outputdir + os.path.sep + filename)


def plot_losses(gen_loss, disc_loss, outputdir="images", filename="losses.png", figsize=(6, 6)):
    fig = plt.figure(figsize=figsize)
    x = np.arange(len(gen_loss))
    plt.plot(x, gen_loss, label="Generator")
    plt.plot(x, disc_loss, label="Discriminator")
    plt.legend()
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    plt.savefig(outputdir + os.path.sep + filename)


def main():

    tf.random.set_seed(1234)
    
    BATCH_SIZE = 256

    NOISE_DIM = 100

    parser = argparse.ArgumentParser(description="Train a GAN on the MNIST dataset. \
                                     This work is licensed \
                                     under the Creative Commons Attribution \
                                     4.0 International License. (C) Nikolay Manchev 2019")

    parser.add_argument("--lr", help="Learning rate",
                        default=0.0001, required=False, type=float)
    parser.add_argument("--epochs", help="Maximum number of training epochs",
                        default=5, required=False, type=int)

    args = parser.parse_args()

    x_train = load_data()

    epochs = args.epochs

    learning_rate = args.lr

    train_dataset = (tf.data.Dataset.from_tensor_slices(x_train.reshape(x_train.shape[0], 784)).batch(BATCH_SIZE))

    generator = Generator(784, NOISE_DIM)
    discriminator = Discriminator(784)
    
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    epoch_gen_loss = []
    epoch_disc_loss = []

    for epoch in range(epochs):
        batch_idx = 0
        batch_gen_loss = 0
        batch_disc_loss = 0

        for batch in train_dataset:
            gen_loss, disc_loss = train_step(batch, generator, discriminator, NOISE_DIM,
                                             generator_optimizer,
                                             discriminator_optimizer)
            batch_gen_loss += gen_loss
            batch_disc_loss += disc_loss
            batch_idx += 1

        batch_gen_loss = batch_gen_loss / batch_idx
        batch_disc_loss = batch_disc_loss / batch_idx

        epoch_gen_loss.append(batch_gen_loss)
        epoch_disc_loss.append(batch_disc_loss)

        print("Epoch %d / %d completed. Generator loss: %.2f Discrimnator loss: %.2f" %
              (epoch + 1, epochs, batch_gen_loss, batch_disc_loss))

    fake_data = generator(tf.random.normal([BATCH_SIZE, NOISE_DIM]))
    fake_data = np.array(fake_data).reshape((BATCH_SIZE, 28, 28))

    plot_sample(fake_data)
    plot_losses(epoch_gen_loss, epoch_disc_loss)


if __name__ == "__main__":


    main()
