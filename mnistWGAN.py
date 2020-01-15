import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ganmodel.WGAN import Generator
from ganmodel.WGAN import Discriminator
from ganmodel.WGAN import train_step

import argparse

import os

def load_data():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    return x_train


def plot_sample(x, n_rows=5, n_cols=5, outputdir="images", filename="fake_sample_wgan.png", figsize=(5, 5)):
    index = 0
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            axs[i, j].imshow(x[index], cmap="gray")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            index += 1
    plt.savefig(outputdir + os.path.sep + filename)


def plot_losses(disc_loss, outputdir="images", filename="loss_wgan.png", figsize=(6, 6)):
    fig = plt.figure(figsize=figsize)
    x = np.arange(len(disc_loss))
    plt.plot(x, disc_loss, label="Wasserstein distance")
    plt.legend()
    plt.savefig(outputdir + os.path.sep + filename)


def main():

    BATCH_SIZE = 256

    NOISE_DIM = 100

    parser = argparse.ArgumentParser(description="Train a WGAN on the MNIST dataset. \
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

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    epoch_w_metric = []

    for epoch in range(epochs):
        batch_idx = 0
        batch_w_metric = 0

        for batch in train_dataset:
            wasserstein = train_step(batch, generator, discriminator, NOISE_DIM,
                                     generator_optimizer,
                                     discriminator_optimizer)
            batch_w_metric += wasserstein
            batch_idx += 1

        batch_w_metric = batch_w_metric / batch_idx

        epoch_w_metric.append(wasserstein)

        print("Epoch %d / %d completed. Wasserstein distance: %.2f" %
              (epoch + 1, epochs, batch_w_metric))

    fake_data = generator(tf.random.normal([BATCH_SIZE, NOISE_DIM]))
    fake_data = np.array(fake_data).reshape((BATCH_SIZE, 28, 28))

    plot_sample(fake_data)
    plot_losses(epoch_w_metric)


if __name__ == "__main__":


    main()
