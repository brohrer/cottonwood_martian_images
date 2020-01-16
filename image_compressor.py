"""
A neural net-powered image compressor

This tool compresses a directory full of images by:
1. Using them to train an autoencoder. The autoencoder learns a reduced
    representation of the patterns in the images.
2. Compressing the images and saving them to another directory.

It can also be used to decompress the images by decoding them with the
same autoencoder and writing the reconstituted images to another directory.

To run the compressor on a set of images from the surface of Mars,
run from the command line:

    python3 image_compressor.py

You can see the code this executes at the end of this file and use it for
a template for future use.
"""
import os
import pickle as pkl
import numpy as np
from PIL import Image

from cottonwood.core.activation import Tanh, Logistic, ReLU
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
from cottonwood.core.initializers import Glorot, He
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum
from cottonwood.examples.autoencoder.autoencoder_viz import Printer
import image_loader as ldr

# int, the number of pixel rows and columns in the square image patches
# that the autoencoder works with
patch_size = 11


def train(
    image_path,
    activation_function=Tanh,
    initializer=Glorot,
    learning_rate=1e-4,
    n_nodes_0=79,
    n_nodes_1=9,
    n_nodes_2=79,
):
    """
    Train an autoencoder to represent image patches more economically.

    image_path: str, a path to the directory containing the images
        that are to be compressed. If this is a relative path, it needs to be
        relative to the directory from which this module is run.
    activation_function: one of the classes available in
        cottonwood/core/activation_functions.py
        As of this writing, {Tanh, Sigmoid, ReLU}
    initializer: one of the classes available in
        cottonwood/core/initializers.py
        As of this writing, {Glorot, He}
    learning_rate: float, the learning rate for the Momentum optimizers
        that gets called during backpropagation. Feasible values will probably
        be between 1e-5 and 1e-3.
    n_nodes_x: int, the number of nodes in layer x. Layer 1 is
        the narrowest layer, and its node activities
        are used as the representation
        of the compressed patch.
    """
    training_patches = ldr.get_training_data(patch_size, image_path)

    sample = next(training_patches)
    printer = Printer(input_shape=sample.shape)
    n_pixels = np.prod(sample.shape)
    n_nodes_dense = [n_nodes_0, n_nodes_1, n_nodes_2]
    n_nodes = n_nodes_dense + [n_pixels]

    printer = Printer(input_shape=sample.shape)

    layers = []

    layers.append(RangeNormalization(training_patches))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=activation_function,
            initializer=initializer,
            previous_layer=layers[-1],
            optimizer=Momentum(
                learning_rate=learning_rate,
                momentum_amount=.9,
            )
        )
        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
        layers=layers,
        error_function=Sqr,
        n_iter_train=5e6,
        n_iter_evaluate=1e4,
        n_iter_evaluate_hyperparameters=9,
        printer=printer,
        verbose=True,
        viz_interval=1e6,
    )
    autoencoder.train(training_patches)
    return autoencoder


def compress(autoencoder, image_path, compressed_path):
    """
    Represent each of the images as the activities of the narrowest layer
    in the autoencoder.

    autoencoder: the neural network trained on the images
    image_path, compressed_path: str, the directory containing the raw
        uncompressed images, and the directory that will hold the
        compressed images, respectively
    """
    images, imagenames = ldr.load_images(patch_size, image_path)
    for i_image, image in enumerate(images):
        compressed_filename = os.path.join(
            compressed_path, imagenames[i_image] + ".pkl")
        compressed_image = None
        n_rows, n_cols = image.shape
        n_patch_rows = int(n_rows / patch_size)
        n_patch_cols = int(n_cols / patch_size)
        for i_row in np.arange(n_patch_rows):
            for i_col in np.arange(n_patch_cols):
                patch = image[
                    i_row * patch_size: (i_row + 1) * patch_size,
                    i_col * patch_size: (i_col + 1) * patch_size]
                compressed_patch = autoencoder.forward_pass(
                    patch,
                    evaluating=True,
                    i_stop_layer=3,
                )
                if compressed_image is None:
                    compressed_image = np.zeros((
                        n_patch_rows, n_patch_cols, compressed_patch.size))

                compressed_image[i_row, i_col, :] = compressed_patch
        with open(compressed_filename, "wb") as f:
            pkl.dump(compressed_image, f)


def decompress(autoencoder, compressed_path, decompressed_path):
    """
    Reconstitute the images from their compressed form.

    autoencoder: the neural network trained on the images
    compressed_path, decompressed_path: str, the directory that holds the
        compressed images, and the directory that will hold the
        uncompressed images respectively
    """
    filenames = os.listdir(compressed_path)
    compressed_filenames = [f for f in filenames if f[-4:] == ".pkl"]
    for filename in compressed_filenames:
        with open(os.path.join(compressed_path, filename), "rb") as f:
            compressed_image = pkl.load(f)
            image = None

            n_patch_rows, n_patch_cols, n_vals = compressed_image.shape
            n_rows = n_patch_rows * patch_size
            n_cols = n_patch_cols * patch_size
            for i_row in np.arange(n_patch_rows):
                for i_col in np.arange(n_patch_cols):
                    if image is None:
                        image = np.zeros((n_rows, n_cols), dtype=np.uint8)
                    compressed_vals = compressed_image[i_row, i_col, :]
                    patch = autoencoder.forward_pass(
                        compressed_vals,
                        evaluating=True,
                        i_start_layer=3,
                        i_stop_layer=5,
                    )
                    denormalized_patch = (
                        autoencoder.layers[0].denormalize(patch))
                    denormalized_patch = np.maximum(denormalized_patch, 0)
                    denormalized_patch = np.minimum(denormalized_patch, 1)
                    rescaled_patch = (np.reshape(
                        denormalized_patch, (patch_size, patch_size))
                        * 255).astype(np.uint8)
                    image[
                        i_row * patch_size: (i_row + 1) * patch_size,
                        i_col * patch_size: (i_col + 1) * patch_size
                    ] = rescaled_patch

            decompressed_filename = filename[:-4]
            Image.fromarray(image, mode="L").save(os.path.join(
                decompressed_path, decompressed_filename))


def save_model(autoencoder, model_path):
    """
    Save a copy of the autoencoder model.

    autoencoder: the neural network trained on the images
    model_path: str, the directory that will hold the pickled model
    """
    with open(model_path, "wb") as f:
        pkl.dump(autoencoder, f)


def load_model(model_path):
    """
    Load the autoencoder model from a pickle file.

    autoencoder: the neural network trained on the images
    model_path: str, the directory that will holds the pickled model
    """
    autoencoder = None
    with open(model_path, "rb") as f:
        autoencoder = pkl.load(f)
    return autoencoder


if __name__ == "__main__":
    # Choose the directories to use
    image_path = os.path.join("data", "training")
    model_path = os.path.join("temp", "model.pkl")
    compressed_path = os.path.join("temp", "compressed")
    decompressed_path = os.path.join("temp", "decompressed")

    # Make sure the directories exist
    try:
        os.mkdir("temp")
    except Exception:
        pass
    try:
        os.mkdir(compressed_path)
    except Exception:
        pass
    try:
        os.mkdir(decompressed_path)
    except Exception:
        pass

    # Compress and save the images
    autoencoder = train(image_path)
    compress(autoencoder, image_path, compressed_path)
    save_model(autoencoder, model_path)

    # Decompress the images
    autoencoder = load_model(model_path)
    decompress(autoencoder, compressed_path, decompressed_path)
