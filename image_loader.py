import os
from PIL import Image
import numpy as np
import lodgepole.image_tools as lit


def load_image(path, imagename, patch_size):
    img = np.asarray(Image.open(os.path.join(path, imagename))) / 255

    # Convert color images to grayscale
    if len(img.shape) == 3:
        img = lit.rgb2gray_approx(img)

    n_rows, n_cols = img.shape
    assert len(img.shape) == 2
    assert n_rows > patch_size
    assert n_cols > patch_size

    # Pad out to a multiple of patch_size
    n_rows_pad = int(np.ceil(n_rows / patch_size)) * patch_size
    n_cols_pad = int(np.ceil(n_cols / patch_size)) * patch_size

    padded = np.pad(img, ((0, n_rows_pad - n_rows), (0, n_cols_pad - n_cols)))

    assert np.sum(np.isnan(padded)) == 0

    return padded


def load_images(patch_size, image_path):
    images = []
    filenames = os.listdir(image_path)
    imagenames = []

    for filename in filenames:
        try:
            image = load_image(image_path, filename, patch_size)
            images.append(image)
            imagenames.append(filename)
        except Exception:
            pass

    assert len(images) > 0

    return images, imagenames


def data_generator(imagelist, patch_size):
    img = None
    switch_probability = 1 / 100
    while True:
        # Occasionally switch to a new image
        if img is None or np.random.sample() < switch_probability:
            img = np.random.choice(imagelist)
            n_rows, n_cols = img.shape

        i_row = np.random.randint(n_rows - patch_size)
        i_col = np.random.randint(n_cols - patch_size)
        yield img[i_row: i_row + patch_size, i_col: i_col + patch_size]


def get_training_data(patch_size, image_path):
    """
    This function creates a function that generates training data.
    """
    training_images, _ = load_images(patch_size, image_path)
    return data_generator(training_images, patch_size)
