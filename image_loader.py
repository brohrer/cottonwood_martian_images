import os
from PIL import Image
import numpy as np
import lodgepole.image_tools as lit

training_path = os.path.join("data", "training")
tuning_path = os.path.join("data", "tuning")
evaluation_path = os.path.join("data", "evaluation")

patch_size = 10
switch_probability = 1 / 200


def load_image(path, imagename):
    img = np.asarray(Image.open(os.path.join(path, imagename))) / 255

    # Convert color images to grayscale
    if len(img.shape) == 3:
        # img = lit.rgb2gray(img)
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


def get_data_sets():
    """
    This function creates three other functions that generate data.
    One generates a training data set,
    one a tuning data set, and the other, an evaluation set.

    The examples are pulled from images taken by the Mars Curiosity Rover.
    https://mars.nasa.gov/msl/multimedia/

    To use in a script:

        import data_loader_martian_images as dat

        (training_generator,
            tuning_generator,
            evaluation_generator) = dat.get_data_sets()
        new_training_example = next(training_generator())
        new_tuning_example = next(tuning_generator())
        new_evaluation_example = next(evaluation_generator())
    """

    def data_generator(path):
        filenames = os.listdir(path)
        imagenames = [f for f in filenames if f[-4:] == ".jpg"]

        assert len(imagenames) > 0

        img = None
        while True:
            # Occasionally switch to a new image
            if img is None or np.random.sample() < switch_probability:
                img = load_image(
                    path,
                    np.random.choice(imagenames))
                n_rows, n_cols = img.shape

            i_row = np.random.randint(n_rows - patch_size)
            i_col = np.random.randint(n_cols - patch_size)
            yield img[i_row: i_row + patch_size, i_col: i_col + patch_size]

    return (
        data_generator(training_path),
        data_generator(tuning_path),
        data_generator(evaluation_path)
    )


if __name__ == "__main__":
    training_set, tuning_set, evaluation_set = get_data_sets()
    for _ in range(1000):
        print(next(training_set))
