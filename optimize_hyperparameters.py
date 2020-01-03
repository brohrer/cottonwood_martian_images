import numpy as np

from cottonwood.core.activation import Tanh
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
from cottonwood.core.initializers import Glorot
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum
import image_loader as ldr
from ponderosa.optimizers_parallel import EvoPowell
# from ponderosa.optimizers import EvoPowell

CONDITIONS_0 = {
    "n_nodes_0": [23, 49, 77, 113, 147],
    "n_nodes_1": [9, 10, 11, 13, 17, 23, 29, 37],
    "n_nodes_2": [23, 49, 77, 113, 147],
    "patch_size": [7, 9, 11, 13, 17, 21],
}

CONDITIONS_1 = {
    "n_nodes_1": [ 8, 9, 10, 11, 12, 13, 14, 15, 17],
    "patch_size": [7, 8, 9, 10, 11, 12, 13, 15],
}

CONDITIONS_2 = {
    "n_nodes_00": [23, 49, 77, 113, 147],
    "n_nodes_0": [23, 49, 77, 113, 147],
    "n_nodes_1": [ 8, 9, 10, 11, 12],
    "n_nodes_2": [23, 49, 77, 113, 147],
    "n_nodes_3": [23, 49, 77, 113, 147],
    "patch_size": [7, 9, 11, 13, 17, 21],
}

CONDITIONS = CONDITIONS_2

def main():
    optimizer = EvoPowell()
    lowest_error, best_condition, log_filename = (
        optimizer.optimize(evaluate, CONDITIONS))


def evaluate(**condition):
    autoencoder, training_set, tuning_set = initialize(**condition)
    nn_error = autoencoder.evaluate_hyperparameters(training_set, tuning_set)

    compression_ratio = condition["n_nodes_1"] / condition["patch_size"] ** 2
    # Severely penalize autoencoders that have too much error.
    error_threshold = -3
    if nn_error > error_threshold:
        compression_ratio += 1
    print("nn error", nn_error, "compression", compression_ratio)

    return compression_ratio


def initialize(
    learning_rate=1e-3,
    n_nodes_00=47,
    n_nodes_0=47,
    n_nodes_1=47,
    n_nodes_2=47,
    n_nodes_3=47,
    patch_size=10,
    **kwargs,
):
    training_set, tuning_set, evaluation_set = ldr.get_data_sets(
        patch_size=patch_size)

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    n_nodes_dense = [n_nodes_00, n_nodes_0, n_nodes_1, n_nodes_2, n_nodes_3]
    # n_nodes_dense = [n_nodes_0, n_nodes_1, n_nodes_2]
    n_nodes_dense = [n for n in n_nodes_dense if n > 0]
    # n_nodes_dense = [n_nodes_1]
    n_nodes = n_nodes_dense + [n_pixels]
    layers = []

    layers.append(RangeNormalization(training_set))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=Tanh(),
            initializer=Glorot(),
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
        n_iter_train=5e4,
        n_iter_evaluate=1e4,
        n_iter_evaluate_hyperparameters=7,
        verbose=False,
    )

    return autoencoder, training_set, tuning_set


if __name__ == "__main__":
    main()
