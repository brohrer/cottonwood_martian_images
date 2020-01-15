import numpy as np

from cottonwood.core.activation import Tanh, Logistic, ReLU
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
from cottonwood.core.initializers import Glorot, He
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum
import dev_image_loader as ldr
from ponderosa.optimizers import EvoPowell

CONDITIONS = {
    # "activation_function": [Tanh, Logistic, ReLU],
    "initializer": [Glorot, He],
    "learning_rate": list(np.power(10, np.linspace(-5, -3, 9))),
}


def main():
    optimizer = EvoPowell()
    lowest_error, best_condition, log_filename = (
        optimizer.optimize(evaluate, CONDITIONS))


def evaluate(**condition):
    autoencoder, training_set, tuning_set = initialize(**condition)
    nn_error, nn_error_worst = autoencoder.evaluate_hyperparameters(
        training_set, tuning_set)
    print("nn error", nn_error, "nn_error_worst", nn_error_worst)
    return nn_error


def initialize(
    activation_function=Tanh,
    initializer=Glorot,
    learning_rate=1e-4,
    n_nodes_00=79,
    n_nodes_0=23,
    n_nodes_1=9,
    n_nodes_2=23,
    n_nodes_3=79,
    patch_size=11,
    **kwargs,
):
    training_set, tuning_set, evaluation_set = ldr.get_data_sets(
        patch_size=patch_size)

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    # n_nodes_dense = [n_nodes_00, n_nodes_0, n_nodes_1, n_nodes_2, n_nodes_3]
    n_nodes_dense = [n_nodes_00, n_nodes_1, n_nodes_3]
    # n_nodes_dense = [n_nodes_1]
    n_nodes_dense = [n for n in n_nodes_dense if n > 0]
    n_nodes = n_nodes_dense + [n_pixels]

    layers = []

    layers.append(RangeNormalization(training_set))

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
        n_iter_train=5e4,
        n_iter_evaluate=1e4,
        n_iter_evaluate_hyperparameters=9,
        verbose=False,
    )

    return autoencoder, training_set, tuning_set


if __name__ == "__main__":
    main()
