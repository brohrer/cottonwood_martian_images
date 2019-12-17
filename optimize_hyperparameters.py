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
from ponderosa.optimizers import EvoPowell, Random

CONDITIONS = {
    "learning_rate_0": list(np.power(10, np.linspace(-4.5, -1.5, 13))),
    "learning_rate_1": list(np.power(10, np.linspace(-4.5, -1.5, 13))),
    "learning_rate_2": list(np.power(10, np.linspace(-4.5, -1.5, 13))),
    "learning_rate_3": list(np.power(10, np.linspace(-4.5, -1.5, 13))),
}


def main():
    optimizer = EvoPowell()
    # optimizer = Random()
    lowest_error, best_condition, log_filename = (
        optimizer.optimize(evaluate, CONDITIONS))


def evaluate(**condition):
    autoencoder, training_set, tuning_set = initialize(**condition)
    return autoencoder.evaluate_hyperparameters(training_set, tuning_set)


def initialize(
    learning_rate_0=1e-3,
    learning_rate_1=1e-3,
    learning_rate_2=1e-3,
    learning_rate_3=1e-3,
    momentum_amount_0=.83,
    momentum_amount_1=.83,
    momentum_amount_2=.83,
    momentum_amount_3=.83,
    n_nodes_0=25,
    n_nodes_1=47,
    n_nodes_2=33,
    **kwargs,
):
    training_set, tuning_set, evaluation_set = ldr.get_data_sets()

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    n_nodes_dense = [n_nodes_0, n_nodes_1, n_nodes_2]
    n_nodes_dense = [n for n in n_nodes_dense if n > 0]
    n_nodes = n_nodes_dense + [n_pixels]
    layers = []

    learning_rates = [
        learning_rate_0,
        learning_rate_1,
        learning_rate_2,
        learning_rate_3,
    ]
    momentum_amounts = [
        momentum_amount_0,
        momentum_amount_1,
        momentum_amount_2,
        momentum_amount_3,
    ]
    layers.append(RangeNormalization(training_set))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=Tanh(),
            initializer=Glorot(),
            previous_layer=layers[-1],
            optimizer=Momentum(
                learning_rate=learning_rates[i_layer],
                momentum_amount=momentum_amounts[i_layer],
            )
        )
        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
        layers=layers,
        error_function=Sqr,
        n_iter_train=5e5,
        n_iter_evaluate=1e5,
        n_iter_evaluate_hyperparameters=11,
        verbose=True,
    )

    return autoencoder, training_set, tuning_set


if __name__ == "__main__":
    main()
