import numpy as np

from cottonwood.core.activation import Tanh
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
from cottonwood.core.initializers import Glorot
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum
from cottonwood.core.regularization import Limit, L1, L2
from cottonwood.examples.autoencoder.autoencoder_viz import Printer
import image_loader as ldr
from ponderosa.optimizers import EvoPowell

# There are 2016 unique combinations of parameter values.
CONDITIONS = {
    "limit": list(np.power(2.0, np.arange(-3, 4))),
    "L1_param": list(np.power(10.0, np.arange(-6, 0))),
    "L2_param": list(np.power(10.0, np.arange(-6, 0))),
    "learning_rate": list(np.power(10.0, np.arange(-8, 0))),
}


def main():
    optimizer = EvoPowell()
    lowest_error, best_condition, log_filename = (
        optimizer.optimize(evaluate, CONDITIONS))


def evaluate(**condition):
    autoencoder, training_set, tuning_set = initialize(**condition)
    return autoencoder.evaluate_hyperparameters(training_set, tuning_set)


def initialize(
    limit=None,
    L1_param=None,
    L2_param=None,
    learning_rate=None,
    momentum=.9,
    **kwargs,
):
    training_set, tuning_set, evaluation_set = ldr.get_data_sets()

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    N_NODES = [33]
    n_nodes = N_NODES + [n_pixels]
    layers = []

    layers.append(RangeNormalization(training_set))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=Tanh,
            initializer=Glorot(),
            previous_layer=layers[-1],
            optimizer=Momentum(
                learning_rate=learning_rate,
                momentum_amount=momentum,
            ),
        )
        if limit is not None:
            new_layer.add_regularizer(Limit(limit))
        if L1_param is not None:
            new_layer.add_regularizer(L1(L1_param))
        if L2_param is not None:
            new_layer.add_regularizer(L2(L2_param))

        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
        layers=layers,
        error_function=Sqr,
        n_iter_train=5e4,
        n_iter_evaluate=1e4,
        verbose=False,
    )

    return autoencoder, training_set, tuning_set


if __name__ == "__main__":
    main()
