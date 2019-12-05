import numpy as np
from cottonwood.core.activation import Tanh
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
from cottonwood.core.initializers import Glorot
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum
from cottonwood.core.regularization import Limit
from cottonwood.examples.autoencoder.autoencoder_viz import Printer
import image_loader as ldr


def run():
    training_set, tuning_set, evaluation_set = ldr.get_data_sets()

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    printer = Printer(input_shape=sample.shape)

    N_NODES = [64, 36, 24, 36, 64]
    # N_NODES = [64]
    n_nodes = N_NODES + [n_pixels]
    layers = []

    layers.append(RangeNormalization(training_set))

    for i_layer in range(len(n_nodes)):
        new_layer = Dense(
            n_nodes[i_layer],
            activation_function=Tanh(),
            initializer=Glorot(),
            previous_layer=layers[-1],
            optimizer=Momentum(),
        )
        # new_layer.add_regularizer(L1())
        new_layer.add_regularizer(Limit(4.0))
        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
        layers=layers,
        error_function=Sqr,
        printer=printer,
    )

    msg = """

    Running autoencoder on images of the surface of Mars.
        Find performance history plots, model parameter report,
        and neural network visualizations in the directory
        {}

""".format(autoencoder.reports_path)

    print(msg)

    autoencoder.train(training_set)
    autoencoder.evaluate(tuning_set)
    autoencoder.evaluate(evaluation_set)


run()
