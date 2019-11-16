import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from cottonwood.core.activation import Tanh
from cottonwood.core.model import ANN
from cottonwood.core.error_function import Sqr
from cottonwood.core.initializers import Glorot
from cottonwood.core.layers.dense import Dense
from cottonwood.core.layers.range_normalization import RangeNormalization
from cottonwood.core.layers.difference import Difference
from cottonwood.core.optimizers import Momentum, SGD
from cottonwood.core.regularization import Limit
from cottonwood.examples.autoencoder.autoencoder_viz import Printer
import image_loader as ldr


N_ITER_TRAIN = int(5e4)
N_ITER_EVALUATE = int(1e4)
N_ITER_RUNS = 100
REPORTING_BIN_SIZE = int(5e3)


def initialize():
    training_set, tuning_set, evaluation_set = ldr.get_data_sets()

    sample = next(training_set)
    n_pixels = np.prod(sample.shape)
    printer = Printer(input_shape=sample.shape)

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
            # optimizer=Momentum(),
            optimizer=SGD(),
        )
        # new_layer.add_regularizer(L1())
        new_layer.add_regularizer(Limit(4.0))
        layers.append(new_layer)

    layers.append(Difference(layers[-1], layers[0]))

    autoencoder = ANN(
            layers=layers,
        error_function=Sqr,
        n_iter_train=N_ITER_TRAIN,
        n_iter_evaluate=N_ITER_EVALUATE,
        printer=printer,
        verbose=False,
    )

    msg = """

    Running autoencoder on images of the surface of Mars.
        Find performance history plots, model parameter report,
        and neural network visualizations in the directory
        {}

""".format(autoencoder.reports_path)

    print(msg)

    return autoencoder, training_set, tuning_set


def run():
    error_histories = []
    for _ in range(N_ITER_RUNS):
        autoencoder, training_set, tuning_set = initialize()
        autoencoder.train(training_set)
        error_history = autoencoder.evaluate(tuning_set)
        error_histories.append(error_history)

    return error_histories


def visualize(error_histories):
    ymin = -3
    ymax = 0

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlabel(f"x{REPORTING_BIN_SIZE} iterations")
    ax.set_ylabel("log error")

    for error_history in error_histories:
        n_bins = int(len(error_history) // REPORTING_BIN_SIZE)
        smoothed_history = []
        for i_bin in range(n_bins):
            smoothed_history.append(np.mean(error_history[
                i_bin * REPORTING_BIN_SIZE:
                (i_bin + 1) * REPORTING_BIN_SIZE
            ]))
        error_history = np.log10(np.array(smoothed_history) + 1e-10)

        # blurry_plot(ax, error_history)
        ax.plot(error_history, color="midnightblue", linewidth=.3, alpha=.3)

        ymin = np.minimum(ymin, np.min(error_history))
        ymax = np.maximum(ymax, np.max(error_history))

    ax.set_ylim(ymin, ymax)
    ax.grid()
    fig.savefig(os.path.join("reports", "repeated_runs.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    error_histories = run()

    with open(os.path.join("reports", "error_histories.pkl"), 'wb') as pklfile:
        pkl.dump(error_histories, pklfile)
    with open(os.path.join("reports", "error_histories.pkl"), 'rb') as pklfile:
        error_histories_reloaded = pkl.load(pklfile)

    visualize(error_histories_reloaded)
