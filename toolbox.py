import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def grid_expand(conditions):
    expanded = [{}]
    for key, value_list in conditions.items():
        new_expanded = []
        for args_dict in expanded:
            for value in value_list:
                new_args_dict = copy.copy(args_dict)
                new_args_dict[key] = value
                new_expanded.append(new_args_dict)
        expanded = new_expanded
    return expanded


def get_random_condition(conditions):
    rand_condition = {}
    for key, value in conditions.items():
        rand_condition[key] = np.random.choice(conditions[key])
    return rand_condition


def results_dict_list_to_csv(results_dict, csv_filename):
    """
    Lifted from Matthew Flaschen on StackOverflow
    https://stackoverflow.com/questions/3086973/how-do-i-convert-this-list-of-dictionaries-to-a-csv-file
    """
    keys = results_dict[0].keys()
    with open(csv_filename, 'wt') as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_dict)


def results_csv_to_dict_list(csv_filename):
    with open(csv_filename, 'rt') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        return list(dict_reader)


def progress_report(results, filename):
    """
    Show how the best-so-far error value has evolved.
    """
    best_so_far_error = 1e10
    best_so_far_errors = []
    best_so_far_params = None
    for result in results:
        error = float(result["error"])
        if not np.isnan(error):
            if error < best_so_far_error:
                best_so_far_error = error
                best_so_far_params = result
            best_so_far_errors.append(best_so_far_error)

    fig = plt.Figure()
    ax = fig.gca()
    ax.plot(np.arange(len(best_so_far_errors)) + 1, best_so_far_errors)
    ax.set_xlabel("Parameter combinations tested")
    ax.set_ylabel("Best error so far")
    param_msg = ""
    for key, value in best_so_far_params.items():
        param_msg += f"{key}: {value}\n"
    ax.text(
        len(best_so_far_errors) - 1,
        np.max(best_so_far_errors),
        param_msg,
        horizontalalignment="right",
        verticalalignment="top",
    )
    fig.savefig(filename, dpi=300)

    return best_so_far_params
