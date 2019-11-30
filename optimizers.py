import copy
import datetime as dt
import os
import numpy as np
import toolbox as tb


class HPOptimizer(object):
    def __init__(
        self,
        report_dir=os.path.join(
            "reports",
            "hpo_" + dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ),
        report_filename="hpo_results.csv",
        report_plot_filename="hpo_results.png",
    ):
        self.best_error = 1e10
        self.best_condition = None

        self.report_dir = report_dir
        self.report_filename = os.path.join(
            self.report_dir, report_filename)
        self.report_plot_filename = os.path.join(
            self.report_dir, report_plot_filename)

        # Ensure that report directory exists
        try:
            os.mkdir("reports")
        except Exception:
            pass
        try:
            os.mkdir(self.report_dir)
        except Exception:
            pass

    def optimize(self, evaluate, conditions, verbose=True):
        if verbose:
            print(
                "\nThis is going to take a while.\n"
                + "    You can check on the best-so-far solution at any time\n"
                + "    in " + self.report_plot_filename + "\n"
                + "    The full results log is maintained\n"
                + "    in " + self.report_filename + "\n\n"
            )

        condition_history = []
        for condition in self.condition_generator(conditions):
            if verbose:
                print("    Evaluating condition", condition)
            error = evaluate(**condition)
            condition["error"] = error
            condition_history.append(condition)
            tb.results_dict_list_to_csv(
                condition_history, self.report_filename)

            if error < self.best_error:
                self.best_error = error
                self.best_condition = condition
            if verbose:
                results_so_far = tb.results_csv_to_dict_list(
                    self.report_filename)
                tb.progress_report(results_so_far, self.report_plot_filename)
        results_so_far = tb.results_csv_to_dict_list(self.report_filename)
        tb.progress_report(results_so_far, self.report_plot_filename)
        return self.best_error, self.best_condition, self.report_filename

    def condition_generator(self, conditions):
        pass


class Random(HPOptimizer):
    def __init__(self, n_iter=1e10):
        super().__init__()
        self.n_iter = n_iter

    def condition_generator(self, unexpanded_conditions):
        conditions = tb.grid_expand(unexpanded_conditions)
        np.random.shuffle(conditions)
        if self.n_iter < len(conditions):
            conditions = conditions[:self.n_iter]

        for condition in conditions:
            yield condition

# The Random search using the default n_iter will eventually
# cover the entire grid.
Grid = Random


class EvoPowell(HPOptimizer):
    """
    An evolutionary algorithm inspired by Powell's method.
    """
    def __init__(self, n_iter=1e10):
        super().__init__()
        self.n_iter = int(n_iter)

    def condition_generator(self, conditions):
        condition_names = list(conditions.keys())
        np.random.shuffle(condition_names)

        # When enumerating the values of a parameter,
        # approximately what fraction to test.
        expansion_fraction = .3

        evaluated = []
        scored = []
        to_evaluate = []

        # Before starting in, how many random points to check.
        # Having a few of these helps prevent getting stuck in a
        # "bad luck" initial condition.
        n_initial_random_conditions = 2 * len(condition_names)
        for _ in range(n_initial_random_conditions):
            to_evaluate.append(tb.get_random_condition(conditions))

        # How many conditions to try expanding on (and fail)
        # before declaring the parameter space sufficiently explored.
        n_points_to_try = 3

        def get_next_conditions(to_evaluate):
            """
            Once the `to_evaluate` queue is empty, this method
            repopulates it.
            """
            conditions_to_expand = choose_conditions_to_expand(n_points_to_try)
            for to_expand in conditions_to_expand:
                condition_names.insert(0, condition_names.pop())
                for condition_name in condition_names:
                    to_evaluate += expand_line(to_expand, condition_name)
                    if len(to_evaluate) > 0:
                        return
            # If it gets this far it means that there are no lines radiating
            # from this point that haven't yet been explored.
            # The algorithm is done.
            raise StopIteration

        def choose_conditions_to_expand(n_conditions):
            errors = np.array([cond["error"] for cond in scored])
            to_expand = []

            # Assign a weight to each condition, based on the error (loss)
            # associated with it. We want to choose a low error condition
            # to expand, but it doesn't have to be the lowest. We'll
            # randomly choose one, giving strong preference to conditions
            # with lower errors.
            # The lowest error will have a weight of 1. The highest error
            # will have a weight of 0. An error halfway in between will
            # have a weight of .5**2 = .25
            weights = (
                (np.max(errors) - errors) /
                (np.max(errors) - np.min(errors))
            ) ** 2
            order = np.argsort(weights)
            ordered_weights = weights[order]

            for _ in range(n_conditions):
                i_ordered_weight = np.where(
                    ordered_weights > np.random.uniform())[0][0]
                chosen_weight = ordered_weights[i_ordered_weight]

                i_cond = np.where(weights == chosen_weight)[0]
                if i_cond.size > 1:
                    i_cond = np.random.choice(i_cond)
                to_expand.append(copy.deepcopy(scored[int(i_cond)]))

            for cond in to_expand:
                try:
                    del cond["error"]
                except KeyError:
                    pass
            return to_expand

        def expand_line(cond, line):
            """
            For a given parameter, create a set of conditions for each
            value. This expands the starting condition `cond` along
            a single parameter `line`.
            """
            new_to_evaluate = []
            vals = conditions[line]
            for val in vals:
                new_cond = copy.deepcopy(cond)
                new_cond[line] = val
                if new_cond not in evaluated:
                    new_to_evaluate.append(new_cond)

                # Randomly select just a few of the candidate solutions.
                np.random.shuffle(new_to_evaluate)
                # Force the shuffline to take place before the slicing.
                # to_evaluate = list(to_evaluate)
                n_conditions_max = int(np.ceil(
                    len(list(vals)) * expansion_fraction))
                if len(new_to_evaluate) > n_conditions_max:
                    new_to_evaluate = new_to_evaluate[:n_conditions_max]
            return new_to_evaluate

        for _ in range(self.n_iter):
            if len(to_evaluate) == 0:
                get_next_conditions(to_evaluate)

            condition = to_evaluate.pop()
            # Keeping a copy of condition means that it remains unmodified.
            # It's useful for checking whether a condition has been tested
            # already.
            evaluated.append(copy.deepcopy(condition))
            # Keeping the original object is helpful too. We know that
            # it will have the evaluation error appended to it.
            # We can use it for determining which point to expand.
            scored.append(condition)
            yield condition
