import datetime as dt
import os
import numpy as np
import toolbox as tb


class Grid(object):
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

    def condition_generator(self, unexpanded_conditions):
        conditions = tb.grid_expand(unexpanded_conditions)
        for condition in conditions:
            yield condition
