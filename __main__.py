#!/usr/bin/env python


from datetime import datetime
from simulations import run_simulations as run
from simulations.run_predictions import *


def do_run():
    start = datetime.now()
    print("Start time: {}".format(start.strftime("%H:%M:%S")))

    print("Loading company data for simulations")
    company_data = run.read_file('simulate_1124.csv')

    print("Calculationg regression coefficients")
    regression_results = panel_regressions()

    print("Running simulations and predicting data!")
    predicted_data = create_predictions(regression_results)
    merged_data = merge_data(predicted_data, company_data)

    print("Inferring number of grants predicted!")
    infer_grants(merged_data, write=True)

    end = datetime.now()
    print("End time: {}".format(end.strftime("%H:%M:%S")))
    print("Run time: {}".format(end - start))


if __name__ == "__main__":
    do_run()
