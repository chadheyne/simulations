#!/usr/bin/env python


from datetime import datetime
from simulations.run_simulations import *
from simulations.run_predictions import *


def do_run():
    start = datetime.now()
    print("Start time: {}".format(start.strftime("%H:%M:%S")))

    print("Loading company data for simulations")
    company_data = read_file('simulate_1124.csv')

    print("Calculationg regression coefficients")
    regression_results = panel_regressions()

    print("Running simulations and predicting data!")
    predicted_data = create_predictions(regression_results)
    merged_data = merge_data(predicted_data, company_data, full_prices=True)

    print("Inferring number of grants predicted!")
    full_data = infer_grants(merged_data, write=True)

    end = datetime.now()
    print("End time: {}".format(end.strftime("%H:%M:%S")))
    print("Run time: {}".format(end - start))
    return full_data


def create_plot(data, company=10104, columns=(10, 20, 30, 40, 50)):
    dates = pd.date_range(start="01/01/2008",
                          end="12/31/2012",
                          freq="M", name="dates")
    columns = map('P{}'.format, columns)
    ts = pd.DataFrame(index=dates)
    conditions = (data.permno == company) & (data.security == 1)
    for col in columns:
        ts[col] = data.loc[conditions, col+'_1':col+'_12'].stack().values
    axis = ts.plot(figsize=(20, 20),
                   x_compat=True,
                   title="Predictions for Permno {}".format(company))
    axis.set_ylabel("Price prediction")
    axis.set_xlabel("Date")
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.yaxis.tick_left()
    axis.xaxis.tick_bottom()


if __name__ == "__main__":
    data = do_run()
    create_plot(data)
