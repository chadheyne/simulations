import os
import numpy as np
from numpy import exp, sqrt
from numpy.random import standard_normal as random
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import DateOffset
"""
    You can speed the process up a small amount by removing all of the print statements.
    I just put them in so that you could see in the log if it's running.
"""
DIRECTORY = os.path.dirname(__file__)


def simulate(observation, iteration, full_price=False):
    '''
        Arguments:
            observation - one row of data, either in Pandas Series or Python dictionary format
            negative - either -1 or 1 which is multiplied by the previous period's return based on
                       the parity of iteration number
            limit - what the maximum return should be. 5=500%
            iterations - how many random prices to generate for the current iteration

        Outputs:
            returns the average of the randomly generated prices

        What I should change:
            It's likely possible to compute the entire thing through Pandas without the ugly hacks
            by just drawing iterations number of random numbers at a time from Numpy, creating the
            predicted prices all at once and if any meet the rejection criterion, redraw.
            However, it seems possible that we could get stuck in a loop here because we would only
            need 1 bad observation per draw.
    '''

    outputs = pd.Series(index=['P{}_{}'.format(iteration, i) for i in range(1, 13)])
    initial, ret, vol = observation['S0'], observation['u'], observation['sigma']
    start_date, end_date = observation['date'] + DateOffset(months=1), observation['date'] + DateOffset(months=12)

    for index, month in enumerate(pd.date_range(start=start_date, end=end_date, freq='M'), start=1):
        initial = initial * exp(((ret - vol ** 2 / 2) * (1 / 12) + vol * random() * sqrt(1 / 12)))
        outputs['P{}_{}'.format(iteration, index)] = initial

    observation['P{}_month_st'.format(iteration)] = outputs[observation['grantdate_st'].month - 1]
    observation['P{}_month_opt'.format(iteration)] = outputs[observation['grantdate_opt'].month - 1]
    observation['P{}_year'.format(iteration)] = outputs['P{}_12'.format(iteration)]
    return observation if not full_price else observation.combine_first(outputs)


def in_range(observation):
    '''
        Change the return value to 1 in the first case if you want the oor variable
        to be 1 if the prediction is 'good'
    '''

    if observation['min'] <= observation['S1'] <= observation['max']:
        return 0
    return 1


def run_simulations(company_data, iterations=50, full_prices=False):
    '''
        All of the calls to have axis=1 in order to work with row data. Otherwise it will run simulations over columns.
    '''
    np.random.seed(15)
    predictions = pd.Series(('P{0}_{1}'.format(i, j)
                             for i in range(1, iterations+1) for j in ('month_st', 'month_opt', 'year')))
    if full_prices:
        predictions2 = pd.Series(('P{0}_{1}'.format(i, j)
                                 for i in range(1, iterations+1) for j in range(1, 13)))
    else:
        predictions2 = []

    simulated_data = pd.DataFrame(data=company_data.copy(),
                                  index=company_data.index,
                                  columns=company_data.columns.append([predictions, predictions2]))

    for iteration in range(1, iterations+1):
        print("Running simulation {} of {}:\n  Started at {}".format(iteration,
              iterations, datetime.now().strftime("%H:%M:%S:%f")), end="")

        simulated_data.update(simulated_data.apply(simulate, axis=1, args=(iteration, full_prices)))
        print("  Ended at {}".format(datetime.now().strftime("%H:%M:%S:%f")))

    simulated_data.date = pd.to_datetime(simulated_data.date)
    simulated_data.date = simulated_data.date.apply(lambda date: date.strftime("%d/%m/%Y"))
    simulated_data.grantdate_opt = simulated_data.grantdate_opt.apply(lambda date: date.strftime("%m/%d/%Y"))
    simulated_data.grantdate_st = simulated_data.grantdate_st.apply(lambda date: date.strftime("%m/%d/%Y"))

    return simulated_data


def read_file(filename="simulate_1124.csv"):
    company_data = pd.read_csv(os.path.join(DIRECTORY, 'input_data', filename),
                               index_col=['permno', 'fyear'],
                               parse_dates=['date', 'grantdate_opt', 'grantdate_st'],
                               dtype={'permno': 'str',
                                      'fyear': 'str',
                                      'adjfac': 'float',
                                      'S0': 'float',
                                      'u': 'float',
                                      'sigma': 'float',
                                      'S1': 'float'})
    return company_data


def write_file(simulated_data, filename='outputs_{}.csv'):
    filename = filename.format(datetime.today().strftime("%m%d"))
    simulated_data.to_csv(os.path.join(DIRECTORY, 'output_data',
                          filename))
    print_dict = {'file': filename,
                  'cols': len(simulated_data.columns),
                  'obs': len(simulated_data)}
    print("Finished writing {file} with {cols} columns and {obs} observations".format(**print_dict))


def main():
    start = datetime.now()
    print("Start time: {}".format(start.strftime("%H:%M:%S")))

    company_data = read_file()
    output_data = run_simulations(company_data, iterations=50)
    write_file(output_data)

    end = datetime.now()
    print("End time: {}".format(end.strftime("%H:%M:%S")))
    print("Run time: {}".format(end - start))

if __name__ == "__main__":
    main()
