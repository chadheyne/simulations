import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from .run_simulations import (run_simulations,
                              write_file,
                              read_file)

DIRECTORY = os.path.dirname(__file__)


def panel_regressions(filename='regression_1123.csv'):
    fulldata = pd.read_csv(os.path.join(DIRECTORY,
                                        'input_data',
                                        filename)).sort(columns='fyear')
    idranges = [np.arange(year_count) for year, year_count
                in sorted(fulldata.fyear.value_counts().items())]
    fulldata.index = pd.MultiIndex.from_arrays([fulldata.fyear,
                                                np.concatenate(idranges)])
    paneldata = fulldata.to_panel()

    results = {}
    for year in range(2006, 2013):
        subsample = paneldata.ix[:, 2006:year]
        y, x = subsample.ix['val'], subsample.ix['security':'cfo']
        results[year + 1] = pd.ols(y=y, x=x)
    return results


def create_predictions(results, filename='predict_1123.csv'):
    predict_data = pd.read_csv(os.path.join(DIRECTORY, 'input_data', filename),
                               parse_dates=['grantdate']).sort(columns='fyear')
    predict_data.index = pd.MultiIndex.from_arrays([predict_data.fyear,
                                                    predict_data.permno])
    predict_data['prediction'] = np.nan

    for year in range(2007, 2012):
        coefficients = results[year]
        subsample = predict_data.ix[year, :]
        reg_vars = subsample.ix[:, 'security':'cfo']
        subsample['prediction'].update(coefficients.predict(x=reg_vars))

    prediction = np.exp(predict_data.prediction) - 1
    predict_data.prediction = prediction.clip_lower(0)
    return predict_data.swaplevel('permno', 'fyear')


def merge_data(predict_data, company_data, iterations=50,
               write=False, full_prices=False):

    simulated_data = run_simulations(company_data,
                                     full_prices=full_prices,
                                     iterations=iterations)

    merged_data = predict_data.join(simulated_data,
                                    on=('permno', 'fyear'), rsuffix='_x')

    if write:
        write_file(merged_data)
    return merged_data


def calculate_opt_value(prices, *args):
    t, sigma, r, d, predicted = args
    z = (t * (r - d + sigma ** 2 / 2) / (sigma * np.sqrt(t)))
    arg1 = prices * np.exp(-d * t) * norm.cdf(z)
    arg2 = prices * np.exp(-r * t) * norm.cdf(z - sigma * np.sqrt(t))
    return pd.Series(data=predicted / (arg1 - arg2),
                     index=prices.index,
                     name=prices.name)


def calculate_stock_value(prices, predicted):
    return pd.Series(data=predicted / prices,
                     index=prices.index,
                     name=prices.name)


def infer_grants(merged_data, write=True):
    opt_cols = pd.Index(['P{}_opt_pred'.format(i) for i in range(1, 51)])
    st_cols = pd.Index(['P{}_st_pred'.format(i) for i in range(1, 51)])

    full_data = merged_data.append(pd.DataFrame(columns=opt_cols + st_cols))
    full_data = full_data[merged_data.columns.append((opt_cols, st_cols))]

    criterion_1 = full_data.columns.map(lambda col: col.endswith('month_opt'))
    opt_prices = full_data.loc[:, criterion_1]
    opt_prices.columns = opt_cols

    criterion_2 = full_data.columns.map(lambda col: col.endswith('month_st'))
    st_prices = full_data.loc[:, criterion_2]
    st_prices.columns = st_cols

    t, sigma, r, d, predicted = map(full_data.get,
                                   ('T', 'sigma', 'r', 'd', 'prediction'))

    full_data.loc[:, opt_cols] = (opt_prices.apply(calculate_opt_value,
                                                   args=(t, sigma, r, d, predicted)))

    full_data.loc[:, st_cols] = (st_prices.apply(calculate_stock_value,
                                                 args=(predicted, )))

    if write:
        write_file(full_data)
    return full_data


if __name__ == "__main__":
    start = datetime.now()
    print("Start time: {}".format(start.strftime("%H:%M:%S")))

    company_data = read_file()
    regression_results = panel_regressions()
    predicted_data = create_predictions(regression_results)
    merged_data = merge_data(predicted_data, company_data)
    full_data = infer_grants(merged_data, write=True)

    end = datetime.now()
    print("End time: {}".format(end.strftime("%H:%M:%S")))
    print("Run time: {}".format(end - start))
