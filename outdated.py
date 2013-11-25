    '''
        The syntax for the stuff below is kind of tricky. Basically what it's doing is
        company_data[mean] = company_data[P1:P_iterations].mean()
    '''

    for stat in ('count', 'mean', 'std', 'median', 'min', 'max'):
        results[stat] = getattr(results.loc[:, 'P1':'P{}'.format(iterations)], stat)(axis=1)

    results['range'] = results['max'] - results['min']

    '''
        I couldn't remember which direction your out of range variable was so now it's 0 if it's between min/max else 1
    '''

    results['oor'] = results.apply(in_range, axis=1)

    '''
        This section here can be removed if you want. I was only doing it because all of the decimal places in the output
        was starting to bother me. I rounded all price variables to 2 digits and the summary statistics/input std, etc to 5
    '''

    for column in results.loc[:, 'S0':'range']:
        if column.startswith(('P', 'S')):
            results[column] = results[column].round(decimals=2).astype('str')
        else:
            results[column] = results[column].round(decimals=5).astype('str')

    return results