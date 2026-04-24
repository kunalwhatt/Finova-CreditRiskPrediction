import numpy as np
import pandas as pd

def calculate_woe_iv(df, feature, target):
    g = df.groupby(feature)[target].agg(['count','sum'])
    g.columns = ['total','bad']
    g['good'] = g['total'] - g['bad']
    g['bad_dist'] = g['bad'] / (g['bad'].sum() + 1e-9)
    g['good_dist'] = g['good'] / (g['good'].sum() + 1e-9)
    g['WOE'] = np.log((g['good_dist'] + 1e-9) / (g['bad_dist'] + 1e-9))
    g['IV'] = (g['good_dist'] - g['bad_dist']) * g['WOE']
    return g, g['IV'].sum()
