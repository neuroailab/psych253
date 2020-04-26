
"""
data utils for psych253
"""

import os
import pandas as pd


def get_SRO_data(SRO_datadir='./data/SRO',
                 vars=None, add_demog=True):
    """
    load Eisenberg et al. dataset
    - by default, load the summary variables from meaningful_variables.csv
    and the demographic/health data from demographics_health.csv

    vars: specific variables to return (if None, return all)
    """

    mvars = pd.read_csv(
        os.path.join(SRO_datadir, 'meaningful_variables.csv'), index_col=0)

    if add_demog:
        dvars = pd.read_csv(
            os.path.join(SRO_datadir, 'demographic_health.csv'), index_col=0)
        alldata = mvars.join(dvars)
    else:
        alldata = mvars

    if vars is not None:
        assert isinstance(vars, list)
        alldata = alldata[vars]

    return(alldata)
