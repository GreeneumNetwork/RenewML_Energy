import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.data import Data
from utils import utils
from models.VAR import VARModel

if __name__ == '__main__':
    LOGGER = utils.get_logger(
        script_name=os.path.basename(__file__),
    )

    # Get and normalize data
    raw_data = Data.get_data(datafile='data/4Y_Historical.csv',
                             powerfile='data/maabarot_johnson_from_2010_04_22_to_2020_12_31.csv',
                             logger=LOGGER)

    raw_data.df = raw_data.df.drop(columns=['global_rad:W', 'direct_rad:W'])
    stationary = raw_data.transform(lag=['hour', 'day'],
                                    resample=False,
                                    scaler=None)

    # stationary.ts_plot(lags=24 * 7)
    # stationary.ADF()

    var = VARModel(stationary)
    var.train(train_percent=0.7)
    var.summary()
    # var.forecast()
