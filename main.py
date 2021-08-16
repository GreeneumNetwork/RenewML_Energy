import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date

from utils.data import Data
from utils import utils
from models.VAR import VARModel

if __name__ == '__main__':
    LOGGER = utils.get_logger(
        log_file=f'logs/{date.today()}',
        script_name=os.path.basename(__file__),
    )

    # Get and normalize data
    raw_data = Data.get_data(datafile='data/4Y_Historical.csv',
                             powerfile='data/gym_from_2010_04_06_to_2020_12_31.csv',
                             logger=LOGGER)

    raw_data.df = raw_data.df.drop(columns=['diffuse_rad:W', 'direct_rad:W'])
    stationary = raw_data.transform(lag=['hour', 'day'],
                                    resample=False,
                                    scaler=None)

    # stationary.ts_plot(lags=24 * 7)
    # fig, axs = plt.subplots(nrows=len(raw_data.df.columns), ncols=2)
    # fig.subplots_adjust(hspace=0)
    # raw_data.FFT(axs=axs.T[0])
    # stationary.FFT(axs=axs.T[1])
    # for x in range(len(axs)):
    #     max_ylim = np.max([axs[x][0].get_ylim()[1], axs[x][1].get_ylim()[1]])
    #     smaller = np.argmax([axs[x][0].get_ylim(), axs[x][1].get_ylim()])
    #     axs[x][smaller].set_ylim(bottom=None, top=max_ylim)
    # plt.show()

    var = VARModel(stationary, order=(2, 0))
    var.train(train_percent=0.7)
    var.summary()
    # var.forecast()
