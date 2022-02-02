import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date

from utils.data import Data
from utils import utils
from models.VAR import VARModel


def show_fft(datacls: Data, save_png=None):
    utils.config_plot()
    fig, axs = plt.subplots(nrows=len(datacls.raw_data.columns), ncols=2, figsize=(16, 9))
    fig.subplots_adjust(hspace=0)
    datacls.FFT(axs=axs.T[0], raw=True)
    datacls.FFT(axs=axs.T[1])
    axs[0][0].set_title('Raw Data')
    axs[0][1].set_title('Stationarized Data')
    fig.suptitle('FFT: Raw vs Stationary Data')
    for i in range(len(axs)):
        max_ylim = np.max([axs[i][0].get_ylim()[1], axs[i][1].get_ylim()[1]])
        smaller = np.argmax([axs[i][0].get_ylim(), axs[i][1].get_ylim()])
        axs[i][smaller].set_ylim(bottom=None, top=max_ylim)
    if save_png:
        plt.savefig(f'figures/transparent/FFT/{save_png}', transparent=True)
    plt.show()


if __name__ == '__main__':
    # retrieve logging utility to store logs.
    # If you do not wish to store logs, remove log_file
    LOGGER = utils.get_logger(
        log_file=f'logs/{date.today()}',
        script_name=os.path.basename(__file__),
    )

    # Specify which dataset (gym, johonson (for maabarot_johnson, or no_weather)
    # Specify order of model to use in var model
    order = 1
    file = 'data/maabarot_trima_15min.csv'

    dataset = Data.get_data(datafile='data/4Y_Historical.csv',
                            powerfile=file, rescale_power=False)

    # if dataset.df.index.freqstr != '15T':
    #     stationary = dataset.transform(resample=None, lag=['hour', 'day'])
    # else:
    #     stationary = dataset.transform(resample=None, lag=['15Minutes', 'day'])
    stationary = dataset.transform(resample='1H', lag=['hour', 'day'])


    # Declare model - inherits from statsmodels.tsa.VARModel
    var = VARModel(stationary,
                   order=(order, 0),
                   # load='models/saved_models/var_maabarot_trima_order_1.pkl'
                   )
    var.fit()
    var.predict(
        start='2019-10-03 00:00:00',
        end='2019-10-04 00:00:00',
        save_png=f'real_v_pred_{dataset.filename}_{order}.png'
    )
    # Save model results
    var.summary()
    # var.save(f'maabarot_trima_order_{order}.pkl', remove_data=False)
