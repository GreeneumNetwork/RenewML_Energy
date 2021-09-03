import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date

from utils.data import Data
from utils import utils
from models.VAR import VARModel


def show_fft(stationary: pd.DataFrame, raw_data: pd.DataFrame):
    fig, axs = plt.subplots(nrows=len(raw_data.df.columns), ncols=2)
    fig.subplots_adjust(hspace=0)
    raw_data.FFT(axs=axs.T[0])
    stationary.FFT(axs=axs.T[1])
    axs[0][0].set_title('Raw Data')
    axs[0][1].set_title('Stationarized Data')
    fig.suptitle('FFT: Raw vs Stationary Data')
    for x in range(len(axs)):
        max_ylim = np.max([axs[x][0].get_ylim()[1], axs[x][1].get_ylim()[1]])
        smaller = np.argmax([axs[x][0].get_ylim(), axs[x][1].get_ylim()])
        axs[x][smaller].set_ylim(bottom=None, top=max_ylim)
    plt.show()


if __name__ == '__main__':
    LOGGER = utils.get_logger(
        log_file=f'logs/{date.today()}',
        script_name=os.path.basename(__file__),
    )

    save_str = 'gym'
    order = 10

    stationary = utils.make_datasets(save_str)

    var = VARModel(stationary,
                   order=(order, 0),
                   load=f'models/saved_models/var_{save_str}_{order}.pkl'
                   )

    var.fit()
    var.predict(
        start='2017-01-03 00:00:00',
        end='2017-01-04 00:00:00',
        # save_png=f'real_v_pred_{save_str}_{order}.png'
    )
    # var.save(f'{save_str}_order_{order}.pkl', remove_data=False)
    # var.summary()
