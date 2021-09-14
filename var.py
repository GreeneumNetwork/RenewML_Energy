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
    LOGGER = utils.get_logger(
        log_file=f'logs/{date.today()}',
        script_name=os.path.basename(__file__),
    )

    save_str = 'gym'
    order = 10

    stationary = utils.make_datasets(save_str)

    show_fft(stationary,
             save_png=f'FFT_{save_str}')

    for x in range(1, 24):
        var = VARModel(stationary,
                       order=(x, 0),
                       )
        var.fit()
        var.predict(
            start='2017-01-03 00:00:00',
            end='2017-01-04 00:00:00',
            save_png=f'real_v_pred_{save_str}_{order}.png'
        )
        var.save(f'{save_str}_order_{order}.pkl', remove_data=False)
        var.summary()
