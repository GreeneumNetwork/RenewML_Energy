import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date

from utils.data import Data
from utils import utils
from models.ARIMA import ARIMAModel

if __name__ == '__main__':
    LOGGER = utils.get_logger(
        log_file=f'logs/{date.today()}',
        script_name=os.path.basename(__file__),
    )


    save_str = 'gym'
    order = 10

    stationary = utils.make_datasets(save_str)
    arima = ARIMAModel(stationary,
                       order=(order, 0, 0))
    arima.fit()
    arima.summary(plot=False)
    # arima.predict(start='2017-01-03 00:00:00',
    #                   end='2017-01-04 00:00:00',
    #                   save_png=f'real_v_pred_{save_str}_{order}.png'
    #                   )
