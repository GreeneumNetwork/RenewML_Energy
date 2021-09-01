import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import greenium.utils.data


class ARIMAModel(ARIMA):

    def __init__(self,
                 dataclass: greenium.utils.data.Data,
                 train_percent: float = 0.7,
                 load: str = None
                 ):
        self.dataclass = dataclass
        self.scaler = dataclass.scaler or None
        self.dataset = dataclass.df
        self.logger = dataclass.logger
        self.load = load
        self.train_set, self.test_set = self.split(train_percent)
        # model parameters

        super(ARIMAModel, self).__init__(self.train_set)

    def split(self, train_percent):
        stop_idx = np.floor(train_percent * len(self.dataset)).astype(int)
        train_set = self.dataset.iloc[:stop_idx]
        test_set = self.dataset.iloc[stop_idx:]
        # self.train_set = self.train_set.set_index('validdate').asfreq('D')
        # self.test_set = self.test_set.set_index('validdate').asfreq('D')

        self.logger.info(f'Test and train set successfully created.')

        return train_set, test_set

    def fit(self, **kwargs) -> object:
        """
        Inherits from ARIMAResults.fit()

        :param kwargs:
        :return:
        """
        if self.load:
            self.model_result = ARIMAResults.load(self.load)
            self.logger.info(f'model loaded from {self.load}')
            return self.model_result

        self.model_result = super(ARIMAModel, self).fit()
        self.logger.info('Trained and fit')

        return self.model_result
