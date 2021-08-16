import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import greenium.utils.data


class VARModel():

    def __init__(self,
                 dataclass: greenium.utils.data.Data,
                 order: tuple = (1, 0)):

        self.dataclass = dataclass
        self.scaler = dataclass.scaler or None
        self.dataset = dataclass.df
        self.logger = dataclass.logger

        # model parameters
        self.order = order

    def split(self, train_percent):

        stop_idx = np.floor(train_percent * len(self.dataset)).astype(int)
        self.train_set = self.dataset.iloc[:stop_idx]
        self.test_set = self.dataset.iloc[stop_idx:]
        # self.train_set = self.train_set.set_index('validdate').asfreq('D')
        # self.test_set = self.test_set.set_index('validdate').asfreq('D')

        self.logger.info(f'Test and train set successfully created.')

        return self.train_set, self.test_set

    def train(self, train_percent: float):

        self.split(train_percent)

        model = VARMAX(self.train_set, order=self.order, trend='c')
        self.model_result = model.fit(maxiter=1000, disp=False)
        self.logger.info('Trained and fit')
        print(self.model_result.summary())

        return self.model_result

    def summary(self):
        num_hours = 24

        print(self.model_result.summary())
        self.model_result.plot_diagnostics()
        plt.show()

        pred = self.model_result.forecast(num_hours)
        real = self.test_set.iloc[:num_hours]
        pred = pred.set_index(real.index)
        pred = self.dataclass.inverse_transform(pred)
        real = self.dataclass.inverse_transform(real)

        fig, axs = plt.subplots(len(real.columns), 1)
        myFmt = DateFormatter("%H:%M")
        for i, col in enumerate(real.columns):
            rmse = mean_squared_error(real[col].iloc[:num_hours], pred[col].iloc[:num_hours], squared=False)
            mae = mean_absolute_error(real[col].iloc[:num_hours], pred[col].iloc[:num_hours])
            self.logger.info(f'RMSE {col}: {rmse}')
            axs[i].plot(real.index[:num_hours], real[col].iloc[:num_hours], label='Real' if i==0 else '_nolegend_', c='b')
            axs[i].plot(real.index[:num_hours], pred[col].iloc[:num_hours], label='Predicted' if i==0 else '_nolegend_', c='r')
            axs[i].set_title(col)
            axs[i].text(0, 0.95, f'RMSE: {rmse}\nMAE: {mae}',
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=10,
                        transform=axs[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axs[i].xaxis.set_major_formatter(myFmt)
        fig.legend()
        fig.suptitle('Real v. Predicted values 24h')
        plt.tight_layout()
        plt.show()



    # def forecast(self):


        # for i in range(1, len(test_result)):
        #     # print(test_y.iloc[i])
        #     test_result.iloc[i] = test_result.iloc[i]+test_result.iloc[i-1]
