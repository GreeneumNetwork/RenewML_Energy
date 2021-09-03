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
                 order: tuple = (0,0,0),
                 load: str = None
                 ):
        self.dataclass = dataclass
        self.scaler = dataclass.scaler or None
        self.dataset = dataclass.df
        self.logger = dataclass.logger
        self.order = order
        self.load = load
        self.train_set, self.test_set = self.split(train_percent)
        # model parameters

        self._init_arima()

    def _init_arima(self):

        cols = list(self.train_set.columns)
        if 'max_power_johnson' in cols and 'max_power_gym' in cols:
            raise KeyError('Too many endog columns. Select dataset with single target variable.')
        elif 'max_power' in cols:
            super(ARIMAModel, self).__init__(endog=self.dataclass.df['max_power'],
                                             exog=self.dataclass.df.drop('max_power', axis=1),
                                             order=self.order
                                             )
        else:
            raise KeyError('No endog columns available. Select dataset with max_power.')

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

    def summary(self, plot=True):

        print(self.model_result.summary())
        self.logger.info(f'AIC: {self.model_result.aic}')
        self.logger.info(f'Total MSE: {self.model_result.mse}')

        residuals = self.model_result.resid
        print(residuals.describe())

        if plot:
            fig, axs = plt.subplots(1, 2)
            residuals.plot(ax=axs[0],
                           title=f'Residuals for VAR Model order {self.order}')
            residuals.plot(ax=axs[1],
                           title=f'KDE of Residuals of VAR Model order {self.order}',
                           kind='kde',
                           label=residuals.name)
            axs[1].axvline(residuals.mean(), c='r')
            axs[1].text(residuals.mean(), np.amax(axs[1].lines[0].get_ydata()),
                        s=f'{residuals.mean():0.3e}',
                        ha='left',
                        va='bottom',
                        )
            axs[1].legend()
            plt.show()

    def predict(self, start: str, end: str,
                *args,
                plot: bool = True,
                save_png: str = None,
                **kwargs):

        """
               Predictions for in-sample dates.

               :rtype: object
               """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        num_hours = np.round((end - start).value / (60 * 60 * 10e8)).astype(int)

        real = self.dataclass.raw_data['max_power']
        pred = self.model_result.predict()
        real = real.to_frame()
        pred = pred.to_frame(name=real.columns[0])
        pred = self.dataclass.inverse_transform(pred)
        pred = pred[start:end]
        real = real[start:end]

        if plot:
            # Plot predictions
            fig, ax = plt.subplots(len(real.columns), 1, figsize=(16, 10))
            fig.subplots_adjust(hspace=0)
            myFmt = DateFormatter("%H:%M")

            real_vals = real.loc[start:end]
            pred_vals = pred.loc[start:end]
            rmse = mean_squared_error(real_vals, pred_vals, squared=False)
            mae = mean_absolute_error(real_vals, pred_vals)
            r2 = r2_score(real_vals, pred_vals)
            self.logger.info(f'RMSE max_power: {rmse}')
            real_vals /= 1000
            pred_vals /= 1000
            real_vals.rename(columns={'max_power': 'Real'}).plot(ax=ax,
                           label='Real', c='b')
            pred_vals.rename(columns={'max_power': 'Predicted'}).plot(ax=ax,
                           label='Predicted', c='r')
            ax.set_title('Power Output', y=1.0, pad=-14)
            ax.set_ylabel('Output (kW)', fontsize=8)
            ax.set_ylim(
                [None, np.amax([np.amax([real_vals, pred_vals]) * 1.4, np.amax([real_vals, pred_vals]) + 0.005])])
            ax.text(0.1, 0.9, f'RMSE: {rmse:.3f}\n$R^2$: {r2:.3f}\n'
                              f'MAPE: {mae / (np.amax(real_vals) - np.amin(real_vals)).item():0.3f}',
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=8,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.xaxis.set_major_formatter(myFmt)
            ax.set_xticks([])

            ax.set_xlabel('Time (hr)')
            fig.suptitle('Real v. Predicted values 24h')
            if save_png:
                plt.savefig(f'scratch/figures/transparent/arima_{save_png}', transparent=True)
            plt.show()

        return pred, real
