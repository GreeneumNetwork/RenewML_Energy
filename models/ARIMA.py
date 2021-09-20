import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import greenium.utils.data
from greenium.utils.utils import config_plot


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

        config_plot()
        self._init_arima()

    def _init_arima(self):

        cols = list(self.train_set.columns)
        if 'max_power_johnson' in cols and 'max_power_gym' in cols and len(cols) > 2:
            raise KeyError('Too many endog columns. Select dataset with single target variable.')
        elif 'max_power_johnson' in cols and 'max_power_gym' in cols and len(cols) == 2:
            super(ARIMAModel, self).__init__(endog=self.dataclass.df['max_power_gym'],
                                             exog=self.dataclass.df.drop('max_power_gym', axis=1),
                                             order=self.order
                                             )
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
            if self.model_result.model.order != self.order:
                raise ValueError(f'Specified order {self.order} is not equal to that of loaded model {self.model_result.model.order}')
            return self.model_result

        self.model_result = super(ARIMAModel, self).fit()
        self.logger.info('Trained and fit')

        return self.model_result

    def summary(self, plot=True, save_png=False):

        self.logger.info(f'AIC: {self.model_result.aic}; AICC: {self.model_result.aicc}')
        self.logger.info(f'BIC: {self.model_result.bic}')
        self.logger.info(f'HQIC: {self.model_result.hqic}')
        self.logger.info(f'Total RMSE: {np.sqrt(self.model_result.mse)}')
        # self.logger.info(f'Model Parameter Standard Error: {self.model_result.bse}')
        print(self.model_result.summary())

        residuals = self.model_result.resid.to_frame()
        print(residuals.describe())

        if plot:
            for col in residuals.columns:
                fig, axs = plt.subplots(1, 2)
                residuals[col].plot(ax=axs[0],
                                    title=f'Residuals for ARIMA Model order {self.order}')
                residuals[col].plot(ax=axs[1],
                                    title=f'KDE of Residuals of ARIMA Model order {self.order}',
                                    kind='kde')
                axs[1].axvline(residuals[col].mean(), c='r')
                axs[1].text(residuals[col].mean(), np.amax(axs[1].lines[0].get_ydata()),
                            s=f'{residuals[col].mean():0.3e}',
                            ha='left',
                            va='bottom',
                            )
                axs[1].legend()
                if save_png:
                    plt.savefig(f'figures/transparent/residuals/arima_{self.order}.png', transparent=True)

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

        real = self.dataclass.raw_data[self.dataclass.raw_data.columns[-1]].to_frame()
        real = real[start:end]

        pred_results = self.model_result.get_prediction()
        pred = self.dataclass.inverse_transform(pred_results.predicted_mean.rename('max_power'))
        pred = pred[start:end]

        ci_upper = pred['max_power'] + 1.96 * pred_results.se_mean[start:end]
        ci_lower = pred['max_power'] - 1.96 * pred_results.se_mean[start:end]
        mean_ci = (1.96 * pred_results.se_mean).mean()

        if plot:
            # Plot predictions
            fig, ax = plt.subplots(len(real.columns), 1, figsize=(16, 10))
            fig.subplots_adjust(hspace=0)
            myFmt = DateFormatter("%H:%M")

            rmse = mean_squared_error(real, pred, squared=False)
            r2 = r2_score(real, pred)
            mape = np.mean((np.abs(real - pred))/(np.amax(real)-np.amin(real))).item()*100

            real.rename(columns={'max_power': 'Real'}).plot(ax=ax, label='Real', c='b')
            pred.rename(columns={'max_power': 'Predicted'}).plot(ax=ax, label='Predicted', c='r')
            ax.fill_between(pred.index, ci_lower, ci_upper, color='r', alpha=0.1, label='95% C.I.')
            ax.set_title('Power Output', y=1.0, pad=-14)
            ax.set_ylabel('Output (kW)', fontsize=8)
            ax.set_ylim(
                [None, np.amax([np.amax([real, pred]) * 1.4, np.amax([real, pred]) + 0.005])])
            ax.text(0.1, 0.9, f'RMSE: {rmse:.2f}  '
                              f'$R^2$: {r2:.2f}\n'
                              f'MAPE: {mape:0.2f}%  '
                              f'95% C.I.: $\pm${mean_ci:.2f}',
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=8,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.xaxis.set_major_formatter(myFmt)
            ax.set_xticks([])
            ax.set_xlabel('Time (hr)')

            ax.legend()
            fig.suptitle('Real v. Predicted values 24h')
            if save_png:
                plt.savefig(f'figures/transparent/arima_{save_png}', transparent=True)
            plt.show()

        return pred, real
