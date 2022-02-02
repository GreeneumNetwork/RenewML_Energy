import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

import greenium.utils.data
from greenium.utils.utils import config_plot


class VARModel(VARMAX):

    def __init__(self,
                 dataclass: greenium.utils.data.Data,
                 order: tuple = (1, 0),
                 train_percent: float = 0.7,
                 load: str = None):

        self.dataclass = dataclass
        self.scaler = dataclass.scaler or None
        self.dataset = dataclass.df
        self.logger = dataclass.logger
        self.load = load
        self.train_set, self.test_set = self.split(train_percent)
        # model parameters
        self.order = order

        self._init_var()

        config_plot()

    def _init_var(self):
        super(VARModel, self).__init__(
            self.train_set,
            order=self.order)
        self.logger.info(f'VARMax model initiated, order {self.order}')

    def split(self, train_percent):

        stop_idx = np.floor(train_percent * len(self.dataset)).astype(int)
        train_set = self.dataset.iloc[:stop_idx]
        test_set = self.dataset.iloc[stop_idx:]
        # self.train_set = self.train_set.set_index('validdate').asfreq('D')
        # self.test_set = self.test_set.set_index('validdate').asfreq('D')

        return train_set, test_set

    def fit(self, **kwargs) -> object:
        """
        Inherits from MLEResults.fit()

        :param kwargs:
        :return:
        """
        if self.load:
            self.model_result = VARMAXResults.load(self.load)
            self.logger.info(f'model loaded from {self.load}')
            if self.model_result.model.order != self.order:
                raise ValueError(
                    f'Specified order {self.order} is not equal to that of loaded model {self.model_result.model.order}')
            return self.model_result

        t_start = datetime.now()
        self.model_result = super(VARModel, self).fit(maxiter=1000, disp=False)
        t_end = datetime.now()
        self.logger.info(f'Trained and fit. Training time = {t_end - t_start}')

        return self.model_result

    def predict(self,
                start: str, end: str,
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
        idx = pd.date_range(start=start, end=end, freq=self.dataset.index.freq)

        real = self.dataclass.raw_data
        # pred = self.model_result.predict(*args, **kwargs)
        pred_results = self.model_result.get_prediction()
        pred_mean = self.dataclass.inverse_transform(pred_results.predicted_mean)

        if start not in pred_mean.index or end not in pred_mean.index:
            raise IndexError(f'Start and end dates out of range for predictions {pred_mean.index[0]} to {pred_mean.index[-1]}')

        pred = pred_mean.loc[idx]
        real = real.loc[idx]

        ci_upper = pred + 1.96 * pred_results.se_mean.loc[idx]
        ci_lower = pred - 1.96 * pred_results.se_mean.loc[idx]
        mean_ci = (1.96 * pred_results.se_mean).mean()

        if plot:
            # Plot predictions
            fig, axs = plt.subplots(len(real.columns), 1, figsize=(16, 10))
            fig.subplots_adjust(hspace=0)
            myFmt = DateFormatter("%H:%M")
            label_dict = {'max_power_gym': ('Power Output Gym Dataset', 'Output (kW)'),
                          'max_power_johnson': ('Power Output Johnson Dataset', 'Output (kW)'),
                          'max_power': ('Power Output', 'Output (kW)'),
                          't_2m:C': ('Temperature', 'Degrees Celsius'),
                          'global_rad:W': ('Global Irradiance', 'Irradiance $(W/m^2)$'),
                          'effective_cloud_cover:p': ('Effective Cloud Cover', 'Percent'),
                          'precip_1h:mm': ('Precipitation', 'mm/hr')}

            for i, col in enumerate(real.columns):

                real_vals = real[col]
                pred_vals = pred[col]
                ci_upper_vals = ci_upper[col]
                ci_lower_vals = ci_lower[col]

                rmse = mean_squared_error(real_vals, pred_vals, squared=False)
                r2 = r2_score(real_vals, pred_vals)
                mape = np.mean((np.abs(real_vals - pred_vals)) / (np.amax(real_vals) - np.amin(real_vals))).item() * 100
                self.logger.info(f'Prediction interval RMSE {col}: {rmse}')

                axs[i].plot(real_vals, label='Real' if i == 0 else '_nolegend_', c='b')
                axs[i].plot(pred_vals, label='Predicted' if i == 0 else '_nolegend_', c='r')
                axs[i].fill_between(pred_vals.index, ci_lower_vals, ci_upper_vals, color='r', alpha=0.1,
                                    label='95% C.I.' if i == 0 else '_nolegend_')
                axs[i].set_title(label_dict[col][0], y=1.0, pad=-14)
                axs[i].set_ylabel(label_dict[col][1])
                axs[i].set_ylim(
                    [None, np.amax([np.amax([real_vals, pred_vals]) * 1.4, np.amax([real_vals, pred_vals]) + 0.005])])
                axs[i].text(0.1, 0.9, f'RMSE: {rmse:.2f}  '
                                      f'$R^2$: {r2:.2f}\n'
                                      f'MAPE: {mape:0.2f}%  '
                                      f'95% C.I.: $\pm${mean_ci[col]:.2f}',
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=8,
                            transform=axs[i].transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axs[i].xaxis.set_major_formatter(myFmt)
                if col != real.columns[-1]:
                    axs[i].set_xticks([])

            axs[i].set_xlabel('Time (hr)')
            fig.legend()
            fig.suptitle('Real v. Predicted values 24h')
            if save_png:
                plt.savefig(f'figures/transparent/varmax_{save_png}', transparent=True)
            plt.show()

        return pred, real

    def simulate(self, params, nsimulations, **kwargs):

        sim = super(VARModel, self).simulate(params, nsimulations, **kwargs)
        return sim

    def summary(self, plot=True, save_png=None):

        self.logger.info(f'AIC: {self.model_result.aic}; AICC: {self.model_result.aicc}')
        self.logger.info(f'BIC: {self.model_result.bic}')
        self.logger.info(f'HQIC: {self.model_result.hqic}')
        self.logger.info(f'Total RMSE: {np.sqrt(self.model_result.mse)}')
        # self.logger.info(f'Model Parameter Standard Error: {self.model_result.bse}')
        print(self.model_result.summary())

        residuals = self.model_result.resid
        print(residuals.describe())
        print("Root Sum of Squared Error\n", np.sqrt((residuals**2).mean()))

        if plot:
            for col in residuals.columns:
                fig, axs = plt.subplots(1, 2)
                residuals[col].plot(ax=axs[0],
                                    title=f'Residuals for VAR Model order {self.order}')
                residuals[col].plot(ax=axs[1],
                                    title=f'KDE of Residuals of VAR Model order {self.order}',
                                    kind='kde')
                axs[1].axvline(residuals[col].mean(), c='r')
                axs[1].text(residuals[col].mean(), np.amax(axs[1].lines[0].get_ydata()),
                            s=f'{residuals[col].mean():0.3e}',
                            ha='left',
                            va='bottom',
                            )
                axs[1].legend()
                if save_png:
                    plt.savefig(f'figures/transparent/residuals/varmax_{self.order}.png', transparent=True)

    def save(self, filename: str, remove_data: bool = False):
        self.model_result.save(f'models/saved_models/var_{filename}', remove_data=remove_data)
