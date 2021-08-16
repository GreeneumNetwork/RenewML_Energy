import logging

from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Data:

    def __init__(self,
                 df: pd.DataFrame,
                 logger: logging.Logger):

        self.scaler_ = None
        self.lags = None
        self.df = df
        self.logger = logger

        self.raw_data = df

    def __repr__(self):
        return self.df

    def __str__(self):
        return str(self.df)

    @classmethod
    def get_data(cls,
                 datafile,
                 logger,
                 powerfile: str = None):

        df = pd.read_csv(datafile, index_col='validdate', parse_dates=True)
        logger.info(f'Data file found at {datafile}')
        df.index = df.index.tz_localize(None)

        if powerfile:
            power_df = pd.read_csv(powerfile, usecols=['timestamp', 'max_power'], index_col='timestamp',
                                   parse_dates=True)
            df = pd.concat([df, power_df], axis=1, join='inner')

        return cls(df, logger)

    def transform(self,
                  lag: list or str,
                  resample=True,
                  scaler=None,
                  copy=True):
        """Make input data stationary, resampling to daily frequency. Input choice of {day|week|month|season|year}
        for differencing or list of multiple for multi-order differencing.
        Options:
            - lag: {day|week|month|season|year} (list)
            - Norm: [ minmax | standard ]
        """
        # make copy of class
        if copy:
            newcls = deepcopy(self)
        else:
            newcls = self

        newcls.scaler = scaler
        # load copy of dataframe
        df = newcls.df
        transformed = df.copy()
        # transformed = np.log(transformed)

        lag_dict = {'DAY': 1,
                    'WEEK': 7,
                    'MONTH': 30,
                    'SEASON': 3 * 30,
                    'YEAR': 365}

        # resample dataframe to daily values
        if resample:
            data_mean = transformed.resample('D').mean()
            # data_min = transformed.resample('D').min()
            # data_max = transformed.resample('D').max()
            newcls.logger.info(f'Resample shape: {data_mean.shape}')
            newcls.logger.info(f'Daily mean values:\n{data_mean.describe().to_string()}')
            transformed = data_mean
        else:
            lag_dict = {key: lag_dict[key] * 24 for key in lag_dict.keys()}
            lag_dict['HOUR'] = 1

        # make sure input is list
        if type(lag) == str:
            lag = [lag]

        # make list of indexes to lag
        try:
            lags = []
            for i in lag:
                lags.append(lag_dict[i.upper()])
        except KeyError:
            raise KeyError(f'{i} not valid choice. Choose from {{day|week|month|season|year}}')

        newcls.lags = lags
        # apply differencing
        diff = lambda x, distance: [x[i + distance] - x[i] for i in range(len(x) - distance)]
        index = transformed.index
        for lag_num in newcls.lags:
            transformed = transformed.apply(lambda x: diff(x, lag_num), axis=0)
            index = index[lag_num:]
            transformed.index = index
        # apply scaling
        if newcls.scaler == 'standard':
            newcls.scaler_ = StandardScaler()
            transformed[transformed.columns] = newcls.scaler_.fit_transform(transformed[transformed.columns])
        elif newcls.scaler == 'minmax':
            newcls.scaler_ = MinMaxScaler()
            transformed[transformed.columns] = newcls.scaler_.fit_transform(transformed[transformed.columns])

        newcls.raw_data = self.df
        newcls.df = transformed

        return newcls

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.scaler:
            df[df.columns] = self.scaler_.inverse_transform(df[df.columns])

        # prepend dataframe with initial values
        start_idx = self.raw_data.index.get_loc(df.index[0])
        x_i = pd.concat([self.raw_data.iloc[start_idx - np.max(self.lags):start_idx], df], join='inner')
        invert_diff = lambda x, distance: [x[i] + x[i - distance] for i in range(len(x))]
        for lag_num in self.lags:
            inverted = x_i.apply(lambda x: invert_diff(x, lag_num), axis=0)

        return inverted

    def ts_plot(self, lags=None, figsize=(10, 6)):
        df = self.df
        for column_name in df.columns:
            y = df[column_name]
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0))
            hist_ax = plt.subplot2grid(layout, (0, 1))
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title(column_name, fontsize=12, fontweight='bold')
            y.plot(ax=hist_ax, kind='hist', bins=25)
            hist_ax.set_title('Histogram')
            plot_acf(y, lags=lags, ax=acf_ax)
            plot_pacf(y, lags=lags, ax=pacf_ax)
            sns.despine()
            plt.tight_layout()
            plt.show()

    def plot(self):

        fig, axs = plt.subplots(len(self.df.columns), 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        for ax, col in enumerate(self.df.columns):
            self.df[col].plot(ax=axs[ax])
            axs[ax].text(.5, .9, col,
                         horizontalalignment='center',
                         verticalalignment='top',
                         transform=axs[ax].transAxes)
            if col != self.df.columns[-1]:
                axs[ax].set_xticklabels([])

        plt.show()

    def ADF(self):
        df = self.df
        for column_name in df.columns:
            y = df[column_name]
            result = adfuller(y)
            msg = str(column_name) + '/n'
            msg += '-' * 20
            msg += f'\nADF Statistic: {result[0]}\n'
            msg += f'p-value: {result[1]}\n'
            msg += 'Critical Values:\n'
            for key, value in result[4].items():
                msg += (f'\t{key}: {value:.3f}\n')

            print(msg)

    def plot_df(self):
        df = self.df
        for i in range(len(df.columns)):
            idx = df.columns[i]
            fig, (ax1, ax2) = plt.subplots(2, 1)
            df[idx].plot(ax=ax1)
            df[idx].hist(ax=ax2, bins=50)
            fig.suptitle(f'{idx} differenced on {lags}')
            plt.tight_layout()
            for tick in ax1.get_xticklabels():
                tick.set_rotation(45)
            plt.show()

    def FFT(self, axs=None):

        if axs is None:
            fig, axs = plt.subplots(len(self.df.columns), 1, sharex=True)
            fig.subplots_adjust(hspace=0)
        SAMPLE_RATE = 24*365 # h/year

        for ax, col in enumerate(self.df.columns):
            yf = np.fft.rfft(self.df[col])
            xf = np.fft.rfftfreq(len(self.df[col]), 1/SAMPLE_RATE)
            sns.lineplot(x=xf[xf < 800], y=np.abs(yf)[xf < 800], ax=axs[ax])
            axs[ax].text(.5, .9, col,
                         horizontalalignment='center',
                         verticalalignment='top',
                         transform=axs[ax].transAxes)
            axs[ax].set_xlabel('Frequency $(year^{-1})$')

        return axs
