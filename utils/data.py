import logging
from inspect import stack
from copy import deepcopy
from os.path import basename
from pathlib import Path
import pandas as pd
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from itertools import permutations
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Data():

    def __init__(self, df: pd.DataFrame, filename: str, raw_weather, raw_power):

        self.scaler_ = None
        self.lags = None
        self.df = df
        self.logger = logging.getLogger(basename(stack()[-1].filename))
        self.raw_data = df
        self.raw_weather = raw_weather
        self.filename = filename

    def __repr__(self):
        return self.df

    def __str__(self):
        return str(self.df)

    @classmethod
    def get_data(cls,
                 datafile,
                 powerfile: str = None,
                 dropna: bool = True,
                 rescale_power: bool = True):

        logger = logging.getLogger(basename(stack()[-1].filename))
        df = pd.read_csv(datafile,
                         usecols=['validdate','t_2m:C','global_rad:W','precip_1h:mm','effective_cloud_cover:p'],
                         index_col='validdate', parse_dates=True)
        logger.info(f'Weather data file found at {datafile}')
        df.index = df.index.tz_localize(None)
        raw_weather = df.copy()

        if powerfile:
            try:
                power_df = pd.read_csv(powerfile, usecols=['timestamp', 'max_power'], index_col='timestamp',
                                       parse_dates=True)
            except ValueError:
                #if files are from 15 minute data
                logger.info(f'15 Minute interval power file found at {powerfile}')
                power_df = pd.read_csv(powerfile, usecols=['Date_&_Time', 'Power', 'Irradiance'],
                                       index_col='Date_&_Time',
                                       parse_dates=True,
                                       na_values='None'
                                       )
                raw_power = power_df.copy()
                power_df = power_df.rename(columns={'Power': 'max_power', 'Irradiance': 'global_rad:W'})
                power_df.index = power_df.index.rename('timestamp')

                for col in power_df.columns:
                    power_df = power_df[power_df[col].isnull().astype(int).groupby(
                        power_df[col].notnull().astype(int).cumsum()).cumsum() <= 1]
                power_df = power_df.dropna(how='all').asfreq('15T')
                power_df.iloc[0:5] = power_df.iloc[0:5].fillna(method='bfill')
                power_df = power_df.groupby([power_df.index.minute, power_df.index.hour]).fillna(method='ffill')
                power_df = power_df.interpolate(method='time').dropna().asfreq('15T').interpolate(method='time')
                df = df.asfreq('15T').interpolate(method='time')
                df = df.drop(columns=['global_rad:W'])

            if rescale_power:
                power_df['max_power'] /= 1000
            df = pd.concat([df, power_df], axis=1, join='inner')
        df = df.asfreq(pd.infer_freq(df.index))

        if dropna:
            df = df.dropna()

        return cls(df, Path(powerfile).stem, raw_weather, raw_power)

    def transform(self,
                  lag: list or str,
                  resample=None,
                  scaler=None,
                  copy=True):
        """Make input data stationary, resampling to daily frequency. Input choice of {day|week|month|season|year}
        for differencing or list of multiple for multi-order differencing.
        Options:
        :param lag: {day|week|month|season|year} (list)
        :param scaler: [ minmax | standard ]
        :param resample: Numpy frequency string.
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

        lag_dict = {'15MINUTES': pd.Timedelta('15T'),
                    'MINUTE': pd.Timedelta('1T'),
                    'HOUR': pd.Timedelta('1H'),
                    'DAY': pd.Timedelta('1D'),
                    'WEEK': pd.Timedelta('1W'),
                    'MONTH': DateOffset(months=1),
                    'SEASON': DateOffset(months=3),
                    'YEAR': DateOffset(months=12)}

        # resample dataframe to daily values
        if resample:
            data_mean = transformed.resample(pd.Timedelta(resample)).mean()
            newcls.logger.info(f'Resample to: {resample}')
            # newcls.logger.info(f'Daily mean values:\n{data_mean.describe().to_string()}')

            if pd.Timedelta(resample) == pd.Timedelta('1H'):
                data_mean = self.raw_weather.join(data_mean[['max_power']], how='right')

            transformed = data_mean


        # make sure input is list
        if type(lag) == str:
            lag = [lag]

        # make list of indexes to lag
        try:
            lags = []
            for i in lag:
                lags.append(lag_dict[i.upper()])
        except KeyError:
            raise KeyError(f'{i} not valid choice. Choose from {{15minutes|Minute|hour|day|week|month|season|year}}')

        newcls.lags = lags
        # apply differencing
        diff = lambda x, i_l, i_lag: [x[i] - x[i - i_lag] for i in x.loc[i_l:].index]
        # index = transformed.index
        l = transformed.index[0]
        for lag_num in newcls.lags:
            l += lag_num
            new = transformed.apply(lambda x: diff(x, l, lag_num), axis=0)
            new.index = transformed[l:].index
            transformed.loc[new.index] = new

        newcls.trunc = transformed.loc[np.setdiff1d(transformed.index, new.index)]
        transformed = transformed.loc[new.index]

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

        if self.scaler_:
            df[df.columns] = self.scaler_.inverse_transform(df[df.columns])

        if isinstance(df, pd.Series):
            df = df.to_frame()

        # prepend dataframe with initial values
        # start_idx = self.raw_data.index.get_loc(df.index[0])
        try:
            x_i = pd.concat([self.trunc, df], join='inner', axis=0)
        except AttributeError as e:
            raise e

        # invert_diff = lambda x, distance: [x[i] + x[i - distance] for i in range(len(x))]
        def invert_diff(x: pd.Series, i_l: pd.Timestamp, i_lag: pd.Timedelta):
            arr = x.values
            for i in x[i_l:].index:
                s = x[i] + x[i - i_lag]
                x[i] = s
            return pd.Series(data=arr, index=x.index)

        l = np.sum(self.lags) + x_i.index[0]
        for lag_num in reversed(self.lags):
            inverted = x_i.apply(lambda x: invert_diff(x, l, lag_num), axis=0)
            l -= lag_num

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

    def granger(self, *args, **kwargs):
        """
        Use to display granger causality metrics up to lag.
        :param args:
        :param kwargs:
        :return:
        """
        for (col1, col2) in permutations(self.df.columns, 2):
            g = grangercausalitytests(self.df[[col1, col2]], *args, **kwargs)
            print('=' * 50)
            print(f'\nGranger Causality Metrics {col2} on {col1}')
            print('-' * 80)
            print(f'{"Null Hypothesis":60}Lag\tF-Stat\tDecision')
            print('-' * 80)
            for (key, val) in g.items():
                val = val[0]
                if key == 1:
                    print(f'{col2 + " does not cause " + col1:60}', end='')
                else:
                    key = str(key).rjust(61)
                print(f'{key}\t{val["ssr_ftest"][0]:0.2f}\t{"Reject" if val["ssr_ftest"][1] <= 0.05 else "Accept"}')

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

    def FFT(self, raw=False, axs=None):

        df = self.raw_data if raw else self.df

        if axs is None:
            fig, axs = plt.subplots(len(df.columns), 1, sharex=True)
            fig.subplots_adjust(hspace=0)

        SAMPLE_RATE = 24 * 365  # h/year

        for ax, col in enumerate(df.columns):
            yf = np.fft.rfft(df[col])
            xf = np.fft.rfftfreq(len(df[col]), 1 / SAMPLE_RATE)
            sns.lineplot(x=xf[xf < 800], y=np.abs(yf)[xf < 800], ax=axs[ax])
            axs[ax].text(.5, .9, col,
                         horizontalalignment='center',
                         verticalalignment='top',
                         transform=axs[ax].transAxes)
            axs[ax].set_xlabel(None)
        axs[-1].set_xlabel('Frequency $(year^{-1})$')

        return axs
