from typing import Union
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .data import Data


def get_logger(
        script_name: str,
        log_file: Union[str, None] = None,
        stream_handler: bool = True,
) -> logging.getLogger:
    """Initiate the logger to log the progress into a file.

    Args:
    -----
        script_name (str): Name of the scripts outputting the logs.
        log_file (str): Name of the log file.
        stream_handler (bool, optional): Whether or not to show logs in the
            console. Defaults to True.

    Returns:
    --------
        logging.getLogger: Logger object.
    """
    logger = logging.getLogger(name=script_name)
    logger.setLevel(logging.INFO)

    if log_file is not None:
        # create handlers
        f_handler = logging.FileHandler(os.path.join(log_file), mode="a")
        # create formatters and add it to the handlers
        f_format = logging.Formatter(
            "%(asctime)s:%(name)s->%(funcName)s: %(levelname)s:%(message)s",
            "%H:%M:%S"
        )
        f_handler.setFormatter(f_format)
        # add handlers to the logger
        logger.addHandler(f_handler)

    # display the logs in console
    if stream_handler:
        s_handler = logging.StreamHandler()
        s_format = logging.Formatter("%(levelname)s: %(name)s->%(funcName)s: %(message)s")
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    return logger

def make_datasets(power_dataset: str) -> Data:
    # Get and normalize data
    raw_data = Data.get_data(datafile='data/4Y_Historical.csv')

    if 'johnson_gym' in power_dataset:
        raw_data_johnson_gym = Data.get_data(datafile='data/4Y_Historical.csv',
                                             powerfile='data/maabarot_johnson_from_2010_04_22_to_2020_12_31.csv')
        raw_data_gym = Data.get_data(datafile='data/4Y_Historical.csv',
                                     powerfile='data/gym_from_2010_04_06_to_2020_12_31.csv')
        raw_data_johnson_gym.df = pd.concat([raw_data_johnson_gym.df, raw_data_gym.df['max_power']], axis=1,
                                            join='inner')
        raw_data_johnson_gym.df.columns = np.concatenate((raw_data.df.columns, ['max_power_johnson', 'max_power_gym']))

        stationary = raw_data_johnson_gym.transform(lag=['hour', 'day'],
                                                    resample=False,
                                                    scaler=None)
        stationary.df['max_power_gym'] = stationary.df['max_power_gym'].astype(np.float64)

    elif 'gym' in power_dataset:
        raw_data_gym = Data.get_data(datafile='data/4Y_Historical.csv',
                                     powerfile='data/gym_from_2010_04_06_to_2020_12_31.csv')
        stationary = raw_data_gym.transform(lag=['hour', 'day'],
                                            resample=False,
                                            scaler=None)
        stationary.df['max_power'] = stationary.df['max_power'].astype(np.float64)

    elif 'johnson' in power_dataset:
        raw_data_johnson = Data.get_data(datafile='data/4Y_Historical.csv',
                                         powerfile='data/maabarot_johnson_from_2010_04_22_to_2020_12_31.csv')
        stationary = raw_data_johnson.transform(lag=['hour', 'day'],
                                                resample=False,
                                                scaler=None)
        stationary.df['max_power'] = stationary.df['max_power'].astype(np.float64)

    stationary.df = stationary.df.drop(columns=['diffuse_rad:W', 'direct_rad:W'])
    stationary.raw_data = stationary.raw_data.drop(columns=['diffuse_rad:W', 'direct_rad:W'])

    return stationary

def config_plot():
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
