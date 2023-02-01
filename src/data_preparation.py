"""
This module contains all the functions used preprocess and prepare the data
"""

import logging
import os
import sys
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from causalimpact import CausalImpact
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from scipy.fft import irfft, rfft, rfftfreq
from skimage.restoration import denoise_wavelet

from src import costants, data_acquisition, logging_utilities


def data_preparation_stage1(data):
    """
    Executes the first data acquisition part of the project
    (before visualizing the data).

    Args:
        - data (Dict): contains all the data to prepare/clean

    Returns:
        - prepared_data (Dict): contains all the (not totally) prepared and cleaned data
    """
    logging_utilities.print_name_stage_project("DATA PREPARATION STAGE 1")

    check_missing_values(data=data)

    # HANDLES MISSING VALUES
    prepared_data = fill_missing_values(data=data)
    logging.info("- MISSING VALUES HANDLED")

    # Removes the data of the economic indicators before a certain date
    # (because that is the first day on which we have all the data)
    start_date_economic_indicators = "2018-01-31"
    end_date_economic_indicators = "2022-04-29"
    prepared_data[costants.COLLECTION_ECONOMIC_INDICATORS] = cut_economic_indicators_data(
        df=prepared_data[costants.COLLECTION_ECONOMIC_INDICATORS], start_date=start_date_economic_indicators,
        end_date=end_date_economic_indicators)
    logging.info(
        f"\n- Removed period before {start_date_economic_indicators} and after {end_date_economic_indicators} of '{costants.COLLECTION_ECONOMIC_INDICATORS}'")

    # data_preparation.check_missing_values_from_df(
    #     df=df_stock_data)

    # data_preparation.check_missing_values_from_df(
    #     df=df_covid_data)

    # data_preparation.check_missing_days_from_csv(
    #     costants.AAPL_STOCK_DATA_CSV, "\t")

    # data_preparation.check_noisy_data_from_csv(
    #     costants.AAPL_STOCK_DATA_CSV, "\t")

    # data_preparation.check_missing_values_from_csv(
    #     "xv/bashvxas", "\t")

    # data_preparation.prophet_anomaly_detection(
    #     costants.AAPL_STOCK_DATA_CSV, "\t", variable="Close")

    return prepared_data


def data_preparation_stage2(data):
    """
    Executes the second data acquisition part of the project
    (after visualizing the data).

    Args:
        - data (Dict): contains all the data to prepare/clean

    Returns:
        - final_df (pd.Dataframe): contains all the
        prepared and cleaned data ready for the inference stage
        (all data)
        - df_test (pd.Dataframe): contains the test set ('Adj Close'
        of May 2022)
        (all data)
        - final_df_only_close (pd.Dataframe): contains all the
        prepared and cleaned data ready for the inference stage (only adj. close)
        - df_test_only_close (pd.Dataframe): contains the test set ('Adj Close'
        of May 2022) (only adj. close)
    """
    logging_utilities.print_name_stage_project("DATA PREPARATION STAGE 2")
    # - drop columns
    prepared_data = drop_columns(data)
    logging.info("\n- COLUMNS DROPPED")

    # - outliers (probably going to keep them)
    handle_outliers(prepared_data)
    logging.info("\n- OUTLIERS HANDLED")

    # - noise
    #prepared_data = denoise_data(prepared_data)

    # - seas. adjustment
    #prepared_data = seasonal_adjustment(prepared_data)

    # if not only_stock_data:
    # - join data (remove periods where some data is not presents aka cut data)

    # MULTIVARIATE
    final_df = join_data(prepared_data)
    logging.info("\n- JOINED DATA")
    check_missing_values_from_df(df=final_df)

    if not os.path.exists(costants.DATA_FOLDER):
        os.mkdir(costants.DATA_FOLDER)

    final_df.describe().to_csv(
        os.path.join(costants.DATA_FOLDER,
                     "descriptive_statistics_all_data"),
        sep=costants.SEPARATOR)
    df_test = data_acquisition.acquire_test_data()
    df_test = scale_test_df(df=df_test,
                            df_train=final_df)
    # - scale
    final_df = scale_df(df=final_df)
    logging.info("\n- COLUMNS DROPPED")

    # UNIVARIATE
    final_df_only_close = prepared_data["stock_data"][["Adj Close"]].copy()
    check_missing_values_from_df(df=final_df_only_close)
    final_df_only_close.describe().to_csv(
        os.path.join(costants.DATA_FOLDER,
                     "descriptive_statistics_only_adj_close"),
        sep=costants.SEPARATOR)
    df_test_only_close = data_acquisition.acquire_test_data()
    df_test_only_close = scale_test_df(df=df_test_only_close,
                                       df_train=final_df_only_close)
    # - scale
    final_df_only_close = scale_df(df=final_df_only_close)

    return final_df, df_test, final_df_only_close, df_test_only_close


def join_data(data):
    """
    Joins all the data acquired and prepared into
    one dataframe before the final step of inference

    Args:
        - joined_df (pd.Dataframe): containd all the joined data
    """

    joined_df = data["covid_data"].copy()

    for df_name, df in data.items():
        if df_name != "covid_data":
            joined_df = joined_df.join(df, how='left')

    return joined_df


def handle_outliers(data):
    """
    Handles outliers in the data

    Args:
        - data (Dict): contains all the data to prepare/clean

    Returns: None

    """
    handle_covid_outliers(data=data)


def handle_covid_outliers(data):
    """
    Handles outliers in the covid data, that have
    been identified as errors.

    Args:
        - data (Dict): contains the data

    Returns: None
    """
    for covid_label, outliers_dates in costants.COVID_OUTLIERS.items():
        for date in outliers_dates:
            data["covid_data"].loc[data["covid_data"].index ==
                                   date, covid_label] = None
    data["covid_data"].interpolate(inplace=True)


def drop_columns(data):
    """
    Removes certain columns (variables) from the data

    Args:
        - data (Dict): contains all the data

    Returns:
        - prepared_data (Dict): contains all the data
        without the columns that we wanted removed
    """

    for data_name, df in data.items():
        columns_to_remove = costants.COLUMNS_TO_REMOVE[data_name]
        df.drop(labels=columns_to_remove, axis='columns', inplace=True)

    return data


# def seasonal_decomposition(df_column):
#     """
#     Decomposes a time series into seasonality, trend
#     and residual

#     Args:
#         df_column (pd.Series): contains the time series data
#     """
#     result = seasonal_decompose(df_column)

#     return
#     result.plot()
#     plt.show()

# def scale_data(data):
#     """
#     Scales the data.

#     Args:
#         - data (Dict): contains all the data

#     Returns:
#         - scaled_data (Dict): contains all the
#         scaled data
#     """
#     for data_name, df in data.items():
#         df = scale_df(df=df)


def scale_df(df):
    """
    Scale every column of a dataframe between 0 and 1
    using te following formula

    x_scaled = (x - min) / (max - min)

    Args:
        - df (pd.Dataframe): dataframe that we want to scale

    Returns:
        - df_scaled (pd.Dataframe): scaled dataframe
    """
    df_scaled = df.copy()
    for column in df.columns:
        min = df_scaled[column].min()
        max = df_scaled[column].max()
        df_scaled[column] = (df_scaled[column] - min) / (max - min)

    return df_scaled


def scale_test_df(df, df_train):
    """
    Scales the test set between 0 and 1
    using the following formula

    x_scaled = (x - min) / (max - min)

    Args:
        - df (pd.Dataframe): dataframe that we want to scale
        - df_train (pd.Dataframe): contains the training set
        (needed to scale the test set with the same parameters
        as the training set)

    Returns:
        - df_scaled (pd.Dataframe): scaled dataframe
    """
    df_scaled = df.copy()
    # print(df_train)
    column = "Adj Close"
    min = df_train[column].min()
    max = df_train[column].max()
    df_scaled[column] = (df_scaled[column] - min) / (max - min)

    return df_scaled


def wavelet_denoising(noisy_input, wavelet):
    """
    Denoises a signal using Wavelet Transform

    Args:
        - noisy_input (np.ndarray): noisy data
        - wavelet (str): inidicates what wavelet to use (see
        scikit-image documentation for more details)

    Returns:
        - denoised_output (np.ndarray): denoised data
    """
    denoised_output = denoise_wavelet(image=noisy_input, wavelet=wavelet)

    return denoised_output


def fft_denoising(noisy_input, threshold, data_step=0.001):
    """
    Denoises a signal using FFT (Fast Fourier Transform)

    Args:
        - noisy_input (np.ndarray): noisy data
        - threshold (float): threshold we want to use in the
        frequency domain to filter noise

    Returns:
        - denoised_output (np.ndarray): denoised data
        - yf_abs (np.ndarray): noisy signal in the frequency
        domain
        - yf_clean (np.ndarray): frequency domain signal after
        filtering with the threshold
    """
    n = len(noisy_input)
    yf = rfft(noisy_input)
    xf = rfftfreq(n, data_step)
    yf_abs = np.abs(yf)
    indices = (yf_abs > threshold)
    yf_clean = indices * yf
    denoised_output = irfft(yf_clean)
    yf_clean = np.abs(yf_clean)

    return denoised_output, yf_abs, yf_clean


def cut_economic_indicators_data(df, start_date, end_date):
    """
    Removes a first period from a dataframe with a date index

    Args:
        - df (pd.Dataframe): the dataframe from which
        we want to remove the initial missing periods
        - start_date (str): first day of the period of
        which we want to keep the data
        - end_date (str): last day of the period of
        which we want to keep the data

    Returns:
        - new_df (pd.Dataframe): the new dataframe with period
        before 'first_day' removed
    """
    mask = ((df.index >= start_date) & (df.index <= end_date))
    new_df = df.loc[mask]

    return new_df


def fill_missing_values(data):
    """
    Deals with the missing values in the acquired data

    Args:
        - data (Dict): contains all the data to fill

    Returns: None
    """
    data[costants.COLLECTION_STOCK_DATA] = fill_df(
        raw_data=data[costants.COLLECTION_STOCK_DATA],
        fill_func='linear'
    )
    data[costants.COLLECTION_COVID_DATA] = fill_df(
        raw_data=data[costants.COLLECTION_COVID_DATA],
        fill_func='zeros'
    )
    data[costants.COLLECTION_TECHNICAL_INDICATORS] = fill_df(
        raw_data=data[costants.COLLECTION_TECHNICAL_INDICATORS],
        fill_func='linear'
    )
    data[costants.COLLECTION_ECONOMIC_INDICATORS] = fill_df(
        raw_data=data[costants.COLLECTION_ECONOMIC_INDICATORS],
        fill_func='linear'
    )

    return data


def fill_df(raw_data, fill_func):
    """
    It first insert the missing days in the time series data
    and then fills the missing values

    Args:
        - raw_data (pd.DataFrame): contains the acquired raw data

    Returns:
        - filled_data (pd.DataFrame): contains the data with
        filled missing values
    """
    filled_data = raw_data.asfreq(
        freq='D')

    if fill_func == "linear":
        # interpolate and then fill with zeros because
        # interpolating only will leaving the first
        # missing values to NaN
        filled_data = filled_data.interpolate(method='linear')
        filled_data = filled_data.fillna(value=0)
    elif fill_func == "zeros":
        filled_data = filled_data.fillna(value=0)
    else:
        raise ValueError(
            "The function fill_data has a non possible value for the parameter fill_func")

    return filled_data


def check_missing_values(data):
    """
    Checks for missing values in acquired data

    Args:
        - data (Dict): contains all the data to check
    """
    for data_name, data_value in data.items():
        logging.info(f"- Checking for missing values in '{data_name}' data\n")
        data_value_daily = data_value.asfreq(freq='D')
        check_missing_values_from_df(data_value_daily)


def check_missing_values_from_df(df):
    """
    Reads a dataframe and checks if there are any missing values
    for every column of the dataframe.

    Args:
        - df (pandas.core.frame.DataFrame)): dataframe we want to check for missing values

    Returns: None
    """
    total_missing = 0
    dict_missing = {}
    for column in df.keys():
        missing_in_column = df[column].isnull().sum()
        dict_missing[column] = missing_in_column
        total_missing += missing_in_column
        logging.info(
            f"Column {column} number of missing values: {missing_in_column}")

    columns_with_missing_values = {
        column: dict_missing[column] for column in df.keys() if dict_missing[column] > 0}

    if total_missing > 0:
        logging.info(
            f"\nWARNING: THE DATA PRESENTS {total_missing} MISSING VALUES AS FOLLOWS\n")
        logging.info(f"{pformat(columns_with_missing_values)}")
    else:
        logging.info("\nTHE DATA DOES NOT PRESENT ANY MISSING VALUES.")
    logging.info("\n")


def check_missing_days_from_csv(path_csv, separator):
    """
    Reads a csv that contains time series data and checks
    for missing days.

    Args:
        - path_csv (str): path of the csv file
        - separator (str): separator for the function read_csv()

    Returns: None
    """
    # if not os.path.isfile(path_csv):
    #     function_name = sys._getframe().f_code.co_name
    #     raise FileNotFoundError(
    #         f"The file you are trying to read does not exist.\nCheck the path you are passing to the function {function_name}.")
    raise NotImplementedError


def check_noisy_data_from_csv(path_csv, separator):
    """
    """
    # if not os.path.isfile(path_csv):
    #     function_name = sys._getframe().f_code.co_name
    #     raise FileNotFoundError(
    #         f"The file you are trying to read does not exist.\nCheck the path you are passing to the function {function_name}.")
    raise NotImplementedError


def prophet_anomaly_detection_csv(path_csv, separator, variable):
    """
    Reads a csv that contains time series data and performs
    anomaly detection (also called outlier detection) using
    the Meta Prophet model (https://facebook.github.io/prophet/).

    Args:
        - path_csv (str): path of the csv file
        - separator (str): separator for the function read_csv()
        - variable (str): variable (column) on which we want to perform
        the anomaly detection

    Returns: None
    """
    if not os.path.isfile(path_csv):
        function_name = sys._getframe().f_code.co_name
        raise FileNotFoundError(
            f"The file you are trying to read does not exist.\nCheck the path you are passing to the function {function_name}.")

    df = pd.read_csv(path_csv, sep=separator)
    df = df[["Date", variable]]
    df = df.rename(columns={"Date": "ds", variable: "y"})
    # print(df.head())

    prophet_model = Prophet(interval_width=0.99,
                            yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(df)
    forecast = prophet_model.predict(df)

    prophet_model.plot(forecast)
    plots_directory = os.path.join(costants.PLOTS_FOLDER,
                                   "prophet_anomaly_detection")
    plt.savefig(os.path.join(plots_directory, "prophet_forecast"))
    prophet_model.plot_components(forecast)
    plt.savefig(os.path.join(plots_directory, "prophet_forecast_components"))
    # trend changepoints plotting
    fig = prophet_model.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), prophet_model, forecast)
    plt.savefig(os.path.join(plots_directory,
                "prophet_forecast_trend_changepoints"))


# def causal_impact_analysis(path_csv, separator, variable, pre_period, post_period):
#     """"
#     Reads a csv that contains time series data and performs causal impact
#     analysis regarding an event, that is it analyses the impact of an event
#     on the time series measuring the difference between the forecast done on
#     the timeseries after that event of a model trained on the time series until
#     that event and the actual time series after the event.
#     It uses the library developed by Google CasualImpact
#     (https://google.github.io/CausalImpact/CausalImpact.html).

#     Args:
#         - path_csv (str): path of the csv file
#         - separator (str): separator for the function read_csv()
#         - variable (str): variable (column) on which we want to perform
#         the causal impact
#         - pre_period (list): contains to dates that indicate the start
#         and the end dates of the period before the event
#         - post_period (list): contains to dates that indicate the start
#         and the end dates of the period after the event

#     Returns:
#         None
#     """
#     if not os.path.isfile(path_csv):
#         function_name = sys._getframe().f_code.co_name
#         raise FileNotFoundError(
#             f"The file you are trying to read does not exist.\nCheck the path you are passing to the function {function_name}.")

#     df = pd.read_csv(path_csv, sep=separator)
#     df = df[['Date', variable]]
#     df.set_index('Date', inplace=True)
#     # print(df)

#     # Plots:
#     # 1- The observed ‘post-event’ time series and fitted model’s forecast
#     # 2- The pointwise causal effect, as estimated by the model(difference between forecasted and actual time series)
#     # 3- The cumulative effect
#     ci_plots_directory = os.path.join(
#         costants.PLOTS_FOLDER, "causal_impact_analysis")
#     plot_path = os.path.join(ci_plots_directory,
#                              "observed_pointwise_cumulative")

#     ci = CausalImpact(df, pre_period, post_period, prior_level_sd=None)
#     CausalImpact.plot = causalimpact_override.new_plot_causalimpact
#     ci.plot(fname=plot_path)
#     logging.info(ci.summary())


def normalize_and_scale():
    raise NotImplementedError
