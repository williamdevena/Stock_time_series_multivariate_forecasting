"""
This module contains all the function related to the inference stage
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from scipy import stats

from statsmodels.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson


def data_inference(mode, final_df, df_test, plot_path):
    """
    Performs inference

    Args:
        - mode (str) : 'univariate' or 'multivariate'
        - final_df (pd.Dataframe): contains the all the data
        in the final form
        - df_test (pd.Dataframe): contains the testing data
        - plot_path (str): path of the inference plot

    Returns: None
    """
    if mode == 'univariate':
        forecast_1 = prophet_inference(df=final_df,
                                       forecasting_period=21)
        forecast_2 = forecast_1
    else:
        forecast_1 = prophet_inference(df=final_df,
                                       forecasting_period=36)
        forecast_2 = prophet_inference(df=final_df,
                                       forecasting_period=21)

    # print(forecast)

    mask = ((forecast_1['ds'] > "2022-04-29") &
            (forecast_1['ds'] < "2022-06-01"))
    forecast_test = forecast_1.loc[mask]

    concat_train_test = pd.concat(
        [final_df['Adj Close'], df_test['Adj Close']])

    plot_forecast(forecast_1,
                  concat_train_test,
                  plot_path)

    #print(forecast_test, df_test['Adj Close'])
    # plot_forecast(forecast_test,
    #               df_test['Adj Close'],
    #               plot_path)

    # y_true = concat_train_test
    # y_pred = forecast['yhat']
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)

    y_true = df_test['Adj Close']
    #y_pred = forecast_test['yhat']
    y_true = np.array(y_true)
    #y_pred = np.array(y_pred)

    calculate_metrics(forecast=forecast_test,
                      y_true=y_true)

    y_true = concat_train_test
    y_pred = forecast_2['yhat']
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    residuals = np.subtract(y_true, y_pred)
    plot_residuals(residuals=residuals,
                   plot_path=plot_path)


def arima_inference(df, forecasting_period):
    pass


def prophet_inference(df, forecasting_period):
    """
    Performs data inference using Meta Prophet model.

    Args:
        - df (pd.Dataframe): final dataframe containing all the data
        - forecasting_period (int): number of days we want to predict

    Returns: None
    """
    prophet_model = Prophet(interval_width=0.99)

    df["Date"] = df.index
    df = df.rename(columns={"Date": "ds", "Adj Close": "y"})

    for col in df.columns:
        if col != 'ds' and col != 'y':
            # print(col)
            prophet_model.add_regressor(name=col)

    prophet_model.fit(df)
    future = prophet_model.make_future_dataframe(
        periods=forecasting_period,
        include_history=True)

    for col in df.columns:
        if col != 'ds' and col != 'y':
            prophet_model_aux = Prophet(interval_width=0.99)
            df_aux = df[['ds', col]]
            df_aux = df_aux.rename(columns={col: "y"})
            prophet_model_aux.fit(df_aux)
            # future = prophet_model.make_future_dataframe(
            #     periods=forecasting_period,
            #     include_history=True)
            forecast = prophet_model_aux.predict(future)
            future[col] = forecast['yhat']

    forecast = prophet_model.predict(future)

    return forecast


def plot_forecast(forecast, y_true, plot_path):
    plt.figure(figsize=(10, 7))
    sns.lineplot(x=y_true.index, y=y_true, label="Ground truth", color='red')
    sns.lineplot(x=forecast['ds'], y=forecast['yhat'],
                 label="Forecast", color='blue')
    # sns.lineplot(x=forecast['ds'], y=forecast['yhat_upper'])
    # sns.lineplot(x=forecast['ds'], y=forecast['yhat_lower'])
    plt.fill_between(forecast['ds'],
                     forecast['yhat_upper'],
                     forecast['yhat_lower'],
                     color='blue',
                     alpha=0.17,
                     label="0.99 confidence interval")
    plt.legend()
    plt.savefig(plot_path)


def calculate_metrics(forecast, y_true):
    """
    Calculates and prints the evaluation metrics.

    Args:
        - forecast (pd.dataframe): contains all the variables of
         the forecast of the model
         - y_true (pd.Series): ground truth values

    Returns: None
    """

    y_pred = forecast['yhat']
    #print(y_pred, y_true)
    errors = y_true - y_pred

    mse = np.mean(errors ** 2)
    mae = np.mean(abs(errors))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(y_pred - y_true) / np.abs(y_true))
    smape = np.mean(np.abs(y_pred - y_true) /
                    ((np.abs(y_true)+np.abs(y_pred))/2))
    mpe = np.mean((y_pred - y_true) / y_true)

    logging.info("\nResults:")
    logging.info(f'- MSE: {mse}')
    logging.info(f'- MAE: {mae}')
    logging.info(f'- RMSE: {rmse}')
    logging.info(f'- MAPE: {mape}')
    logging.info(f'- MPE: {mpe}')
    logging.info(f'- SMAPE: {smape}')


def plot_residuals(residuals, plot_path):
    """
    Plots some useful statistics on the residuals

    Args:

    """
    residuals = residuals[1:]

    mean = residuals.mean()
    median = np.median(residuals)
    skew = stats.skew(residuals)
    # DURBIN-WATSON TEST FOR RESIDUALS AUTOCORRELATION
    durbin_watson_stat = durbin_watson(residuals)
    # D'AGOSTINO-PERSON'S TEST (HYPOTHESIS TESTING)
    # (HYPOTHESIS: THE RESIDUALS DISTRIB. IS NORMAL)
    dagostino_pearson_test_p_value = stats.normaltest(residuals)[1]

    logging.info('\nResidual information:')
    logging.info(f'- Mean: {mean}')
    logging.info(f'- Median: {median}')
    logging.info(f'- Skewness: {skew}')
    logging.info(f'- Durbin: {durbin_watson_stat}')
    logging.info(f'- Anderson p-value: {dagostino_pearson_test_p_value}')

    sns.set()
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.3))
    residuals = (residuals - np.nanmean(residuals)) / np.nanstd(residuals)

    residuals_non_missing = residuals[~(np.isnan(residuals))]
    qqplot(residuals_non_missing, line='s', ax=axes[0])
    axes[0].set_title('Normal Q-Q')

    x = np.arange(0, len(residuals), 1)
    #print(residuals, x)
    sns.lineplot(x=x, y=residuals, ax=axes[1])
    axes[1].set_title('Standardized residual')

    kde = stats.gaussian_kde(residuals_non_missing)
    x_lim = (-1.96 * 2, 1.96 * 2)
    x = np.linspace(x_lim[0], x_lim[1])
    axes[2].plot(x, stats.norm.pdf(x), label='Normal (0,1)', lw=2)
    axes[2].plot(x, kde(x), label='Residuals', lw=2)
    axes[2].set_xlim(x_lim)
    axes[2].legend()
    axes[2].set_title('Estimated density')

    plot_acf(residuals, ax=axes[3])
    #plot_pacf(residuals, ax=axes[4], lags=9)
    fig.tight_layout()
    plt.savefig("".join([plot_path, "_residuals"]))
