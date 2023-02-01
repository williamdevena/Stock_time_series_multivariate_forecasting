"""
This module contains the functions to visualize data in different forms
"""
import logging
import os


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prophet import Prophet
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

from src import costants, logging_utilities

sns.set()


def data_visualization(data):
    """
    Executes the data visualization part of the project.

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    if not os.path.exists(costants.PLOTS_FOLDER):
        os.mkdir(costants.PLOTS_FOLDER)
    logging_utilities.print_name_stage_project("DATA VISUALIZATION")
    logging.info(
        f"\n- Plotting data and saving the plots locally in {costants.PLOTS_FOLDER}")
    save_plot_all_variables_all_data(data=data)
    save_plot_outliers(data=data)
    save_plot_seasonal_decomposition(data=data)
    save_plot_autocorrelation(data=data)
    save_pair_plots(data=data)
    save_correlation_matrix(data=data)


def calculate_z_score_outliers_loc(df_column):
    """
    Calculates and returns the indexes of the outliers, using
    the criteria |z_score| > 3

    Args:
        - df (pd.Series): contains the data

    Returns:
        - outliers_loc (np.ndarray): contains the indexes of the outliers
    """
    threshold = 3

    z = np.abs(stats.zscore(df_column))
    outlier_loc = np.where(z > threshold)

    return outlier_loc


def save_plot_all_variables_all_data(data):
    """
    Creates and saves a plot for every column in a df

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    logging.info("\n- Plotting single variables")
    for df_name, df in data.items():
        #df = df.fillna(value=0)
        #df = df.interpolate(method='time')
        plot_folder_path = os.path.join(costants.PLOTS_FOLDER, df_name)

        if not os.path.exists(plot_folder_path):
            os.mkdir(plot_folder_path)

        for column in df.keys():
            plot_path = os.path.join(plot_folder_path, column)

            fig, axs = plt.subplots(2, figsize=(15, 15))
            # fig.tight_layout()
            plt.subplots_adjust(hspace=0.5)

            # print(column)

            axs[0].plot(df.index, df[column])
            axs[0].set_title(f"{column} Time series")

            sns.kdeplot(data=df[column], ax=axs[1], color='r')
            #axs[1].set_xlim((df["value"].min(), df["value"].max()))
            axs1_2 = axs[1].twinx()
            sns.histplot(data=df[column], ax=axs1_2,)
            # axs[1].grid()
            axs1_2.grid(False)
            axs[1].set_title(f"{column} Distribution histogram and KDE")

            plt.savefig(plot_path)
            plt.close()


def save_plot_outliers(data):
    """
    Creates and saves several kind of plots to highlight
    the outliers detected using several methods.

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    logging.info("- Plotting outliers")
    for df_name, df in data.items():
        #df = df.fillna(value=0)
        #df = df.interpolate(method='time')
        plot_folder_path = os.path.join(costants.PLOTS_FOLDER, df_name)

        if not os.path.exists(plot_folder_path):
            os.mkdir(plot_folder_path)

        for column in df.keys():
            plot_path = os.path.join(
                plot_folder_path, "".join([column, "_outliers"]))

            fig, axs = plt.subplots(2, figsize=(20, 20))
            # fig.tight_layout()
            plt.subplots_adjust(hspace=0.5)

            # outliers detection using boxplot (IQR method)
            sns.boxplot(data=df[column], ax=axs[0], orient='h')
            axs[0].set_title(
                f"{column} Box plot (outlier detection with IQR method)")

            # outliers detection using z-score
            outlier_loc = calculate_z_score_outliers_loc(
                df_column=df[column])[0]
            axs[1].scatter(x=df[column].index, y=df[column])
            axs[1].scatter(x=df[column].index[outlier_loc],
                           y=df[column][outlier_loc], c='r')
            axs[1].set_title(
                f"{column} outlier detection with z-score (the outliers are highlighted in red)")

            # plt.plot(df.index, df[column])
            # plt.title(column)
            plt.savefig(plot_path)
            plt.close()

            # logger = logging.getLogger()
            # logger.disabled = True

            # plot_prophet_anomaly_detection(data=data,
            #                                 data_name=df_name,
            #                                 variable_name=column,
            #                                 plot_path="".join([plot_path, "_prophet"]))

            #logger.disabled = False


def plot_prophet_anomaly_detection(data, data_name, variable_name, plot_path):
    """
    Performs anomaly detection (also called outlier detection) using
    the Meta Prophet model (https://facebook.github.io/prophet/) and
    plots the result.

    Args:
        - data (Dict): contains all the data to check
        - data_name (str): represents the name of the type
        of data (e.g. "covid_data")
        - variable_name: name of the variable on which we want
        to perform
        - plot_path (str): path of the saved plot image

     Returns: None
    """
    prophet_model = Prophet(interval_width=0.99,
                            # yearly_seasonality=True, weekly_seasonality=True
                            )

    #f, ax = plt.subplots()

    df = data[data_name].copy()
    df["Date"] = df.index
    df = df[["Date", variable_name]]
    df = df.rename(columns={"Date": "ds", variable_name: "y"})

    prophet_model.fit(df)
    forecast = prophet_model.predict(df)
    # prophet_model.plot(forecast)

    # performance = pd.merge(
    #     df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    # # Create an anomaly indicator
    # performance['anomaly'] = performance.apply(lambda rows: 1 if (
    #     (rows.y < rows.yhat_lower) | (rows.y > rows.yhat_upper)) else 0, axis=1)

    # # Visualize the anomalies
    # sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly', ax=ax)
    # sns.lineplot(x='ds', y='yhat', data=performance, color='black', ax=ax)
    # plt.fill_between(performance['yhat'], performance['yhat_upper'], performance['yhat_lower'])

    prophet_model.plot(forecast)
    plt.savefig(plot_path)
    # return ax


def save_plot_seasonal_decomposition(data):
    """
    Creates and saves a seasonal decomposition plot for
    every type of data. The seasonal decomposition plot
    is composed of three plots (season, trend and residual).

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    logging.info("- Plotting seasonal decomposition")
    for df_name, df in data.items():
        #df = df.fillna(value=0)
        #df = df.interpolate(method='time')
        plot_folder_path = os.path.join(costants.PLOTS_FOLDER, df_name)

        if not os.path.exists(plot_folder_path):
            os.mkdir(plot_folder_path)

        for column in df.keys():
            plot_path = os.path.join(plot_folder_path, "".join(
                [column, "_seasonal_decomposition"]))
            result = seasonal_decompose(df[column])
            #seasonal_adjusted = result.trend.add(result.resid)

            fig, axs = plt.subplots(4, figsize=(20, 20))
            # fig.tight_layout()
            plt.subplots_adjust(hspace=0.5)
            axs[0].plot(df.index, result.observed)
            axs[0].set_title(f"{column}")
            # axs[1].plot(df.index, seasonal_adjusted)
            # axs[1].set_title(f"Seasonal adjusted")
            axs[1].plot(df.index, result.trend)
            axs[1].set_title("Trend")
            axs[2].plot(df.index, result.seasonal)
            axs[2].set_title("Season")
            axs[3].plot(df.index, result.resid)
            axs[3].set_title(" Residual")

            plt.savefig(plot_path)
            plt.close()


def save_plot_autocorrelation(data):
    """
    Creates and saves autocorrelation plot for
    every type of data.

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    logging.info("- Plotting autocorrelation")
    for df_name, df in data.items():
        #df = df.fillna(value=0)
        #df = df.interpolate(method='time')
        plot_folder_path = os.path.join(costants.PLOTS_FOLDER, df_name)

        if not os.path.exists(plot_folder_path):
            os.mkdir(plot_folder_path)

        for column in df.keys():
            plot_path = os.path.join(
                plot_folder_path, "".join([column, "_autocorrelation"]))
            plot_acf(df[column])
            plt.title(f"{column} autocorrelation")
            plt.savefig(plot_path)
            plt.close()


def save_pair_plots(data):
    """
    Creates and saves a pair plot for every type of data
    in combination with the target variable Adj Close

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    logging.info("- Plotting pair plots")
    adj_close = data['stock_data']['Adj Close']
    for df_name, df in data.items():
        df_copy = df.copy()
        df_copy['Adj Close'] = adj_close
        plot_path = os.path.join(costants.PLOTS_FOLDER, df_name, "pair_plot")
        sns.pairplot(df_copy)
        plt.savefig(plot_path)
        plt.close()


def save_correlation_matrix(data):
    """
    Creates and saves a correlation matrix for every type of data
    in combination with the target variable Adj Close

    Args:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)

    Returns: None
    """
    adj_close = data['stock_data']['Adj Close']
    logging.info("- Plotting correlation matrices")
    for df_name, df in data.items():
        df_copy = df.copy()
        df_copy['Adj Close'] = adj_close
        plt.figure(figsize=(8, 8))
        plot_path = os.path.join(
            costants.PLOTS_FOLDER, df_name, "correlation_matrix")
        heatmap = sns.heatmap(df_copy.corr(), annot=True)
        heatmap.set_title("Correlation matrix")
        plt.savefig(plot_path)
        plt.close()


# def save_correlation_matrix_with_adj_close(data):
#     """
#     Creates and saves a correlation matrix for every type of
#     data in combination with the adjusted closing price
#     variable of the stock data (the target variable).

#     Args:
#         - data (Dict) : contains all the data of the data acquisition stage
#         in the form of pd.Dataframe(s)

#     Returns: None
#     """
#     logging.info("- Plotting correlation matrices with Adj Close variable")
#     #print(data['stock_data'].columns)
#     adj_close = data['stock_data']['Adj Close']
#     for df_name, df in data.items():
#         joined_data = adj_close.merge(df, how='left')


#         plt.figure(figsize=(8, 8))
#         plot_path = os.path.join(costants.PLOTS_FOLDER, df_name, "correlation_matrix_with_adj_close")
#         heatmap = sns.heatmap(joined_data.corr(), annot=True)
#         heatmap.set_title("Correlation matrix")
#         plt.savefig(plot_path)
#         plt.close()


# OLD DO NOT USE
# def save_pair_plots_stock(start_date, end_date):
#     """
#     Plots and save the pair plot of the stock data in a
#     certain period.

#     Args:
#         - start_date (datetime.date): first day of the period we
#         want to plot
#         - end_date (datetime.date): last day of the period we
#         want to plot

#     Returns: None
#     """
#     stock_df = data_acquisition.read_stock_data_csv()
#     mask = (stock_df['Date'] > start_date) & (stock_df['Date'] < end_date)
#     stock_df = stock_df.loc[mask]

#     sns.pairplot(stock_df)

#     plot_path = "".join(["pairs_plot_",
#                          str(start_date.year), "_", str(start_date.month), "_", str(start_date.day),
#                          "_", "_",
#                          str(end_date.year), "_", str(end_date.month), "_", str(end_date.day),
#                          ".png"])
#     plot_path = os.path.join(costants.PLOTS_FOLDER, plot_path)
#     plt.savefig(plot_path)


# def save_plot_all_variables_stock(start_date, end_date):
#     """
#     Plots and save all the AAPL stock variables ('Open', 'Close', 'Low', 'High',
#     'Volume', 'Adj Close') values in a certain period.

#     Args:
#         - start_date (datetime.date): first day of the period we
#         want to plot
#         - end_date (datetime.date): last day of the period we
#         want to plot

#     Returns: None

#     """
#     stock_df = data_acquisition.read_stock_data_csv()
#     mask = (stock_df['Date'] > start_date) & (stock_df['Date'] < end_date)
#     stock_df = stock_df.loc[mask]

#     fig, axs = plt.subplots(3, 2, figsize=(25, 15))
#     axs[0, 0].plot(stock_df['Date'], stock_df['Open'])
#     axs[0, 0].set_title('Open')
#     axs[0, 1].plot(stock_df['Date'], stock_df['Close'])
#     axs[0, 1].set_title('Close')
#     axs[1, 0].plot(stock_df['Date'], stock_df['High'])
#     axs[1, 0].set_title('High')
#     axs[1, 1].plot(stock_df['Date'], stock_df['Low'])
#     axs[1, 1].set_title('Low')
#     axs[2, 0].plot(stock_df['Date'], stock_df['Volume'])
#     axs[2, 0].set_title('Volume')
#     axs[2, 1].plot(stock_df['Date'], stock_df['Adj Close'])
#     axs[2, 1].set_title('Adj Close')

#     plot_path = "".join(["all_features_",
#                          str(start_date.year), "_", str(start_date.month), "_", str(start_date.day),
#                          "_", "_",
#                          str(end_date.year), "_", str(end_date.month), "_", str(end_date.day),
#                          ".png"])
#     plot_path = os.path.join(costants.PLOTS_FOLDER, plot_path)
#     plt.savefig(plot_path)
