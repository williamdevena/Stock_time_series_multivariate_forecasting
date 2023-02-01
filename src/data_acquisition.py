"""
This module contains the functions used to acquire data both locally and remotely
"""

import json
import logging
from datetime import datetime

import pandas as pd
import requests
#import talib
import yfinance as yf
#from covid19dh import covid19

from src import auth, costants, data_storing, logging_utilities


def data_acquisition(source="database"):
    """
    Executes the data acquisition part of the project.

    Args:
        - source (str): indicates from which source we want to acquire the data.
        Possible values:
            - 'online': uses APIs or web scraping
            - 'database': acquires from a MongoDB database

    Returns:
        - data (Dict): contains all the data acquired in the data acquisition phase
    """
    logging_utilities.print_name_stage_project("DATA ACQUISITION")

    # acquiring using APIs or web scraping
    if source == "online":
        data = acquire_online()
    elif source == "database":
        data = acquire_from_database()
    else:
        raise ValueError(
            f"The parameter 'source' has an invalid value {source}")

    return data


def acquire_from_database():
    """
    Acquires all the data (covid data, stock data, technical data, ...)
    from the MongoDB database.

    Returns:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)
    """
    database = costants.DATABASE_NAME_RAW
    data = {}

    for collection in costants.COLLECTIONS:
        logging.info(
            f"\n- Acquiring '{collection}' from the MongoDB '{database}' database")
        collection_data = read_mongodb_collection(
            cluster_name=costants.CLUSTER_NAME, database_name=database,
            collection_name=collection
        )
        df_collection_data = pd.DataFrame(list(collection_data))
        df_collection_data = df_collection_data.drop('_id', axis=1)
        df_collection_data['Date'] = pd.to_datetime(df_collection_data['Date'])
        df_collection_data = df_collection_data.set_index('Date')
        data[collection] = df_collection_data

    return data


def acquire_online():
    """
    Acquires all the data (covid data, stock data, technical data, ...)
    from online resources (APIs, web scraping, ...).

    Returns:
        - data (Dict) : contains all the data of the data acquisition stage
        in the form of pd.Dataframe(s)
    """
    data = {}

    data[costants.COLLECTION_STOCK_DATA] = acquire_stock_data_from_yfinance()
    data[costants.COLLECTION_COVID_DATA] = acquire_covid_data_from_github_dataset()

    # THIS HAS BEEN COMMENTED BECAUSE OF REPRODUCIBILITY PROBLEMS
    # (ON THE MACHINE WERE IT WAS DEVELOPED DID NOT GIVE ERRORS)
    # data[costants.COLLECTION_TECHNICAL_INDICATORS] = produce_technical_indicators_data(
    #     df_stock_data=data[costants.COLLECTION_STOCK_DATA])
    data[costants.COLLECTION_ECONOMIC_INDICATORS] = acquire_economic_indicators()

    return data


def acquire_test_data():
    """
    Acquires the stock data for the testing stage.

    Args: None

    Returns:
        - df_test (pd.Dataframe): contains the Adj Close
        variable
    """
    collection_data = read_mongodb_collection(
        cluster_name=costants.CLUSTER_NAME,
        database_name=costants.DATABASE_NAME_RAW,
        collection_name=costants.COLLECTION_TEST_DATA
    )
    df_test = pd.DataFrame(list(collection_data))
    df_test = df_test.drop('_id', axis=1)
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    df_test = df_test.set_index('Date')

    return df_test


def acquire_economic_indicators():
    """
    Acquires data on several economic indicators:
    - PE ratio
    - PS ratio
    - Market cap
    - USD dollars to canadian dollars
    - USD dollars to chinese yuan
    - USD dollars to japanese yen
    - USD dollars to euro
    - USD dollars to british pound
    - CSI (University of Michigan Consumer Sentiment Index)
    - CPI (Consumer Price Index)
    - U.S. Exports
    - U.S. Industrial Production Index
    - Gold price % change (period=1)
    - Oil price % change (period=1)

    Returns:
        - df_economic_indicators (pd.DataFrame): contains all the
        economic indicators indicated above
    """
    logging.info("\n- Acquiring economic indicators")
    df_cpi = acquire_cpi_from_bls_api()
    df_ycharts = scrape_economic_data_from_ycharts()
    # normalize() to set the hour of every day to midnight
    # to avoid problems when joining
    df_cpi.index = df_cpi.index.normalize()
    df_ycharts.index = df_ycharts.index.normalize()
    df_economic_indicators = df_ycharts.join(df_cpi, how='outer')

    return df_economic_indicators


def acquire_cpi_from_bls_api():
    """
    Uses BLS (U.S. Bureau of Labor Statistics) API to acquire data
    on U.S. CPI (Consumer Price Index).

    Returns:
        - df_cpi (pd.DataFrame): contains the CPI data
    """
    headers = {'Content-type': 'application/json'}
    data = json.dumps(
        {"seriesid": ['CUSR0000SA0'], "startyear": "2017", "endyear": "2022", "registrationkey": auth.BLS_API_KEY})
    p = requests.post(
        'https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)

    # cpi_dict has the following form date: value
    cpi_dict = {datetime.strptime("".join([value['year'], "-", value['period'].split(
        "M")[1]]), '%Y-%m'): float(value['value']) for value in json_data['Results']['series'][0]['data']}
    cpi = pd.Series(data=cpi_dict, name='cpi')
    df_cpi = cpi.to_frame()

    return df_cpi


def scrape_economic_data_from_ycharts():
    """
    Scrapes various types of economic data (fundamental indicators
    and macroeconomic indicators) from the website ycharts.com

    Args: None

    Returns:
        - df_economic_indicators (pandas.core.frame.DataFrame): contains
        the economic indicators data
    """
    logging.info("\n- Scraping economic indicators data from ycharts.com")
    dict_economic_indicators = {}
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    }

    for data_name, url in costants.DICT_SCRAPING_DATA.items():
        raw_data = extract_raw_data_from_scraped_ycharts_data(
            url=url, headers=headers)
        dict_economic_indicators[data_name] = raw_data

    # WE WANT THE CHANGE RATE AND NOT THE RAW VALUE
    dict_economic_indicators['gold'] = dict_economic_indicators['gold'].pct_change(
        periods=1)
    dict_economic_indicators['oil'] = dict_economic_indicators['oil'].pct_change(
        periods=1)

    df_economic_indicators = pd.DataFrame(data=dict_economic_indicators)

    return df_economic_indicators


def extract_raw_data_from_scraped_ycharts_data(url, headers):
    """
    Extract the raw data from a scraped json object

    Args:
        url (str): url we want to use to scrape data

    Returns:
        - raw_data (pd.Series): contains the raw data associated
        with the corresponding dates
    """
    response = requests.get(url=url, headers=headers)
    data = response.json()
    # print(data['start_date'])
    # print(datetime.fromtimestamp(
    #     data['chart_data'][0][0]['raw_data'][-1][0]/1000))
    data = data['chart_data'][0][0]['raw_data']
    raw_data = list(map(transform_dates_timestamps, data))
    raw_data = {date: value for date, value in raw_data}
    raw_data = pd.Series(data=raw_data)
    # print(len(raw_data))

    return raw_data


def transform_dates_timestamps(x):
    """
    Trasforms millisecond timestamps

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return [datetime.fromtimestamp(x[0]/1000), x[1]]


# def produce_technical_indicators_data(df_stock_data):
#     """
#     Produces a dataframe that contains several advanced technical
#     indicators (like SMA, RSI, EMA, ...) from the dataframe with the
#     basic technical indicators ('close', 'open', 'high', ...).

#     Args:
#         - df_stock_data (pandas.core.frame.DataFrame): contains the basic technical indicators

#     Returns:
#         - df_techical_indicators (pandas.core.frame.DataFrame): contains the calculated advanced
#         technical indicators
#     """
#     logging.info(
#         f"\n- Producing TECHNICAL INDICATORS DATA")
#     close = df_stock_data['Close']
#     low = df_stock_data['Low']
#     high = df_stock_data['High']

#     dict_techical_indicators = {}
#     dict_techical_indicators['sma'] = talib.SMA(close)
#     dict_techical_indicators['willr'] = talib.WILLR(
#         high, low, close, timeperiod=14)
#     dict_techical_indicators['rsi'] = talib.RSI(close, timeperiod=14)
#     dict_techical_indicators['ema'] = talib.EMA(close, timeperiod=30)
#     macd_result = talib.MACD(close, fastperiod=12,
#                              slowperiod=26, signalperiod=9)
#     dict_techical_indicators['macd'] = macd_result[0]
#     dict_techical_indicators['macd_signal'] = macd_result[1]
#     dict_techical_indicators['macd_hist'] = macd_result[2]

#     dict_techical_indicators['return_1'] = close.pct_change(periods=1)
#     dict_techical_indicators['return_10'] = close.pct_change(periods=10)
#     dict_techical_indicators['return_20'] = close.pct_change(periods=20)

#     df_techical_indicators = pd.DataFrame(data=dict_techical_indicators)

#     return df_techical_indicators


def acquire_covid_data_from_github_dataset():
    """
    Acquires the covid data from "datasets/covid19" github repository.

    Returns:
        - df_covid_data (pandas.core.frame.DataFrame) : the covid data
    """
    logging.info(
        f"\n- Acquiring COVID DATA using \"datasets/covid19\" github repository")

    df_covid_data = pd.read_csv(
        'https://github.com/datasets/covid-19/blob/main/data/worldwide-aggregate.csv?raw=true')
    df_covid_data['New'] = df_covid_data['Confirmed'].diff(periods=1)
    df_covid_data['Daily Deaths'] = df_covid_data['Deaths'].diff(periods=1)
    df_covid_data['Date'] = pd.to_datetime(df_covid_data['Date'])
    df_covid_data = df_covid_data.set_index('Date')

    return df_covid_data


# OLD DO NOT USE
# def acquire_covid_data_from_covid19dh():
#     """
#     Acquires the covid data using covid19dh API.

#     Returns:
#         - df_covid_data (pandas.core.frame.DataFrame) : the covid data
#     """
#     logging.info(
#         f"\n- Acquiring COVID DATA using covid19dh API")
#     df_covid_data, src = covid19(country="USA", raw=False, verbose=False)
#     variables_to_remove = [
#         col for col in df_covid_data.columns if col not in costants.COVID_VARIABLES]
#     df_covid_data = df_covid_data.drop(variables_to_remove, axis=1)
#     # for consistency through the different types of data
#     df_covid_data = df_covid_data.rename(columns={'date': 'Date'})
#     df_covid_data = df_covid_data.set_index('Date')

#     return df_covid_data


def acquire_covid_data_from_database():
    """
    Acquires the covid data from the MongoDB database.

    Returns:
        - df_covid_data (pandas.core.frame.DataFrame) : the covid data
    """
    database = costants.DATABASE_NAME_RAW
    logging.info(
        f"\n- Acquiring COVID DATA from the MongoDB '{database}' database")
    covid_data = read_mongodb_collection(
        cluster_name=costants.CLUSTER_NAME, database_name=database,
        collection_name=costants.COLLECTION_COVID_DATA
    )
    df_covid_data = pd.DataFrame(list(covid_data))
    df_covid_data = df_covid_data.drop('_id', axis=1)
    df_covid_data['Date'] = pd.to_datetime(df_covid_data['Date'])
    #df_covid_data['Date'] = pd.to_datetime(df_covid_data['date'])
    #df_covid_data = df_covid_data.drop('date', axis=1)
    df_covid_data = df_covid_data.set_index('Date')

    return df_covid_data


def acquire_stock_data_from_database():
    """
    Acquires the stock data from a MongoDB database.

    Returns:
        - df_stock_data (pandas.core.frame.DataFrame) : the stock data
    """
    database = costants.DATABASE_NAME_RAW
    logging.info(
        f"\n- Acquiring STOCK DATA from the MongoDB '{database}' database")
    stock_data = read_mongodb_collection(
        cluster_name=costants.CLUSTER_NAME, database_name=database,
        collection_name=costants.COLLECTION_STOCK_DATA
    )
    df_stock_data = pd.DataFrame(list(stock_data))
    df_stock_data = df_stock_data.drop('_id', axis=1)
    df_stock_data['Date'] = pd.to_datetime(df_stock_data['Date'])
    df_stock_data = df_stock_data.set_index('Date')

    return df_stock_data


def acquire_stock_data_from_yfinance():
    """
    Acquires the stock data from the yahoo finance API.

    Returns:
        - df_stock_data (pandas.core.frame.DataFrame) : the stock data
    """
    logging.info("\n- Acquiring stock data using yahoo finance API.\nATTENTION:  The data that you are acquiring could change from the one you acquired before.")
    df_stock_data = download_stock_data_yfinance(
        costants.AAPL, costants.START_DATE, costants.END_DATE)

    return df_stock_data


def download_stock_data_yfinance(company, start_date, end_date):
    """
    Downloads and returns hystorical stock value of a company in a specified
    range.

    Args:
        - company (str): indicates the company of the stocks we want to download
        (in this project we use 'AAPL')
        - start-date (datetime.datetime): first day of the period we want to download
        - end-date (datetime.datetime): last day of the period we want to download

    Returns:
        - df_hystorical_stock_data (pandas.core.frame.DataFrame): contains the closing
        stock value for each day between start_date and end_date

    """
    logging.info(f"- Downloading '{company}' stock data")
    hystorical_stock_data = yf.download(
        company, start=start_date, end=end_date)
    df_hystorical_stock_data = pd.DataFrame(hystorical_stock_data)

    return df_hystorical_stock_data


def read_stock_data_csv():
    """
    Reads the aapl stock csv and returns the corresponding csv where the
    'Date' column has been transformed in a datetime type.

    Args: None

    Returns:
        - stock_df (pandas.core.frame.DataFrame): conatins aapl stock values between April 2017
        and April 2022
    """
    file_path = costants.STOCK_DATA_CSV
    stock_df = pd.read_csv(file_path,
                           sep=costants.SEPARATOR)

    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    logging.info(f"\n- Read stock data from the local file {file_path}")

    return stock_df


def read_mongodb_collection(cluster_name, database_name, collection_name, condition={}):
    """
    Reads from a MongoDB database a certain collection and if given querys with certain conditions.

    Args:
        - cluster_name (str): Name of the MongoDB cluster
        - database_name (str): Name of the MongoDB database
        - collection_name (str): Name of the MongoDB collection
        - condition (dict): Dictionary containing the conditions of the query.
        (EX: condition = {'name' : 'William'} gets all the documents of the collection
        that have 'name'='William')

    Returns:
        - (pymongo.cursor.Cursor): A pymongo Cursor object that is iterable and that
        represents the result of the query.
    """
    client = data_storing.connect_cluster_mongodb(
        cluster_name, auth.MONGODB_USERNAME, auth.MONGODB_PASSWORD)
    database = data_storing.connect_database(client, database_name)
    collection = data_storing.connect_collection(database, collection_name)[0]
    # logging.info(
    #     f"\n- Reading the '{collection_name}' collection in the '{database_name}' database")

    return collection.find(condition)


def main():
    # start_date = datetime(2017, 4, 1)
    # end_date = datetime(2022, 4, 30)

    # data = download_stock_data(costants.AAPL, start_date, end_date)
    # plt.plot(data)
    # plt.show()
    pass


if __name__ == "__main__":
    main()
