import os
from datetime import datetime, timedelta

DATA_DIR = 'data'
LOGS_DIR = 'logs'

START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

BENCHMARK_INDICES = {
    'NIFTY_50': '^NSEI',
    'NIFTY_MIDCAP_150': '^NSMIDCAP'
}

LARGE_CAP_FUNDS = [
    'RELIANCE.NS',
    'INFY.NS',
    'TCS.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS',
    'SBIN.NS',
    'MARUTI.NS',
    'BAJAJFINSV.NS',
    'LT.NS',
    'ASIANPAINT.NS'
]

MID_CAP_FUNDS = [
    'PAGEIND.NS',
    'MPHASIS.NS',
    'PERSISTENT.NS',
    'TORNTPHARM.NS',
    'SBICARD.NS',
    'IDFCFIRSTB.NS',
    'INDIGO.NS',
    'APOLLOHOSP.NS',
    'BIOCON.NS',
    'GAIL.NS'
]

PREPROCESSING_CONFIG = {
    'missing_value_method': 'forward_fill',
    'outlier_threshold': 3,
    'volatility_window': 20,
    'return_periods': [1, 5, 20]
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
