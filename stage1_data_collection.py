import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging
from stage1_config import LARGE_CAP_FUNDS, MID_CAP_FUNDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EquityFundDataCollector:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        logger.info(f"Data directory: {self.data_dir}")
    
    def fetch_benchmark_indices(self, start_date='2020-01-01', end_date=None):
        """
        Fetch NIFTY 50 (large-cap) and NIFTY Midcap 150 (mid-cap) indices
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching benchmark indices from {start_date} to {end_date}")
        
        indices = {
            'NIFTY_50': '^NSEI',
            'NIFTY_MIDCAP_150': '^NSMIDCAP'
        }
        
        benchmark_data = {}
        
        for name, ticker in indices.items():
            try:
                logger.info(f"Downloading {name} ({ticker})")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    data['Index'] = name
                    benchmark_data[name] = data
                    logger.info(f"Successfully downloaded {name}: {len(data)} records")
                else:
                    logger.warning(f"No data available for {name} ({ticker})")
            except Exception as e:
                logger.error(f"Error downloading {name}: {str(e)}")
        
        return benchmark_data
    
    def fetch_mutual_fund_nav(self, fund_symbols, start_date='2020-01-01', end_date=None):
        """
        Fetch mutual fund NAV data from Yahoo Finance
        Note: Indian mutual funds may have limited historical data on Yahoo Finance
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching mutual fund NAV data from {start_date} to {end_date}")
        
        fund_data = {}
        
        for symbol in fund_symbols:
            try:
                logger.info(f"Downloading {symbol}")
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    data['Fund'] = symbol
                    fund_data[symbol] = data
                    logger.info(f"Successfully downloaded {symbol}: {len(data)} records")
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
        
        return fund_data
    
    def save_data(self, data_dict, prefix=''):
        """
        Save downloaded data to CSV files
        """
        for name, df in data_dict.items():
            filename = os.path.join(self.data_dir, f"{prefix}_{name}.csv")
            df.to_csv(filename)
            logger.info(f"Saved {name} to {filename}")
    
    def load_data(self, filename):
        """
        Load data from CSV file
        """
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading data from {filepath}")
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            logger.error(f"File not found: {filepath}")
            return None


class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_missing_values(self, df, method='forward_fill'):
        """
        Handle missing values in the dataset
        Methods: 'forward_fill', 'interpolate', 'drop'
        """
        initial_nulls = df.isnull().sum().sum()
        
        if method == 'forward_fill':
            df = df.ffill().bfill()
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        
        final_nulls = df.isnull().sum().sum()
        self.logger.info(f"Missing values: {initial_nulls} -> {final_nulls}")
        
        return df
    
    def calculate_returns(self, df, price_column='Close', periods=[1, 5, 20]):
        """
        Calculate returns for different periods
        """
        for period in periods:
            df[f'Return_{period}D'] = df[price_column].pct_change(periods=period)
        
        self.logger.info(f"Calculated returns for periods: {periods}")
        return df
    
    def calculate_volatility(self, df, price_column='Close', window=20):
        """
        Calculate rolling volatility (standard deviation of returns)
        """
        returns = df[price_column].pct_change()
        df[f'Volatility_{window}D'] = returns.rolling(window=window).std()
        
        self.logger.info(f"Calculated {window}-day rolling volatility")
        return df
    
    def normalize_data(self, df, columns=None):
        """
        Normalize data to 0-1 range
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_normalized = df.copy()
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        
        self.logger.info(f"Normalized {len(columns)} columns")
        return df_normalized
    
    def remove_outliers(self, df, columns=None, threshold=3):
        """
        Remove outliers using z-score method
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(df)
        
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        
        removed_rows = initial_rows - len(df)
        self.logger.info(f"Removed {removed_rows} outlier rows (threshold={threshold})")
        
        return df


def main():
    logger.info("Starting Stage 1: Data Collection & Preprocessing")
    
    collector = EquityFundDataCollector(data_dir='data')
    
    logger.info("\n=== Fetching Benchmark Indices ===")
    benchmark_data = collector.fetch_benchmark_indices(
        start_date='2020-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    collector.save_data(benchmark_data, prefix='benchmark')
    
    logger.info("\n=== Fetching Mutual Fund NAV ===")
    fund_symbols = LARGE_CAP_FUNDS + MID_CAP_FUNDS
    fund_data = collector.fetch_mutual_fund_nav(
        fund_symbols,
        start_date='2020-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    collector.save_data(fund_data, prefix='fund')
    
    logger.info("\n=== Preprocessing Benchmark Data ===")
    preprocessor = DataPreprocessor()
    
    for name, df in benchmark_data.items():
        logger.info(f"Processing {name}")
        df = preprocessor.handle_missing_values(df, method='forward_fill')
        df = preprocessor.calculate_returns(df, price_column='Close')
        df = preprocessor.calculate_volatility(df, price_column='Close', window=20)
        benchmark_data[name] = df
    
    collector.save_data(benchmark_data, prefix='benchmark_processed')
    
    logger.info("\n=== Preprocessing Mutual Fund Data ===")
    for name, df in fund_data.items():
        logger.info(f"Processing {name}")
        df = preprocessor.handle_missing_values(df, method='forward_fill')
        df = preprocessor.calculate_returns(df, price_column='Close')
        df = preprocessor.calculate_volatility(df, price_column='Close', window=20)
        fund_data[name] = df
    
    collector.save_data(fund_data, prefix='fund_processed')
    
    logger.info("\n=== Data Collection & Preprocessing Complete ===")
    logger.info(f"Data saved to {collector.data_dir}/ directory")


if __name__ == '__main__':
    main()
