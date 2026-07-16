"""
Combined Equity Fund Risk Analysis
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import logging
import time
import json
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logger.warning("statsmodels not installed. ARIMA models will be skipped.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logger.warning("TensorFlow not installed. LSTM models will be skipped.")

# Configuration
STRESS_TEST_SCENARIOS = {
    'Market_Crash_10pct': -0.10,
    'Market_Crash_20pct': -0.20,
    'Market_Crash_30pct': -0.30,
    'Market_Rally_10pct': 0.10,
    'Market_Rally_20pct': 0.20,
    'Volatility_Spike_2x': 2.0,
    'Volatility_Spike_3x': 3.0
}

# =============================================================================
# STAGE 1: DATA COLLECTION & PREPROCESSING
# =============================================================================

class SyntheticDataGenerator:
    """Generate realistic synthetic market data based on historical patterns."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
    
    def generate_index_data(self, name, start_date, end_date, base_price, 
                            annual_return=0.12, annual_volatility=0.18):
        """Generate realistic index price data using Geometric Brownian Motion."""
        self.logger.info(f"Generating synthetic data for {name}")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        random_returns = np.random.normal(daily_return, daily_volatility, n_days)
        regime_changes = np.random.choice([0, 1], size=n_days, p=[0.95, 0.05])
        random_returns = random_returns * (1 + regime_changes * np.random.uniform(0.5, 2.0, n_days))
        
        price_multipliers = np.exp(np.cumsum(random_returns))
        close_prices = base_price * price_multipliers
        
        daily_range = daily_volatility * close_prices
        high_prices = close_prices + np.abs(np.random.normal(0, 1, n_days)) * daily_range * 0.5
        low_prices = close_prices - np.abs(np.random.normal(0, 1, n_days)) * daily_range * 0.5
        open_prices = low_prices + np.random.uniform(0.3, 0.7, n_days) * (high_prices - low_prices)
        
        base_volume = 1000000 if 'MIDCAP' in name else 2000000
        volume_volatility = np.abs(random_returns) / daily_volatility
        volumes = base_volume * (1 + volume_volatility) * np.random.uniform(0.8, 1.2, n_days)
        
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes.astype(int),
            'Index': name
        }, index=dates)
        
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        self.logger.info(f"Generated {len(df)} records for {name}")
        return df


class EquityFundDataCollector:
    """Data collector with multiple source options."""
    
    def __init__(self, data_dir='data', alpha_vantage_key=None):
        self.data_dir = data_dir
        self.alpha_vantage_key = alpha_vantage_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.synthetic_generator = SyntheticDataGenerator()
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        logger.info(f"Data directory: {self.data_dir}")
    
    def fetch_from_alpha_vantage(self, symbol, name):
        """Fetch data from Alpha Vantage API."""
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not set.")
            return None
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'compact',
                'apikey': self.alpha_vantage_key
            }
            
            logger.info(f"Fetching {name} from Alpha Vantage...")
            response = requests.get(url, params=params, timeout=30, verify=False)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                ts_data = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                df.columns = [col.split('. ')[1].capitalize() if '. ' in col else col for col in df.columns]
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df['Index'] = name
                logger.info(f"Successfully fetched {name} from Alpha Vantage: {len(df)} records")
                return df
            else:
                logger.warning(f"Unexpected response for {name}")
                return None
        except Exception as e:
            logger.error(f"Alpha Vantage error for {name}: {str(e)}")
            return None
    
    def fetch_benchmark_indices(self, start_date='2020-01-01', end_date=None):
        """Fetch NIFTY 50 and NIFTY Midcap 150 indices."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching benchmark indices from {start_date} to {end_date}")
        
        indices_config = {
            'NIFTY_50': {
                'alpha_vantage_symbol': 'SPY',
                'base_price': 17000,
                'annual_return': 0.12,
                'annual_volatility': 0.16
            }
        }
        
        benchmark_data = {}
        
        for name, config in indices_config.items():
            df = None
            
            if self.alpha_vantage_key:
                df = self.fetch_from_alpha_vantage(config['alpha_vantage_symbol'], name)
                if df is not None:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                time.sleep(12)
            
            if df is None or len(df) == 0:
                logger.info(f"Using synthetic data for {name}")
                df = self.synthetic_generator.generate_index_data(
                    name=name,
                    start_date=start_date,
                    end_date=end_date,
                    base_price=config['base_price'],
                    annual_return=config['annual_return'],
                    annual_volatility=config['annual_volatility']
                )
            
            benchmark_data[name] = df
        
        return benchmark_data
    
    def save_data(self, data_dict, prefix=''):
        """Save downloaded data to CSV files"""
        for name, df in data_dict.items():
            filename = os.path.join(self.data_dir, f"{prefix}_{name}.csv")
            df.to_csv(filename)
            logger.info(f"Saved {name} to {filename}")


class DataPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_missing_values(self, df, method='forward_fill'):
        """Handle missing values in the dataset"""
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
        """Calculate returns for different periods"""
        for period in periods:
            df[f'Return_{period}D'] = df[price_column].pct_change(periods=period)
        self.logger.info(f"Calculated returns for periods: {periods}")
        return df
    
    def calculate_volatility(self, df, price_column='Close', window=20):
        """Calculate rolling volatility"""
        returns = df[price_column].pct_change()
        df[f'Volatility_{window}D'] = returns.rolling(window=window).std()
        self.logger.info(f"Calculated {window}-day rolling volatility")
        return df


# =============================================================================
# STAGE 2: PREDICTIVE MODELING
# =============================================================================

class ARIMAForecaster:
    def __init__(self, order=(5, 1, 2)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, data, column='Close'):
        """Fit ARIMA model to time series data"""
        if not ARIMA_AVAILABLE:
            self.logger.error("ARIMA not available. Install statsmodels: pip install statsmodels")
            return False
        
        try:
            self.logger.info(f"Fitting ARIMA{self.order} model on {column}")
            self.model = ARIMA(data[column].dropna(), order=self.order)
            self.fitted_model = self.model.fit()
            self.logger.info(f"ARIMA model fitted successfully")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}, BIC: {self.fitted_model.bic:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Error fitting ARIMA model: {str(e)}")
            return False
    
    def forecast(self, steps=20):
        """Generate forecast for specified number of steps"""
        if self.fitted_model is None:
            self.logger.error("Model not fitted. Call fit() first.")
            return None
        
        try:
            self.logger.info(f"Forecasting {steps} steps ahead")
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast_df = forecast_result.summary_frame()
            self.logger.info(f"Forecast generated successfully")
            return forecast_df
        except Exception as e:
            self.logger.error(f"Error generating forecast: {str(e)}")
            return None


class LSTMForecaster:
    def __init__(self, lookback=60, epochs=50, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.logger = logging.getLogger(__name__)
    
    def create_sequences(self, data, lookback):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def prepare_data(self, data, column='Close', test_size=0.2):
        """Prepare data for LSTM training"""
        try:
            self.logger.info(f"Preparing data for LSTM with lookback={self.lookback}")
            
            values = data[column].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(values)
            
            X, y = self.create_sequences(scaled_data, self.lookback)
            
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            self.logger.info(f"Data prepared: Train={X_train.shape}, Test={X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None, None, None, None
    
    def build_model(self, input_shape):
        """Build LSTM neural network"""
        if not LSTM_AVAILABLE:
            self.logger.error("TensorFlow not available. Install: pip install tensorflow")
            return False
        
        try:
            self.logger.info("Building LSTM model")
            self.model = Sequential([
                LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu', return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.logger.info("LSTM model built successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            return False
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model"""
        if self.model is None:
            self.logger.error("Model not built. Call build_model() first.")
            return False
        
        try:
            self.logger.info(f"Training LSTM model for {self.epochs} epochs")
            
            callbacks = [EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]
            
            if X_val is not None and y_val is not None:
                self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                              validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
            else:
                self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                              callbacks=callbacks, verbose=0)
            
            self.logger.info("LSTM model trained successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict(self, X_test):
        """Generate predictions on test data"""
        if self.model is None:
            self.logger.error("Model not trained. Call train() first.")
            return None
        
        try:
            predictions = self.model.predict(X_test, verbose=0)
            predictions_rescaled = self.scaler.inverse_transform(predictions)
            return predictions_rescaled
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return None


# =============================================================================
# STAGE 3: RISK ANALYSIS
# =============================================================================

class RiskMetricsCalculator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk (VaR) using historical method"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        self.logger.info(f"VaR ({confidence_level*100}%): {var:.4f}")
        return var
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        """Calculate Expected Shortfall (ES) / Conditional VaR"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        es = returns[returns <= var].mean()
        self.logger.info(f"Expected Shortfall ({confidence_level*100}%): {var:.4f}")
        return es
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.05):
        """Calculate Sharpe Ratio"""
        excess_return = returns.mean() - risk_free_rate / 252
        volatility = returns.std()
        
        if volatility == 0:
            sharpe = 0
        else:
            sharpe = excess_return / volatility
        
        self.logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        return sharpe
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.05):
        """Calculate Sortino Ratio"""
        excess_return = returns.mean() - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std()
        
        if downside_volatility == 0:
            sortino = 0
        else:
            sortino = excess_return / downside_volatility
        
        self.logger.info(f"Sortino Ratio: {sortino:.4f}")
        return sortino
    
    def calculate_max_drawdown(self, prices):
        """Calculate Maximum Drawdown"""
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        self.logger.info(f"Maximum Drawdown: {max_drawdown:.4f}")
        return max_drawdown
    
    def calculate_volatility(self, returns, periods=252):
        """Calculate annualized volatility"""
        volatility = returns.std() * np.sqrt(periods)
        self.logger.info(f"Annualized Volatility: {volatility:.4f}")
        return volatility


class ExPostRiskAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_volatility_prediction_error(self, actual_volatility, predicted_volatility):
        """Analyze how well volatility was predicted"""
        self.logger.info("Analyzing volatility prediction accuracy")
        
        vol_error = actual_volatility - predicted_volatility
        vol_underestimation = (vol_error > 0).sum() / len(vol_error) * 100
        vol_overestimation = (vol_error < 0).sum() / len(vol_error) * 100
        
        analysis = pd.DataFrame({
            'Actual_Volatility': actual_volatility,
            'Predicted_Volatility': predicted_volatility,
            'Volatility_Error': vol_error,
            'Underestimated': vol_error > 0,
            'Overestimated': vol_error < 0
        })
        
        self.logger.info(f"  Volatility Underestimated: {vol_underestimation:.2f}%")
        self.logger.info(f"  Volatility Overestimated: {vol_overestimation:.2f}%")
        
        return analysis


class StressTestAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply_scenario(self, prices, scenario_type, magnitude):
        """Apply stress scenario to prices"""
        stressed_prices = prices.copy()
        
        if scenario_type == 'crash' or scenario_type == 'rally':
            stressed_prices = prices * (1 + magnitude)
        elif scenario_type == 'volatility':
            returns = prices.pct_change()
            stressed_returns = returns * magnitude
            stressed_prices = prices.iloc[0] * (1 + stressed_returns).cumprod()
        
        return stressed_prices
    
    def calculate_scenario_impact(self, prices, scenario_type, magnitude):
        """Calculate impact of stress scenario"""
        stressed_prices = self.apply_scenario(prices, scenario_type, magnitude)
        
        original_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        stressed_return = (stressed_prices.iloc[-1] - stressed_prices.iloc[0]) / stressed_prices.iloc[0]
        
        original_drawdown = ((prices - prices.cummax()) / prices.cummax()).min()
        stressed_drawdown = ((stressed_prices - stressed_prices.cummax()) / stressed_prices.cummax()).min()
        
        impact = {
            'Original_Return': original_return,
            'Stressed_Return': stressed_return,
            'Return_Impact': stressed_return - original_return,
            'Original_Drawdown': original_drawdown,
            'Stressed_Drawdown': stressed_drawdown,
            'Drawdown_Impact': stressed_drawdown - original_drawdown
        }
        
        return impact
    
    def generate_stress_test_report(self, large_cap_prices, midcap_prices, scenarios):
        """Generate comprehensive stress test report"""
        self.logger.info("Generating Stress Test Report")
        
        results = []
        
        for scenario_name, magnitude in scenarios.items():
            if 'Crash' in scenario_name:
                scenario_type = 'crash'
            elif 'Rally' in scenario_name:
                scenario_type = 'rally'
            else:
                scenario_type = 'volatility'
            
            lc_impact = self.calculate_scenario_impact(large_cap_prices, scenario_type, magnitude)
            mc_impact = self.calculate_scenario_impact(midcap_prices, scenario_type, magnitude)
            
            results.append({
                'Scenario': scenario_name,
                'LC_Return_Impact': lc_impact['Return_Impact'],
                'MC_Return_Impact': mc_impact['Return_Impact'],
                'LC_Drawdown_Impact': lc_impact['Drawdown_Impact'],
                'MC_Drawdown_Impact': mc_impact['Drawdown_Impact'],
                'LC_More_Resilient': abs(lc_impact['Return_Impact']) < abs(mc_impact['Return_Impact'])
            })
        
        return pd.DataFrame(results)


class BetaAlphaCalculator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_beta(self, fund_returns, benchmark_returns, risk_free_rate=0.05):
        """Calculate Beta (systematic risk)"""
        min_len = min(len(fund_returns), len(benchmark_returns))
        fund_returns = fund_returns.iloc[:min_len]
        benchmark_returns = benchmark_returns.iloc[:min_len]
        
        covariance = np.cov(fund_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            beta = 0
        else:
            beta = covariance / benchmark_variance
        
        self.logger.info(f"Beta: {beta:.4f}")
        return beta
    
    def calculate_alpha(self, fund_returns, benchmark_returns, risk_free_rate=0.05):
        """Calculate Alpha (excess return)"""
        beta = self.calculate_beta(fund_returns, benchmark_returns, risk_free_rate)
        
        fund_annual_return = fund_returns.mean() * 252
        benchmark_annual_return = benchmark_returns.mean() * 252
        
        alpha = fund_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        self.logger.info(f"Alpha: {alpha:.4f}")
        return alpha
    
    def calculate_information_ratio(self, fund_returns, benchmark_returns):
        """Calculate Information Ratio"""
        min_len = min(len(fund_returns), len(benchmark_returns))
        fund_returns = fund_returns.iloc[:min_len]
        benchmark_returns = benchmark_returns.iloc[:min_len]
        
        active_return = fund_returns.mean() - benchmark_returns.mean()
        tracking_error = (fund_returns - benchmark_returns).std()
        
        if tracking_error == 0:
            ir = 0
        else:
            ir = active_return / tracking_error
        
        self.logger.info(f"Information Ratio: {ir:.4f}")
        return ir


class ComparativeFundAnalysis:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_large_cap_vs_midcap(self, large_cap_data, midcap_data):
        """Compare risk metrics between large-cap and mid-cap funds"""
        self.logger.info("Comparing Large-Cap vs Mid-Cap Risk Metrics")
        
        calculator = RiskMetricsCalculator()
        
        lc_returns = large_cap_data['Close'].pct_change().dropna()
        mc_returns = midcap_data['Close'].pct_change().dropna()
        
        comparison = pd.DataFrame({
            'Metric': [
                'Mean Return',
                'Volatility',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'VaR (95%)',
                'Expected Shortfall'
            ],
            'Large-Cap': [
                lc_returns.mean(),
                calculator.calculate_volatility(lc_returns),
                calculator.calculate_sharpe_ratio(lc_returns),
                calculator.calculate_sortino_ratio(lc_returns),
                calculator.calculate_max_drawdown(large_cap_data['Close']),
                calculator.calculate_var(lc_returns),
                calculator.calculate_expected_shortfall(lc_returns)
            ],
            'Mid-Cap': [
                mc_returns.mean(),
                calculator.calculate_volatility(mc_returns),
                calculator.calculate_sharpe_ratio(mc_returns),
                calculator.calculate_sortino_ratio(mc_returns),
                calculator.calculate_max_drawdown(midcap_data['Close']),
                calculator.calculate_var(mc_returns),
                calculator.calculate_expected_shortfall(mc_returns)
            ]
        })
        
        comparison['Difference'] = comparison['Large-Cap'] - comparison['Mid-Cap']
        
        self.logger.info("\nComparison Results:")
        self.logger.info(comparison.to_string())
        
        return comparison
    
    def analyze_correlation(self, large_cap_data, midcap_data):
        """Analyze correlation between large-cap and mid-cap"""
        self.logger.info("Analyzing Large-Cap vs Mid-Cap Correlation")
        
        lc_returns = large_cap_data['Close'].pct_change().dropna()
        mc_returns = midcap_data['Close'].pct_change().dropna()
        
        min_len = min(len(lc_returns), len(mc_returns))
        lc_returns = lc_returns.iloc[:min_len]
        mc_returns = mc_returns.iloc[:min_len]
        
        correlation = lc_returns.corr(mc_returns)
        
        self.logger.info(f"Correlation: {correlation:.4f}")
        
        return {
            'Correlation': correlation,
            'Large_Cap_Volatility': lc_returns.std(),
            'Mid_Cap_Volatility': mc_returns.std(),
            'Volatility_Ratio': mc_returns.std() / lc_returns.std()
        }


class RollingWindowAnalyzer:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.logger = logging.getLogger(__name__)
    
    def analyze_rolling_metrics(self, actual_data, predicted_data, metric_name='Close', window=60):
        """Analyze prediction accuracy over rolling windows"""
        self.logger.info(f"Analyzing rolling metrics with window={window}")
        
        if isinstance(predicted_data, pd.Series):
            predicted_data = predicted_data.values
        elif isinstance(predicted_data, pd.DataFrame):
            predicted_data = predicted_data.iloc[:, 0].values
        
        actual_values = actual_data[metric_name].values[:len(predicted_data)]
        
        rolling_results = []
        
        for i in range(window, len(actual_values)):
            window_actual = actual_values[i-window:i]
            window_predicted = predicted_data[i-window:i]
            
            mae = np.mean(np.abs(window_actual - window_predicted))
            rmse = np.sqrt(np.mean((window_actual - window_predicted) ** 2))
            mape = np.mean(np.abs((window_actual - window_predicted) / window_actual)) * 100
            
            rolling_results.append({
                'Window_End': i,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            })
        
        rolling_df = pd.DataFrame(rolling_results)
        
        self.logger.info(f"Rolling Analysis Complete: {len(rolling_df)} windows analyzed")
        self.logger.info(f"  Mean MAE: {rolling_df['MAE'].mean():.4f}")
        self.logger.info(f"  Mean RMSE: {rolling_df['RMSE'].mean():.4f}")
        
        return rolling_df


# =============================================================================
# STAGE 4: GENERATIVE AI
# =============================================================================

class SentimentAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentiment_keywords = {
            'positive': ['bullish', 'growth', 'surge', 'rally', 'gain', 'profit', 'strong', 'outperform', 'beat', 'upgrade'],
            'negative': ['bearish', 'decline', 'crash', 'loss', 'weak', 'underperform', 'miss', 'downgrade', 'risk', 'concern'],
            'neutral': ['stable', 'flat', 'sideways', 'consolidate', 'hold', 'maintain']
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of market news text"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.sentiment_keywords['neutral'] if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count + neutral_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = 'NEUTRAL'
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            
            if sentiment_score > 0.3:
                sentiment_label = 'POSITIVE'
            elif sentiment_score < -0.3:
                sentiment_label = 'NEGATIVE'
            else:
                sentiment_label = 'NEUTRAL'
        
        self.logger.info(f"Sentiment Analysis: {sentiment_label} (Score: {sentiment_score:.4f})")
        
        return sentiment_score, sentiment_label
    
    def batch_analyze_sentiment(self, news_list):
        """Analyze sentiment for multiple news items"""
        results = []
        
        for i, news in enumerate(news_list):
            score, label = self.analyze_sentiment(news)
            results.append({
                'News_ID': i,
                'Text': news[:100],
                'Sentiment_Score': score,
                'Sentiment_Label': label
            })
        
        return pd.DataFrame(results)
    
    def calculate_aggregate_sentiment(self, sentiment_scores):
        """Calculate aggregate sentiment from multiple news items"""
        if len(sentiment_scores) == 0:
            return {
                'Mean_Sentiment': 0.0,
                'Sentiment_Strength': 0.0,
                'Positive_Ratio': 0.0,
                'Negative_Ratio': 0.0
            }
        
        mean_sentiment = np.mean(sentiment_scores)
        sentiment_strength = np.std(sentiment_scores)
        
        positive_count = sum(1 for s in sentiment_scores if s > 0.3)
        negative_count = sum(1 for s in sentiment_scores if s < -0.3)
        
        positive_ratio = positive_count / len(sentiment_scores)
        negative_ratio = negative_count / len(sentiment_scores)
        
        self.logger.info(f"Aggregate Sentiment: Mean={mean_sentiment:.4f}, Strength={sentiment_strength:.4f}")
        
        return {
            'Mean_Sentiment': mean_sentiment,
            'Sentiment_Strength': sentiment_strength,
            'Positive_Ratio': positive_ratio,
            'Negative_Ratio': negative_ratio
        }


class RiskForecastAdjuster:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def adjust_volatility_forecast(self, base_volatility, sentiment_score, sentiment_strength):
        """Adjust volatility forecast based on sentiment metrics"""
        volatility_adjustment = sentiment_strength * 0.5
        
        if sentiment_score < -0.3:
            volatility_adjustment *= 1.5
        
        adjusted_volatility = base_volatility * (1 + volatility_adjustment)
        
        self.logger.info(f"Volatility Adjustment: {base_volatility:.4f} -> {adjusted_volatility:.4f}")
        
        return adjusted_volatility
    
    def adjust_var_forecast(self, base_var, sentiment_score):
        """Adjust VaR forecast based on sentiment"""
        if sentiment_score < -0.3:
            adjustment_multiplier = 1 + abs(sentiment_score) * 0.3
            adjusted_var = base_var * adjustment_multiplier
        else:
            adjusted_var = base_var
        
        self.logger.info(f"VaR Adjustment: {base_var:.4f} -> {adjusted_var:.4f}")
        
        return adjusted_var


class ScenarioGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_monte_carlo_scenarios(self, historical_returns, num_scenarios=1000, num_days=20, sentiment_adjustment=1.0):
        """Generate Monte Carlo scenarios with sentiment adjustment"""
        self.logger.info(f"Generating {num_scenarios} Monte Carlo scenarios for {num_days} days")
        
        mean_return = historical_returns.mean()
        volatility = historical_returns.std() * sentiment_adjustment
        
        scenarios = np.zeros((num_scenarios, num_days))
        
        for i in range(num_scenarios):
            daily_returns = np.random.normal(mean_return, volatility, num_days)
            scenarios[i] = np.exp(np.cumsum(daily_returns))
        
        self.logger.info(f"Scenarios generated: Mean={scenarios.mean():.4f}, Std={scenarios.std():.4f}")
        
        return scenarios


class NarrativeGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_risk_narrative(self, risk_metrics, sentiment_score, sentiment_label):
        """Generate narrative summary of risk analysis"""
        narrative = f"""
RISK ANALYSIS NARRATIVE
======================

Market Sentiment: {sentiment_label} (Score: {sentiment_score:.4f})

Risk Profile:
- Value at Risk (95%): {risk_metrics.get('VaR_95', 'N/A')}
- Expected Shortfall: {risk_metrics.get('Expected_Shortfall', 'N/A')}
- Maximum Drawdown: {risk_metrics.get('Max_Drawdown', 'N/A')}
- Volatility: {risk_metrics.get('Volatility', 'N/A')}
- Sharpe Ratio: {risk_metrics.get('Sharpe_Ratio', 'N/A')}

Sentiment Impact:
"""
        
        if sentiment_label == 'POSITIVE':
            narrative += "- Market sentiment is positive, suggesting lower near-term downside risk\n"
            narrative += "- Investors show confidence in equity markets\n"
            narrative += "- Volatility may compress in the near term\n"
        elif sentiment_label == 'NEGATIVE':
            narrative += "- Market sentiment is negative, suggesting elevated downside risk\n"
            narrative += "- Investors are cautious about equity exposure\n"
            narrative += "- Volatility may expand, particularly for mid-cap funds\n"
        else:
            narrative += "- Market sentiment is neutral, suggesting stable risk conditions\n"
            narrative += "- No strong directional bias from sentiment\n"
            narrative += "- Risk metrics likely to remain stable\n"
        
        narrative += "\nRecommendations:\n"
        if sentiment_label == 'NEGATIVE':
            narrative += "- Consider increasing hedging positions\n"
            narrative += "- Reduce exposure to high-beta mid-cap funds\n"
            narrative += "- Monitor VaR closely for potential breaches\n"
        elif sentiment_label == 'POSITIVE':
            narrative += "- Consider increasing equity exposure\n"
            narrative += "- Mid-cap funds may offer attractive risk-return\n"
            narrative += "- Monitor for sentiment reversals\n"
        else:
            narrative += "- Maintain current portfolio positioning\n"
            narrative += "- Continue regular risk monitoring\n"
            narrative += "- Be prepared for sentiment shifts\n"
        
        return narrative


class AIInsightGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_adjuster = RiskForecastAdjuster()
        self.scenario_generator = ScenarioGenerator()
        self.narrative_generator = NarrativeGenerator()
    
    def generate_comprehensive_insights(self, risk_metrics, news_items, historical_returns):
        """Generate comprehensive AI insights combining risk analysis and sentiment"""
        self.logger.info("Generating comprehensive AI insights")
        
        sentiment_df = self.sentiment_analyzer.batch_analyze_sentiment(news_items)
        sentiment_scores = sentiment_df['Sentiment_Score'].tolist()
        aggregate_sentiment = self.sentiment_analyzer.calculate_aggregate_sentiment(sentiment_scores)
        
        mean_sentiment = aggregate_sentiment['Mean_Sentiment']
        sentiment_strength = aggregate_sentiment['Sentiment_Strength']
        
        adjusted_volatility = self.risk_adjuster.adjust_volatility_forecast(
            risk_metrics.get('Volatility', 0.12),
            mean_sentiment,
            sentiment_strength
        )
        
        adjusted_var = self.risk_adjuster.adjust_var_forecast(
            risk_metrics.get('VaR_95', -0.025),
            mean_sentiment
        )
        
        scenarios = self.scenario_generator.generate_monte_carlo_scenarios(
            historical_returns,
            num_scenarios=1000,
            num_days=20,
            sentiment_adjustment=1 + sentiment_strength
        )
        
        scenario_var = np.percentile(scenarios, 5)
        scenario_expected_shortfall = scenarios[scenarios <= scenario_var].mean()
        
        risk_narrative = self.narrative_generator.generate_risk_narrative(
            risk_metrics,
            mean_sentiment,
            sentiment_df['Sentiment_Label'].mode()[0] if len(sentiment_df) > 0 else 'NEUTRAL'
        )
        
        insights = {
            'Sentiment_Analysis': {
                'Mean_Sentiment': mean_sentiment,
                'Sentiment_Strength': sentiment_strength,
                'Positive_Ratio': aggregate_sentiment['Positive_Ratio'],
                'Negative_Ratio': aggregate_sentiment['Negative_Ratio'],
                'News_Count': len(news_items)
            },
            'Adjusted_Risk_Metrics': {
                'Original_Volatility': risk_metrics.get('Volatility', 0.12),
                'Adjusted_Volatility': adjusted_volatility,
                'Original_VaR': risk_metrics.get('VaR_95', -0.025),
                'Adjusted_VaR': adjusted_var,
                'Scenario_VaR': scenario_var,
                'Scenario_Expected_Shortfall': scenario_expected_shortfall
            },
            'Monte_Carlo_Scenarios': {
                'Mean_Return': scenarios.mean(),
                'Std_Return': scenarios.std(),
                'Min_Return': scenarios.min(),
                'Max_Return': scenarios.max(),
                'Percentile_5': np.percentile(scenarios, 5),
                'Percentile_95': np.percentile(scenarios, 95)
            },
            'Risk_Narrative': risk_narrative,
            'Sentiment_News_Details': sentiment_df.to_dict('records')
        }
        
        return insights


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_complete_pipeline(data_dir='data', output_dir='output'):
    """
    Run the complete equity fund risk analysis pipeline (risk metrics)
    
    Args:
        data_dir: Directory to load input data from
        output_dir: Directory to save output/metric files to
    """
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE EQUITY FUND RISK ANALYSIS PIPELINE")
    logger.info("=" * 70)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Stage 1: Data Loading & Preprocessing
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: Data Loading & Preprocessing")
    logger.info("=" * 70)
    
    logger.info("\n=== Loading Existing Data ===")
    try:
        # Load fund data
        fund_file = os.path.join(data_dir, 'fund_processed_merged.csv')
        fund_data = pd.read_csv(fund_file)
        fund_data['date'] = pd.to_datetime(fund_data['date'], format='%d-%m-%Y')
        fund_data = fund_data.set_index('date')
        logger.info(f"Loaded fund data: {len(fund_data)} records")
        
        # Load benchmark data
        benchmark_file = os.path.join(data_dir, 'benchmark_processed_NIFTY_50.csv')
        benchmark_data = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded benchmark data: {len(benchmark_data)} records")
        
        # Preprocess benchmark data if needed
        preprocessor = DataPreprocessor()
        if 'Volatility_20D' not in benchmark_data.columns:
            logger.info("Preprocessing benchmark data...")
            benchmark_data = preprocessor.handle_missing_values(benchmark_data, method='forward_fill')
            benchmark_data = preprocessor.calculate_returns(benchmark_data, price_column='Close')
            benchmark_data = preprocessor.calculate_volatility(benchmark_data, price_column='Close', window=20)
        
        # Store in expected format
        benchmark_data_dict = {'NIFTY_50': benchmark_data}
        fund_data_dict = {'Funds': fund_data}
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please ensure the following files exist in the data directory:")
        logger.error("  - fund_processed_merged.csv")
        logger.error("  - benchmark_NIFTY_50.csv")
        return
    
    # Stage 2: Predictive Modeling
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: Predictive Modeling")
    logger.info("=" * 70)
    
    logger.info("\n=== Loading Processed Data ===")
    try:
        nifty50_data = benchmark_data_dict['NIFTY_50']
        logger.info(f"Loaded NIFTY 50 data: {len(nifty50_data)} records")
    except KeyError:
        logger.error("NIFTY 50 data not found in loaded data")
        return
    
    logger.info("\n=== ARIMA Forecasting ===")
    if ARIMA_AVAILABLE:
        arima_forecaster = ARIMAForecaster(order=(5, 1, 2))
        if arima_forecaster.fit(nifty50_data, column='Close'):
            arima_forecast = arima_forecaster.forecast(steps=20)
            if arima_forecast is not None:
                logger.info("ARIMA Forecast (next 20 days):")
                logger.info(arima_forecast[['mean', 'mean_ci_lower', 'mean_ci_upper']].head())
                arima_forecast.to_csv(os.path.join(output_dir, 'arima_forecast_nifty50.csv'))
    else:
        logger.warning("ARIMA forecasting skipped. Install statsmodels: pip install statsmodels")
    
    logger.info("\n=== LSTM Forecasting ===")
    if LSTM_AVAILABLE:
        lstm_forecaster = LSTMForecaster(lookback=60, epochs=50, batch_size=32)
        
        X_train, X_test, y_train, y_test = lstm_forecaster.prepare_data(
            nifty50_data, column='Close', test_size=0.2
        )
        
        if X_train is not None:
            if lstm_forecaster.build_model(input_shape=(X_train.shape[1], 1)):
                if lstm_forecaster.train(X_train, y_train, X_val=X_test, y_val=y_test):
                    y_pred = lstm_forecaster.predict(X_test)
                    if y_pred is not None:
                        logger.info("\n=== LSTM Future Forecast ===")
                        lstm_forecast = lstm_forecaster.forecast_future(nifty50_data, column='Close', steps=20)
                        if lstm_forecast is not None:
                            logger.info(f"LSTM Forecast (next 20 days):")
                            logger.info(lstm_forecast[:5])
                            
                            forecast_df = pd.DataFrame({
                                'Day': range(1, len(lstm_forecast) + 1),
                                'Forecast': lstm_forecast
                            })
                            forecast_df.to_csv(os.path.join(output_dir, 'lstm_forecast_nifty50.csv'), index=False)
    else:
        logger.warning("LSTM forecasting skipped. Install TensorFlow: pip install tensorflow")
    
    # Stage 3: Risk Analysis
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: Risk Analysis & Ex-Post Risk Evaluation")
    logger.info("=" * 70)
    
    logger.info("\n=== Loading Processed Data ===")
    try:
        nifty50_data = benchmark_data_dict['NIFTY_50']
        
        # For fund data, analyze all funds
        if fund_data_dict and 'Funds' in fund_data_dict:
            fund_data = fund_data_dict['Funds']
            # Get unique funds
            unique_funds = fund_data['Fund'].unique()
            logger.info(f"Found {len(unique_funds)} unique funds in data")
            
            # Store all funds data for analysis
            funds_dict = {}
            for fund_name in unique_funds:
                fund_single = fund_data[fund_data['Fund'] == fund_name].copy()
                funds_dict[fund_name] = fund_single
                logger.info(f"Loaded fund {fund_name}: {len(fund_single)} records")
        else:
            logger.error("Fund data not available")
            return
        
        logger.info(f"Loaded NIFTY 50 data: {len(nifty50_data)} records")
        logger.info(f"Loaded {len(funds_dict)} funds for analysis")
    except KeyError as e:
        logger.error(f"Data not found: {str(e)}")
        return
    
    logger.info("\n=== Calculating Risk Metrics ===")
    calculator = RiskMetricsCalculator()
    
    logger.info("\nNIFTY 50 Risk Metrics:")
    nifty50_returns = nifty50_data['Close'].pct_change().dropna()
    nifty50_metrics = {
        'Mean_Return': nifty50_returns.mean(),
        'Volatility': calculator.calculate_volatility(nifty50_returns),
        'Sharpe_Ratio': calculator.calculate_sharpe_ratio(nifty50_returns),
        'Sortino_Ratio': calculator.calculate_sortino_ratio(nifty50_returns),
        'Max_Drawdown': calculator.calculate_max_drawdown(nifty50_data['Close']),
        'VaR_95': calculator.calculate_var(nifty50_returns),
        'Expected_Shortfall': calculator.calculate_expected_shortfall(nifty50_returns)
    }
    
    # Calculate risk metrics for all funds
    logger.info("\n=== Calculating Risk Metrics for All Funds ===")
    all_funds_metrics = {}
    for fund_name, fund_data in funds_dict.items():
        logger.info(f"\n{fund_name} Risk Metrics:")
        fund_returns = fund_data['Close'].pct_change().dropna()
        fund_metrics = {
            'Mean_Return': fund_returns.mean(),
            'Volatility': calculator.calculate_volatility(fund_returns),
            'Sharpe_Ratio': calculator.calculate_sharpe_ratio(fund_returns),
            'Sortino_Ratio': calculator.calculate_sortino_ratio(fund_returns),
            'Max_Drawdown': calculator.calculate_max_drawdown(fund_data['Close']),
            'VaR_95': calculator.calculate_var(fund_returns),
            'Expected_Shortfall': calculator.calculate_expected_shortfall(fund_returns)
        }
        all_funds_metrics[fund_name] = fund_metrics
        
        # Log metrics
        for metric, value in fund_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\n=== Comparative Analysis ===")
    comparative_analyzer = ComparativeFundAnalysis()
    
    # Compare each fund against NIFTY 50
    comparison_results = []
    for fund_name, fund_data in funds_dict.items():
        logger.info(f"\nComparing {fund_name} vs NIFTY 50")
        comparison_df = comparative_analyzer.compare_large_cap_vs_midcap(nifty50_data, fund_data)
        comparison_df['Fund'] = fund_name
        comparison_results.append(comparison_df)
    
    # Combine all comparisons
    all_comparisons_df = pd.concat(comparison_results, ignore_index=True)
    
    logger.info("\n=== Beta & Alpha Analysis ===")
    beta_alpha_calc = BetaAlphaCalculator()
    
    beta_alpha_results = []
    for fund_name, fund_data in funds_dict.items():
        logger.info(f"\nBeta & Alpha for {fund_name} vs NIFTY 50")
        fund_returns = fund_data['Close'].pct_change().dropna()
        beta_50 = beta_alpha_calc.calculate_beta(nifty50_returns, fund_returns)
        alpha_50 = beta_alpha_calc.calculate_alpha(nifty50_returns, fund_returns)
        ir_50 = beta_alpha_calc.calculate_information_ratio(nifty50_returns, fund_returns)
        
        beta_alpha_results.append({
            'Fund': fund_name,
            'Beta': beta_50,
            'Alpha': alpha_50,
            'Information_Ratio': ir_50
        })
        
        logger.info(f"Beta: {beta_50:.4f}")
        logger.info(f"Alpha: {alpha_50:.4f}")
        logger.info(f"Information Ratio: {ir_50:.4f}")
    
    beta_alpha_df = pd.DataFrame(beta_alpha_results)
    
    logger.info("\n=== Stress Testing ===")
    stress_analyzer = StressTestAnalyzer()
    
    # Run stress tests for each fund
    stress_results_list = []
    for fund_name, fund_data in funds_dict.items():
        logger.info(f"\nStress testing for {fund_name}")
        fund_stress = stress_analyzer.generate_stress_test_report(
            nifty50_data['Close'],
            fund_data['Close'],
            STRESS_TEST_SCENARIOS
        )
        fund_stress['Fund'] = fund_name
        stress_results_list.append(fund_stress)
    
    stress_results = pd.concat(stress_results_list, ignore_index=True)
    
    logger.info("\n=== Rolling Window Analysis ===")
    rolling_analyzer = RollingWindowAnalyzer(window_size=60)
    
    rolling_metrics_50 = rolling_analyzer.analyze_rolling_metrics(
        nifty50_data,
        nifty50_data['Close'].shift(1),
        metric_name='Close',
        window=60
    )
    
    # Rolling metrics for all funds
    rolling_metrics_funds_list = []
    for fund_name, fund_data in funds_dict.items():
        logger.info(f"\nRolling analysis for {fund_name}")
        fund_rolling = rolling_analyzer.analyze_rolling_metrics(
            fund_data,
            fund_data['Close'].shift(1),
            metric_name='Close',
            window=60
        )
        fund_rolling['Fund'] = fund_name
        rolling_metrics_funds_list.append(fund_rolling)
    
    rolling_metrics_funds = pd.concat(rolling_metrics_funds_list, ignore_index=True)
    
    logger.info("\n=== Ex-Post Risk Analysis ===")
    expost_analyzer = ExPostRiskAnalyzer()
    
    vol_analysis_50 = expost_analyzer.analyze_volatility_prediction_error(
        nifty50_data['Volatility_20D'].dropna(),
        nifty50_data['Volatility_20D'].shift(1).dropna()
    )
    
    # Volatility analysis for all funds
    vol_analysis_funds_list = []
    for fund_name, fund_data in funds_dict.items():
        logger.info(f"\nVolatility analysis for {fund_name}")
        fund_vol = expost_analyzer.analyze_volatility_prediction_error(
            fund_data['Volatility_20D'].dropna(),
            fund_data['Volatility_20D'].shift(1).dropna()
        )
        fund_vol['Fund'] = fund_name
        vol_analysis_funds_list.append(fund_vol)
    
    vol_analysis_funds = pd.concat(vol_analysis_funds_list, ignore_index=True)
    
    logger.info("\n=== Saving Results ===")
    
    # Save all funds metrics
    all_funds_metrics_df = pd.DataFrame.from_dict(all_funds_metrics, orient='index')
    all_funds_metrics_df.index.name = 'Fund'
    all_funds_metrics_df.to_csv(os.path.join(output_dir, 'risk_metrics_all_funds.csv'))
    logger.info(f"Risk metrics for all funds saved to {output_dir}/risk_metrics_all_funds.csv")
    
    # Also save NIFTY 50 metrics separately
    nifty50_df = pd.DataFrame([nifty50_metrics], index=['NIFTY_50'])
    nifty50_df.to_csv(os.path.join(output_dir, 'risk_metrics_nifty50.csv'))
    logger.info(f"NIFTY 50 metrics saved to {output_dir}/risk_metrics_nifty50.csv")
    
    all_comparisons_df.to_csv(os.path.join(output_dir, 'comparative_analysis.csv'), index=False)
    logger.info(f"Comparative analysis saved to {output_dir}/comparative_analysis.csv")
    
    beta_alpha_df.to_csv(os.path.join(output_dir, 'beta_alpha_analysis.csv'), index=False)
    logger.info(f"Beta & Alpha analysis saved to {output_dir}/beta_alpha_analysis.csv")
    
    stress_results.to_csv(os.path.join(output_dir, 'stress_test_results.csv'), index=False)
    logger.info(f"Stress test results saved to {output_dir}/stress_test_results.csv")
    
    rolling_metrics_50.to_csv(os.path.join(output_dir, 'rolling_metrics_nifty50.csv'), index=False)
    logger.info(f"Rolling metrics saved to {output_dir}/rolling_metrics_nifty50.csv")
    
    rolling_metrics_funds.to_csv(os.path.join(output_dir, 'rolling_metrics_all_funds.csv'), index=False)
    logger.info(f"Rolling metrics for all funds saved to {output_dir}/rolling_metrics_all_funds.csv")
    
    vol_analysis_50.to_csv(os.path.join(output_dir, 'volatility_analysis_nifty50.csv'))
    logger.info(f"Volatility analysis saved to {output_dir}/volatility_analysis_nifty50.csv")
    
    vol_analysis_funds.to_csv(os.path.join(output_dir, 'volatility_analysis_all_funds.csv'), index=False)
    logger.info(f"Volatility analysis for all funds saved to {output_dir}/volatility_analysis_all_funds.csv")
    
    # Stage 4: Generative AI Integration
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 4: Generative AI Integration")
    logger.info("=" * 70)
    
    logger.info("\n=== Loading Processed Data ===")
    try:
        nifty50_data = benchmark_data_dict['NIFTY_50']
        
        risk_metrics_file = os.path.join(output_dir, 'risk_metrics_nifty50.csv')
        risk_metrics_df = pd.read_csv(risk_metrics_file, index_col=0)
        
        logger.info(f"Loaded NIFTY 50 data: {len(nifty50_data)} records")
        logger.info(f"Loaded risk metrics")
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return
    
    logger.info("\n=== Sample Market News ===")
    sample_news = [
        "NIFTY 50 surges on strong corporate earnings and bullish investor sentiment",
        "Mid-cap stocks rally as growth prospects improve amid economic recovery",
        "Market shows concern over inflation risks and potential rate hikes",
        "Tech stocks underperform as investors worry about valuation concerns",
        "Banking sector gains strength on improved credit growth and profitability",
        "Market consolidates as investors assess mixed economic data",
        "Equity markets decline on geopolitical tensions and risk-off sentiment",
        "Large-cap funds outperform as investors seek stability and dividends",
        "Mid-cap volatility spikes amid profit-taking and sector rotation",
        "Market sentiment turns positive on strong monsoon forecast and rural demand"
    ]
    
    logger.info("\n=== Generating AI Insights ===")
    ai_generator = AIInsightGenerator()
    
    nifty50_returns = nifty50_data['Close'].pct_change().dropna()
    
    risk_metrics = {
        'Volatility': risk_metrics_df.loc['NIFTY_50', 'Volatility'],
        'Sharpe_Ratio': risk_metrics_df.loc['NIFTY_50', 'Sharpe_Ratio'],
        'VaR_95': risk_metrics_df.loc['NIFTY_50', 'VaR_95'],
        'Expected_Shortfall': risk_metrics_df.loc['NIFTY_50', 'Expected_Shortfall'],
        'Max_Drawdown': risk_metrics_df.loc['NIFTY_50', 'Max_Drawdown']
    }
    
    insights = ai_generator.generate_comprehensive_insights(
        risk_metrics,
        sample_news,
        nifty50_returns
    )
    
    logger.info("\n=== Saving AI Insights ===")
    
    insights_json = {
        'Timestamp': datetime.now().isoformat(),
        'Sentiment_Analysis': insights['Sentiment_Analysis'],
        'Adjusted_Risk_Metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                  for k, v in insights['Adjusted_Risk_Metrics'].items()},
        'Monte_Carlo_Scenarios': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                                  for k, v in insights['Monte_Carlo_Scenarios'].items()}
    }
    
    with open(os.path.join(output_dir, 'ai_insights.json'), 'w') as f:
        json.dump(insights_json, f, indent=2)
    logger.info(f"AI insights saved to {output_dir}/ai_insights.json")
    
    with open(os.path.join(output_dir, 'risk_narrative.txt'), 'w') as f:
        f.write(insights['Risk_Narrative'])
    logger.info(f"Risk narrative saved to {output_dir}/risk_narrative.txt")
    
    sentiment_df = pd.DataFrame(insights['Sentiment_News_Details'])
    sentiment_df.to_csv(os.path.join(output_dir, 'sentiment_analysis.csv'), index=False)
    logger.info(f"Sentiment analysis saved to {output_dir}/sentiment_analysis.csv")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE - All Stages Executed Successfully")
    logger.info("=" * 70)
    logger.info(f"\nInput data loaded from: {data_dir}")
    logger.info(f"Output files saved to: {output_dir}")
    logger.info("\nGenerated Output Files:")
    logger.info("  - arima_forecast_nifty50.csv")
    logger.info("  - lstm_forecast_nifty50.csv")
    logger.info("  - risk_metrics.csv")
    logger.info("  - comparative_analysis.csv")
    logger.info("  - beta_alpha_analysis.csv")
    logger.info("  - stress_test_results.csv")
    logger.info("  - rolling_metrics_*.csv")
    logger.info("  - volatility_analysis_*.csv")
    logger.info("  - ai_insights.json")
    logger.info("  - risk_narrative.txt")
    logger.info("  - sentiment_analysis.csv")
    logger.info("\nTo view the dashboard, run: py -m streamlit run dashboard.py")


if __name__ == '__main__':
    run_complete_pipeline()
