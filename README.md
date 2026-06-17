# AI-Powered Equity Mutual Fund Forecasting & Ex-Post Risk Analysis

## Project Overview
This project forecasts equity mutual fund performance (large-cap vs mid-cap), calculates risk metrics, and performs ex-post risk analysis to compare predicted vs. realized outcomes.

## Project Structure

```
windsurf-project/
├── data/                          # Raw and processed data
├── logs/                          # Execution logs
├── stage1_data_collection.py      # Data collection & preprocessing
├── stage1_config.py               # Configuration for Stage 1
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Stage 1: Data Collection & Preprocessing

### Objectives
- Collect historical NAV data for large-cap and mid-cap equity funds
- Fetch benchmark indices (NIFTY 50, NIFTY Midcap 150)
- Preprocess data (handle missing values, calculate returns, volatility)
- Prepare clean datasets for modeling

### Components

#### `EquityFundDataCollector`
- **fetch_benchmark_indices()**: Downloads NIFTY 50 and NIFTY Midcap 150 data
- **fetch_mutual_fund_nav()**: Downloads mutual fund NAV data
- **save_data()**: Saves data to CSV files
- **load_data()**: Loads data from CSV files

#### `DataPreprocessor`
- **handle_missing_values()**: Handles missing data (forward fill, interpolation, drop)
- **calculate_returns()**: Computes returns for multiple periods (1D, 5D, 20D)
- **calculate_volatility()**: Calculates rolling volatility
- **normalize_data()**: Normalizes data to 0-1 range
- **remove_outliers()**: Removes outliers using z-score method

### Configuration (`stage1_config.py`)
- **Data paths**: DATA_DIR, LOGS_DIR
- **Date range**: START_DATE, END_DATE
- **Benchmark indices**: NIFTY 50, NIFTY Midcap 150
- **Sample funds**: Large-cap and mid-cap equity stocks
- **Preprocessing parameters**: Missing value method, outlier threshold, volatility window

### Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data collection**:
   ```bash
   python stage1_data_collection.py
   ```

3. **Output**:
   - `data/benchmark_NIFTY_50.csv` - Raw NIFTY 50 data
   - `data/benchmark_NIFTY_MIDCAP_150.csv` - Raw NIFTY Midcap 150 data
   - `data/benchmark_processed_NIFTY_50.csv` - Processed NIFTY 50 data
   - `data/benchmark_processed_NIFTY_MIDCAP_150.csv` - Processed NIFTY Midcap 150 data

### Data Features
After preprocessing, each dataset includes:
- **OHLCV**: Open, High, Low, Close, Volume
- **Returns**: 1-day, 5-day, 20-day returns
- **Volatility**: 20-day rolling volatility
- **Index**: Benchmark index name

### Next Steps
- Stage 2: Predictive Modeling (ARIMA, LSTM)
- Stage 3: Risk Analysis (VaR, Expected Shortfall, Sharpe ratio)
- Stage 4: Generative AI Integration
- Stage 5: Dashboard Development

## Dependencies
- pandas: Data manipulation
- numpy: Numerical computing
- yfinance: Financial data download
- matplotlib: Visualization
- seaborn: Statistical visualization
- scikit-learn: Machine learning utilities

## Notes
- Data is fetched from Yahoo Finance; availability may vary for Indian mutual funds
- Consider supplementing with AMFI India or Kaggle datasets for comprehensive coverage
- All timestamps are in UTC; adjust as needed for IST
