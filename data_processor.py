# data_processor.py - Data Download, Processing & LSTM Sequence Creation

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from technical_indicators import add_technical_indicators, get_indicator_columns
import config as cfg


class DataProcessor:
    """
    Handles stock data download, preprocessing, and LSTM sequence generation.
    """
    
    def __init__(self, scaler_type='minmax'):
        """
        Initialize DataProcessor.
        
        Parameters:
        -----------
        scaler_type : str
            'minmax' or 'standard'
        """
        self.scaler_type = scaler_type
        self.scalers = {}
        self.data_cache = {}
        
        # Create directories
        Path(cfg.DATA_PATH).mkdir(exist_ok=True)
        Path(cfg.MODEL_PATH).mkdir(exist_ok=True)
        Path(cfg.PREDICTION_PATH).mkdir(exist_ok=True)
        Path(cfg.SCALER_PATH).mkdir(exist_ok=True)
    
    def download_stock_data(self, tickers, start_date=None, end_date=None):
        """
        Download historical stock data from Yahoo Finance.
        
        Parameters:
        -----------
        tickers : str or list
            Stock ticker(s) to download
        start_date : str
            Start date in 'YYYY-MM-DD' format (default: config START_DATE)
        end_date : str
            End date in 'YYYY-MM-DD' format (default: today)
        
        Returns:
        --------
        dict
            {ticker: dataframe}
        """
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        if start_date is None:
            start_date = cfg.START_DATE
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        data_dict = {}
        
        for ticker in tickers:
            print(f"📥 Downloading {ticker} data ({start_date} to {end_date})...")
            
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                
                if df.empty:
                    print(f"⚠️ No data for {ticker}")
                    continue
                
                # Reset index to have Date as column
                df = df.reset_index()
                
                # Rename columns
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # Sort by date
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Check minimum data points
                if len(df) < cfg.MIN_DATA_POINTS:
                    print(f"⚠️ {ticker} has only {len(df)} data points (need {cfg.MIN_DATA_POINTS})")
                    continue
                
                # Handle missing values
                df = self._handle_missing_values(df)
                
                data_dict[ticker] = df
                self.data_cache[ticker] = df
                
                print(f"✓ Downloaded {ticker}: {len(df)} records")
                
            except Exception as e:
                print(f"❌ Error downloading {ticker}: {e}")
        
        return data_dict
    
    def _handle_missing_values(self, df):
        """Handle missing values in OHLCV data."""
        
        # Forward fill for small gaps
        df = df.fillna(method='ffill', limit=5)
        
        # Drop remaining NaN rows
        df = df.dropna()
        
        return df
    
    def add_indicators(self, df):
        """Add technical indicators to dataframe."""
        return add_technical_indicators(df, cfg.TECHNICAL_INDICATORS)
    
    def prepare_data(self, ticker, start_date=None, end_date=None):
        """
        Download data and prepare with indicators.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        start_date : str
            Start date
        end_date : str
            End date
        
        Returns:
        --------
        tuple
            (df_with_indicators, feature_columns)
        """
        
        # Download data
        data_dict = self.download_stock_data(ticker, start_date, end_date)
        
        if ticker not in data_dict:
            raise ValueError(f"Failed to download {ticker}")
        
        df = data_dict[ticker].copy()
        
        # Add technical indicators
        df = self.add_indicators(df)
        
        # Select features
        feature_cols = get_indicator_columns(cfg.TECHNICAL_INDICATORS)
        
        # Ensure Close is first
        features = ['Close'] + [col for col in feature_cols if col != 'Close']
        features = [col for col in features if col in df.columns]
        
        return df, features
    
    def create_sequences(self, data, lookback):
        """
        Create LSTM sequences from data.
        
        Parameters:
        -----------
        data : np.ndarray
            Scaled feature data (n_samples, n_features)
        lookback : int
            Number of time steps to look back
        
        Returns:
        --------
        tuple
            (X, y) where X is (n_samples, lookback, n_features) 
            and y is (n_samples,)
        """
        
        X, y = [], []
        
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback, 0])  # Close price (first column)
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, data, feature_cols, fit=True, ticker=None):
        """
        Normalize data using MinMaxScaler or StandardScaler.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with OHLCV and indicators
        feature_cols : list
            Columns to normalize
        fit : bool
            If True, fit scaler; if False, use existing
        ticker : str
            Ticker for saving scaler
        
        Returns:
        --------
        tuple
            (scaled_data_array, scaler_object)
        """
        
        # Extract feature data
        feature_data = data[feature_cols].values
        
        # Create scaler
        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()
        
        if fit:
            scaled_data = scaler.fit_transform(feature_data)
        else:
            scaled_data = scaler.transform(feature_data)
        
        # Save scaler
        if ticker and fit:
            scaler_path = os.path.join(cfg.SCALER_PATH, f'{ticker}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            self.scalers[ticker] = scaler
        
        return scaled_data, scaler
    
    def load_scaler(self, ticker):
        """Load saved scaler for a ticker."""
        
        scaler_path = os.path.join(cfg.SCALER_PATH, f'{ticker}_scaler.pkl')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        self.scalers[ticker] = scaler
        
        return scaler
    
    def prepare_datasets(self, ticker, lookback=60, start_date=None, end_date=None):
        """
        Prepare complete datasets for model training.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        lookback : int
            Number of lookback days
        start_date : str
            Data start date
        end_date : str
            Data end date
        
        Returns:
        --------
        dict
            {
                'X_train': (n_train, lookback, n_features),
                'y_train': (n_train,),
                'X_val': (n_val, lookback, n_features),
                'y_val': (n_val,),
                'X_test': (n_test, lookback, n_features),
                'y_test': (n_test,),
                'scaler': MinMaxScaler object,
                'feature_cols': list of feature columns,
                'raw_df': original dataframe
            }
        """
        
        # Prepare data
        df, feature_cols = self.prepare_data(ticker, start_date, end_date)
        
        # Normalize
        scaled_data, scaler = self.normalize_data(
            df, feature_cols, fit=True, ticker=ticker
        )
        
        # Create sequences
        X, y = self.create_sequences(scaled_data, lookback)
        
        # Calculate split indices
        train_size = int(len(X) * cfg.LSTM_CONFIG['train_split'])
        val_size = int(len(X) * cfg.LSTM_CONFIG['validation_split'])
        
        # Split data
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"\n✓ Dataset prepared for {ticker}:")
        print(f"  Train: X{X_train.shape}, y{y_train.shape}")
        print(f"  Val:   X{X_val.shape}, y{y_val.shape}")
        print(f"  Test:  X{X_test.shape}, y{y_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'raw_df': df,
        }
    
    def inverse_transform(self, scaled_data, scaler, feature_cols):
        """
        Convert scaled data back to original values.
        
        Parameters:
        -----------
        scaled_data : np.ndarray
            Scaled predictions (Close prices only)
        scaler : MinMaxScaler
            Fitted scaler
        feature_cols : list
            Original feature columns
        
        Returns:
        --------
        np.ndarray
            Original scale predictions
        """
        
        # Create dummy array with same shape as original features
        dummy = np.zeros((len(scaled_data), len(feature_cols)))
        dummy[:, 0] = scaled_data  # Put predictions in first column (Close)
        
        # Inverse transform
        original_scale = scaler.inverse_transform(dummy)
        
        # Return only Close prices (first column)
        return original_scale[:, 0]
