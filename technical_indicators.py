# technical_indicators.py - 10+ Technical Indicators for Stock Analysis

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import SMAIndicator, EMAIndicator
import warnings

warnings.filterwarnings('ignore')

def add_technical_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add 10+ technical indicators to stock dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with columns: Open, High, Low, Close, Volume
    config : dict
        Technical indicator parameters (periods, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Original data + indicator columns
    """
    
    if config is None:
        from config import TECHNICAL_INDICATORS as config
    
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")
    
    # ==================== SIMPLE MOVING AVERAGES ====================
    for period in config['SMA_periods']:
        sma = SMAIndicator(close=df['Close'], window=period, fillna=True)
        df[f'SMA_{period}'] = sma.sma_indicator()
    
    # ==================== EXPONENTIAL MOVING AVERAGES ====================
    for period in config['EMA_periods']:
        ema = EMAIndicator(close=df['Close'], window=period, fillna=True)
        df[f'EMA_{period}'] = ema.ema_indicator()
    
    # ==================== RELATIVE STRENGTH INDEX ====================
    rsi = RSIIndicator(close=df['Close'], window=config['RSI_period'], fillna=True)
    df['RSI_14'] = rsi.rsi()
    
    # ==================== MACD (Moving Average Convergence Divergence) ====================
    macd = MACD(
        close=df['Close'],
        window_fast=config['MACD_fast'],
        window_slow=config['MACD_slow'],
        window_sign=config['MACD_signal'],
        fillna=True
    )
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # ==================== BOLLINGER BANDS ====================
    bb = BollingerBands(
        close=df['Close'],
        window=config['BB_period'],
        window_dev=config['BB_std'],
        fillna=True
    )
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # Bollinger Bands width and position
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Position'] = df['BB_Position'].fillna(0.5)  # Fill NaN with middle
    
    # ==================== AVERAGE TRUE RANGE ====================
    atr = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=config['ATR_period'],
        fillna=True
    )
    df['ATR_14'] = atr.average_true_range()
    
    # ==================== RATE OF CHANGE ====================
    roc = ROCIndicator(close=df['Close'], window=config['ROC_period'], fillna=True)
    df['ROC_12'] = roc.roc()
    
    # ==================== VOLUME-BASED INDICATORS ====================
    
    # Volume Rate of Change
    df['Volume_ROC'] = df['Volume'].pct_change() * 100
    df['Volume_ROC'] = df['Volume_ROC'].fillna(0)
    
    # Volume Moving Average
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_SMA'] = df['Volume_SMA'].fillna(method='bfill')
    
    # ==================== PRICE METRICS ====================
    
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Daily_Return'] = df['Daily_Return'].fillna(0)
    
    # High-Low Ratio
    df['HL_Ratio'] = (df['High'] - df['Low']) / df['Low'] * 100
    
    # Close-Open Ratio
    df['CO_Ratio'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Fill any remaining NaN values
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df


def get_indicator_columns(config: dict = None) -> list:
    """Get list of all indicator column names."""
    
    if config is None:
        from config import TECHNICAL_INDICATORS as config
    
    indicators = [
        # SMA
        *[f'SMA_{p}' for p in config['SMA_periods']],
        # EMA
        *[f'EMA_{p}' for p in config['EMA_periods']],
        # RSI
        'RSI_14',
        # MACD
        'MACD', 'MACD_Signal', 'MACD_Hist',
        # Bollinger Bands
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
        # ATR
        'ATR_14',
        # ROC
        'ROC_12',
        # Volume
        'Volume_ROC', 'Volume_SMA',
        # Price metrics
        'Daily_Return', 'HL_Ratio', 'CO_Ratio',
    ]
    
    return indicators


def validate_indicators(df: pd.DataFrame) -> bool:
    """Check if all required indicators are present and valid."""
    
    required = ['SMA_20', 'EMA_12', 'RSI_14', 'MACD', 'BB_Upper', 'ATR_14', 'ROC_12']
    
    for col in required:
        if col not in df.columns:
            print(f"❌ Missing indicator: {col}")
            return False
        if df[col].isna().all():
            print(f"❌ Indicator {col} is all NaN")
            return False
    
    print("✓ All indicators validated successfully")
    return True
