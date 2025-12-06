# config.py - Complete Configuration for 65+ Stocks & LSTM Hyperparameters

# ==================== STOCK CONFIGURATION ====================

# 40+ Indian Stocks (NSE)
INDIAN_STOCKS = {
    'IT': {
        'TCS.NS': 'Tata Consultancy Services',
        'INFY.NS': 'Infosys Limited',
        'WIPRO.NS': 'Wipro Limited',
        'HCLTECH.NS': 'HCL Technologies',
        'TECHM.NS': 'Tech Mahindra',
    },
    'Banking': {
        'SBIN.NS': 'State Bank of India',
        'HDFCBANK.NS': 'HDFC Bank Limited',
        'ICICIBANK.NS': 'ICICI Bank Limited',
        'KOTAK.NS': 'Kotak Mahindra Bank',
        'AXISBANK.NS': 'Axis Bank Limited',
        'INDUSIND.NS': 'IndusInd Bank',
    },
    'Pharma': {
        'SUNPHARMA.NS': 'Sun Pharmaceutical',
        'DIVI.NS': 'Divi\'s Laboratories',
        'LUPIN.NS': 'Lupin Limited',
        'CIPLA.NS': 'Cipla Limited',
        'DIVISLAB.NS': 'Divis Labs',
    },
    'FMCG': {
        'ITC.NS': 'ITC Limited',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'MARICO.NS': 'Marico Limited',
        'BRITANNIA.NS': 'Britannia Industries',
        'NESTLEIND.NS': 'Nestle India',
    },
    'Energy': {
        'RELIANCE.NS': 'Reliance Industries',
        'ONGC.NS': 'ONGC Limited',
        'NTPC.NS': 'NTPC Limited',
        'BPCL.NS': 'Bharat Petroleum',
        'IOCL.NS': 'Indian Oil Corporation',
    },
    'Auto': {
        'MARUTI.NS': 'Maruti Suzuki India',
        'TATAMOTORS.NS': 'Tata Motors Limited',
        'HYUNDAI.NS': 'Hyundai Motor India',
        'TVSMOTOR.NS': 'TVS Motor Company',
        'EICHER.NS': 'Eicher Motors',
    },
    'Infrastructure': {
        'LT.NS': 'Larsen & Toubro',
        'ADANIPORTS.NS': 'Adani Ports',
        'ADANIGREEN.NS': 'Adani Green Energy',
        'ASHOKLEY.NS': 'Ashok Leyland',
        'ULTRACEMCO.NS': 'UltraTech Cement',
    },
    'Healthcare': {
        'HDFCLIFE.NS': 'HDFC Life Insurance',
        'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
        'APOLLOHOSP.NS': 'Apollo Hospitals',
        'BIOCON.NS': 'Biocon Limited',
        'AUROPHARMA.NS': 'Aurobindo Pharma',
    },
}

# 25+ US Stocks (NYSE/NASDAQ)
US_STOCKS = {
    'Technology': {
        'AAPL': 'Apple Inc',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc (Google)',
        'META': 'Meta Platforms',
        'NVDA': 'NVIDIA Corporation',
    },
    'Finance': {
        'JPM': 'JPMorgan Chase',
        'BAC': 'Bank of America',
        'WFC': 'Wells Fargo',
        'GS': 'Goldman Sachs',
        'MS': 'Morgan Stanley',
    },
    'Consumer': {
        'WMT': 'Walmart Inc',
        'KO': 'The Coca-Cola Company',
        'PEP': 'PepsiCo Inc',
        'NKE': 'Nike Inc',
        'MCD': 'McDonald\'s Corporation',
    },
    'Energy': {
        'XOM': 'Exxon Mobil',
        'CVX': 'Chevron Corporation',
        'COP': 'ConocoPhillips',
        'SLB': 'Schlumberger',
        'EOG': 'EOG Resources',
    },
    'Healthcare': {
        'JNJ': 'Johnson & Johnson',
        'PFE': 'Pfizer Inc',
        'ABBV': 'AbbVie Inc',
        'UNH': 'UnitedHealth Group',
        'CVS': 'CVS Health',
    },
    'Industrial': {
        'BA': 'The Boeing Company',
        'CAT': 'Caterpillar Inc',
        'GE': 'General Electric',
        'RTX': 'Raytheon Technologies',
        'HON': 'Honeywell International',
    },
}

# Combined all stocks for easy access
POPULAR_STOCKS = {
    **{'India ' + k: list(v.keys()) for k, v in INDIAN_STOCKS.items()},
    **{'US ' + k: list(v.keys()) for k, v in US_STOCKS.items()},
}

# ==================== LSTM HYPERPARAMETERS ====================

LSTM_CONFIG = {
    'lookback_periods': [60, 90, 120],          # Days of history to use
    'forecast_periods': [30, 60, 90],           # Days to forecast ahead
    'lstm_units': [50, 100, 150],               # LSTM layer units
    'dropout_rates': [0.1, 0.2, 0.3, 0.4],      # Dropout probability
    'batch_size': 32,                           # Training batch size
    'epochs': 100,                              # Max training epochs
    'learning_rate': 0.001,                     # Adam optimizer LR
    'validation_split': 0.1,                    # 10% validation
    'test_split': 0.1,                          # 10% test
    'train_split': 0.8,                         # 80% train
    'early_stopping_patience': 15,              # Epochs to wait
    'reduce_lr_patience': 5,                    # Patience for LR reduction
    'reduce_lr_factor': 0.5,                    # LR reduction factor
}

# ==================== TECHNICAL INDICATORS ====================

TECHNICAL_INDICATORS = {
    'SMA_periods': [20, 50, 200],               # Simple Moving Averages
    'EMA_periods': [12, 26],                    # Exponential Moving Averages
    'RSI_period': 14,                           # Relative Strength Index
    'MACD_fast': 12,                            # MACD fast line
    'MACD_slow': 26,                            # MACD slow line
    'MACD_signal': 9,                           # MACD signal line
    'BB_period': 20,                            # Bollinger Bands period
    'BB_std': 2,                                # Bollinger Bands std dev
    'ATR_period': 14,                           # Average True Range
    'ROC_period': 12,                           # Rate of Change
}

# ==================== FEATURE SELECTION ====================

FEATURES_TO_USE = [
    'Close',                    # Target variable
    'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_12', 'EMA_26',
    'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
    'ATR_14',
    'ROC_12',
    'Volume_ROC',
]

# ==================== DATA PATHS ====================

DATA_PATH = 'data/'
MODEL_PATH = 'models/'
PREDICTION_PATH = 'predictions/'

# ==================== DATE & TIME ====================

START_DATE = '2015-01-01'           # 10 years of historical data
MIN_DATA_POINTS = 100               # Minimum data required for training

# ==================== SCALERS ====================

SCALER_PATH = 'models/scalers/'
NORMALIZATION_METHOD = 'minmax'     # 'minmax' or 'standard'
