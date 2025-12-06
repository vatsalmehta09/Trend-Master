# Enhanced Stock Market Predictor - Project Setup

## Overview
This is an advanced stock price prediction system using LSTM neural networks with technical indicators, multiple stock support, and real-time visualization.

## Key Improvements Over Basic Code

### 1. **Multiple Stock Support**
   - User selects from predefined stocks or enters custom tickers
   - Batch processing for multiple companies

### 2. **Enhanced Accuracy**
   - Technical indicators (SMA, EMA, RSI, MACD)
   - Larger lookback window options (60, 90, 120 days)
   - Better data preprocessing
   - Model evaluation metrics (RMSE, MAE, MAPE)
   - Train/validation/test split (80/10/10)

### 3. **Interactive Features**
   - Streamlit web interface
   - Dynamic model configuration
   - Real-time predictions
   - Interactive charts with Plotly
   - Download predictions as CSV

### 4. **Better Architecture**
   - Bidirectional LSTM option
   - Attention mechanism ready
   - Multiple LSTM layers with dropout
   - Batch normalization
   - Early stopping

### 5. **Model Persistence**
   - Save/load trained models
   - Reuse without retraining

## File Structure
```
stock-predictor/
├── main.py                 # Streamlit app
├── model_trainer.py        # LSTM model & training logic
├── data_processor.py       # Data preprocessing
├── technical_indicators.py # TA-Lib indicators
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── models/                # Saved models
├── data/                  # Downloaded data cache
├── predictions/           # Saved predictions
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run main.py
```

## Model Features

- **Input Features**: Close, Volume, SMA, EMA, RSI, MACD, Bollinger Bands
- **Architecture**: 2-3 LSTM layers + Dense layers
- **Dropout**: 20-30% to prevent overfitting
- **Lookback**: Configurable (default 60 days)
- **Forecast**: 30/60/90 days ahead
- **Batch Size**: 32 (optimized for training)
- **Epochs**: 50+ (with early stopping)

## Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

## Best Practices Implemented

1. ✅ Train/validation/test split for proper evaluation
2. ✅ Data normalization with MinMaxScaler
3. ✅ Feature engineering with technical indicators
4. ✅ Model regularization (Dropout, Early Stopping)
5. ✅ Batch processing for multiple stocks
6. ✅ Real-time visualization
7. ✅ Model checkpointing
8. ✅ Error handling & logging

## Performance Tips

- Use GPU if available (10x faster training)
- Cache data to avoid repeated downloads
- Increase LSTM units (50→100) for better accuracy
- More training data = better predictions
- Add sentiment analysis for enhanced predictions

## Disclaimer

⚠️ **This is for educational purposes only. Do not use for actual trading without proper financial advice.**

Stock prices are influenced by countless unpredictable factors. Machine learning models have limitations and should never be your sole decision-making tool.

---

Created: 2025-12-01
Tested with Python 3.8+, TensorFlow 2.x
