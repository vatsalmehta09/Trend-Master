# README.md - TrendMaster: Advanced LSTM Stock Price Predictor

## 🚀 Project Overview

**TrendMaster** - A production-ready stock price prediction system with LSTM neural networks, 65+ stocks, interactive Streamlit dashboard, and 10+ technical indicators.

### ✨ Key Features

✅ **65+ Stocks** - 40+ Indian (NSE) + 25+ US (NYSE/NASDAQ)
✅ **LSTM Architecture** - Stacked layers with dropout, callbacks
✅ **10+ Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ROC
✅ **Interactive Dashboard** - 4 tabs, Plotly charts, real-time progress
✅ **Complete Data Pipeline** - Download → Preprocess → Normalize → Train
✅ **Advanced Metrics** - RMSE, MAE, MAPE, R² score
✅ **Multi-Horizon Forecast** - 30/60/90 day predictions
✅ **Model Persistence** - Save and reuse trained models

---

## 📦 What You Get

### Python Code (5 files, ~1800 lines)
- **main.py** - Streamlit dashboard (700 lines)
- **model_trainer.py** - LSTM model class (350 lines)
- **data_processor.py** - Data pipeline (400 lines)
- **technical_indicators.py** - 10+ indicators (200 lines)
- **config.py** - 65+ stocks, hyperparameters (150 lines)

### Documentation (10 files, ~50 pages)
- README.md (this file)
- QUICKSTART.md - 5-min setup
- COMPLETE_SETUP.md - Detailed installation
- HOW_TO_RUN.md - 3 ways to run
- INDIAN_STOCKS.md - 40+ NSE stocks
- USA_STOCKS.md - 25+ US stocks
- MASTER_SUMMARY.md - Architecture & learning
- IMPROVEMENTS.md - Future enhancements
- INDEX.md - Navigation guide

---

## ⚡ Quick Start

### 1. Setup (2 minutes)
```bash
# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run (1 minute)
```bash
streamlit run main.py
```

### 3. Predict (3-5 minutes)
1. Select stock from sidebar
2. Click "📥 Download Data" (Tab 1)
3. Click "🤖 Train Model" (Tab 2)
4. Click "🔮 Generate Forecast" (Tab 3)
5. Download predictions as CSV

---

## 📊 Architecture

```
User Input (main.py)
    ↓
Download (yfinance)
    ↓
Add Indicators (technical_indicators.py)
    ↓
Normalize (MinMaxScaler)
    ↓
Create Sequences (lookback=60)
    ↓
Split Train/Val/Test (80/10/10)
    ↓
LSTM Model (model_trainer.py)
    ├─ LSTM Layer 1 (100 units, dropout 0.2)
    ├─ LSTM Layer 2 (50 units, dropout 0.2)
    ├─ Dense (50 → 25 → 1)
    └─ Output (Close Price)
    ↓
Evaluate (RMSE, MAE, MAPE, R²)
    ↓
Forecast (30/60/90 days)
    ↓
Visualize (Plotly charts)
```

---

## 🎯 Technical Indicators

### Trend Indicators
- **SMA** (20, 50, 200) - Simple moving average
- **EMA** (12, 26) - Exponential moving average

### Momentum Indicators
- **RSI** (14) - Relative Strength Index (0-100)
- **MACD** - Moving Average Convergence Divergence
- **ROC** (12) - Rate of Change

### Volatility Indicators
- **Bollinger Bands** - Upper, middle, lower bands
- **ATR** (14) - Average True Range

### Volume Indicators
- **Volume ROC** - Volume rate of change
- **Volume SMA** - Volume moving average

### Price Metrics
- **Daily Return** - Daily percentage change
- **HL Ratio** - High-low range
- **CO Ratio** - Close-open range

---

## 📈 Metrics Explained

### RMSE (Root Mean Squared Error)
- **Formula:** √(Σ(actual - predicted)² / n)
- **Range:** 0 to ∞
- **Interpretation:** Average prediction error in dollars
- **Target:** Lower is better

### MAE (Mean Absolute Error)
- **Formula:** Σ|actual - predicted| / n
- **Range:** 0 to ∞
- **Interpretation:** Average absolute error
- **Target:** Lower is better

### MAPE (Mean Absolute Percentage Error)
- **Formula:** Σ|actual - predicted| / actual * 100 / n
- **Range:** 0% to 100%+
- **Interpretation:** Percentage error
- **Target:** < 5% is excellent

### R² Score (Coefficient of Determination)
- **Range:** -∞ to 1.0
- **Interpretation:** % variance explained
- **Scores:**
  - R² > 0.85: Excellent
  - R² > 0.70: Very Good
  - R² > 0.50: Good
  - R² < 0.30: Poor
- **Target:** > 0.80 for reliable predictions

---

## 🎓 Expected Results

### Good Model (R² > 0.80)
```
Stock: AAPL, Lookback: 60 days
RMSE: 1.8, MAE: 1.2, MAPE: 2.1%, R²: 0.88
✓ Reliable for short-term forecasts
```

### Average Model (0.60 < R² < 0.80)
```
Stock: Volatile stock
RMSE: 4.5, MAE: 3.2, MAPE: 5.8%, R²: 0.72
⚠ Use with caution, consider longer lookback
```

### Poor Model (R² < 0.60)
```
Stock: Micro-cap, Lookback: 30 days
RMSE: 8.2, MAE: 6.1, MAPE: 12.1%, R²: 0.45
✗ Not reliable, retrain with different parameters
```

---

## 🐛 Troubleshooting

### Model Predictions Look Bad (R² < 0.50)

**Causes & Solutions:**
1. **Not enough data**
   - Increase date_range: "1 Year" → "5 Years"

2. **Stock too volatile**
   - Try stable stocks: JNJ, PG, KO, TCS.NS

3. **Model underfitting**
   - Increase epochs: 100 → 150
   - Increase LSTM units: 100 → 150
   - Increase lookback: 60 → 120

4. **Hyperparameters not tuned**
   - Try different dropout: 0.2 → 0.3
   - Try different batch_size: 32 → 16

### Training is Very Slow

**Solutions:**
1. Use GPU (10x faster) - Setup CUDA
2. Reduce data range: 10 years → 5 years
3. Reduce model size:
   - LSTM units: 100 → 50
   - num_layers: 2 → 1
   - epochs: 100 → 50
4. Increase batch_size: 32 → 64

### "OutOfMemory" Error

**Solutions:**
1. Reduce lookback: 120 → 60
2. Reduce batch_size: 32 → 16
3. Reduce LSTM units: 150 → 50
4. Use smaller date_range

---

## 📚 Learning Resources

### Recommended Reading
- [LSTM Networks Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Time Series Forecasting](https://www.coursera.org/learn/time-series-forecasting)

### Libraries Used
- TensorFlow/Keras - Deep learning
- Streamlit - Dashboard
- yfinance - Stock data
- TA Library - Technical indicators
- Scikit-learn - Preprocessing

---

## ⚠️ Important Disclaimers

1. **Not Financial Advice** - Educational purposes only
2. **Past Performance ≠ Future Results** - Model learns from history
3. **Cannot Predict Black Swans** - Earnings, crashes, geopolitical events
4. **Use with Risk Management** - Always use stop-losses
5. **Consult Financial Advisor** - For real trading decisions

---

## 🚀 Next Steps

1. ✅ Follow QUICKSTART.md for first run
2. ✅ Read MASTER_SUMMARY.md for architecture
3. ✅ Experiment with 5-10 different stocks
4. ✅ Try ideas from IMPROVEMENTS.md

---

**Created:** 2025-12-06  
**Python:** 3.8+  
**TensorFlow:** 2.13+  
**Streamlit:** 1.28+  

**Happy predicting!** 📈✨