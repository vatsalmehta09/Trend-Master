# Master Summary: TrendMaster Architecture & Learning Path 🏗️

Complete overview of the project architecture and 4-week learning plan.

## 🎯 Project Goals

1. Predict stock prices using LSTM neural networks
2. Support 65+ stocks (India + USA)
3. Provide interactive dashboard
4. Deliver 10+ technical indicators
5. Enable 30/60/90 day forecasting

---

## 📊 Data Pipeline

```
1. Download OHLCV → Yahoo Finance
2. Add Indicators → 10+ technical indicators
3. Normalize → MinMaxScaler (0-1 range)
4. Create Sequences → Lookback window (60 days)
5. Split Data → 80% train, 10% val, 10% test
6. Train LSTM → 2 layers, 100 units, dropout 0.2
7. Evaluate → RMSE, MAE, MAPE, R²
8. Forecast → 30/60/90 days ahead
9. Display → Streamlit dashboard + CSV export
```

---

## 🧠 LSTM Architecture

```
Input: (batch_size, 60, 21)
  [samples, lookback_days, features]
    ↓
LSTM Layer 1: 100 units → Dropout(0.2)
    ↓
LSTM Layer 2: 50 units → Dropout(0.2)
    ↓
Dense: 50 units → Dropout(0.1)
    ↓
Dense: 25 units
    ↓
Output: 1 unit (Close Price)
```

---

## 📈 Metrics Explained

### RMSE (Root Mean Squared Error)
- **Meaning:** Average prediction error in dollars
- **Example:** RMSE=2.5 means error is ±$2.50
- **Target:** Lower is better

### MAE (Mean Absolute Error)
- **Meaning:** Average absolute error
- **Example:** MAE=1.8 means typically off by ±$1.80
- **Target:** Lower is better

### MAPE (Mean Absolute Percentage Error)
- **Meaning:** Percentage error (stock-independent)
- **Example:** MAPE=3.2% means 3.2% average error
- **Target:** < 5% is excellent

### R² Score
- **Meaning:** % of variance explained
- **R² = 0.88:** Model explains 88% of price movement
- **Target:** > 0.80 is reliable

---

## 🎓 4-Week Learning Path

### Week 1: Foundations (Data & Indicators)

**Day 1-2:** Stock Market Basics
- Read INDIAN_STOCKS.md & USA_STOCKS.md
- Understand NSE vs NYSE markets
- Download a few tickers manually

**Day 3-4:** Technical Indicators
- Study technical_indicators.py
- Understand: SMA, EMA, RSI, MACD, Bollinger Bands
- Run Tab 1 in Streamlit, view indicators

**Day 5-7:** Data Processing
- Study data_processor.py
- Create sequences, normalization
- Modify lookback period, observe impact

### Week 2: LSTM & Time Series

**Day 1-2:** LSTM Basics
- Read about RNN, LSTM, gates, sequences
- Visit: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

**Day 3-4:** Model Building
- Study model_trainer.py
- Build & train a model (Tab 2)
- Understand layers, activation, loss

**Day 5-7:** Metrics & Evaluation
- Understand RMSE, MAE, MAPE, R²
- Interpret what metrics mean
- Train same stock multiple times

### Week 3: Streamlit & Dashboard

**Day 1-2:** Streamlit Basics
- Study main.py code
- Understand tabs, state management, widgets

**Day 3-4:** Interactive Features
- Add custom visualizations
- Modify dashboard layout

**Day 5-7:** Multiple Stocks
- Train 5 stocks, compare metrics
- Create comparison table
- Identify best performers

### Week 4: Experimentation

**Day 1-2:** Hyperparameter Tuning
- Change lookback, LSTM units, epochs
- Track how each change affects R²
- Document best configurations

**Day 3-4:** Feature Engineering
- Add custom technical indicators
- Measure impact on performance

**Day 5-7:** Advanced Topics
- Read IMPROVEMENTS.md
- Plan which improvements to implement

---

## 🔧 Key Concepts

| Concept | Definition |
|---------|-----------|
| **Time Series** | Sequential data ordered by time (stock prices) |
| **LSTM** | RNN variant that remembers long-term patterns |
| **Lookback** | How much historical data to use (60 days) |
| **Sequences** | Fixed-length windows of data for training |
| **Normalization** | Scale data to 0-1 range for faster learning |
| **Train/Val/Test** | Data split: learn/tune/evaluate |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau |
| **Forecast** | Predict future prices using trained model |

---

## 🎯 Troubleshooting Decision Tree

**Problem: Model predictions bad (R² < 0.50)**
- Check: How much data? (Increase if < 2 years)
- Check: Is stock volatile? (Try stable stock first)
- Check: Model parameters? (Increase epochs, LSTM units)
- Check: Indicators? (Validate they're calculating correctly)

**Problem: Training slow**
- Check: Data size? (Reduce date range)
- Check: Model size? (Reduce units, layers)
- Check: Have GPU? (Use CUDA for 10x speedup)
- Check: Batch size? (Increase if too small)

**Problem: Memory error**
- Check: Reduce lookback (120 → 60)
- Check: Reduce epochs (100 → 50)
- Check: Reduce batch_size (32 → 16)
- Check: Reduce LSTM units (100 → 50)

---

## 📚 Resources for Learning

- TensorFlow.org - Official documentation
- Streamlit.io - Dashboard framework
- Investopedia - Finance concepts
- Medium - Articles on ML & finance

---

**Start with QUICKSTART.md and follow the 4-week path for best results!** 🚀