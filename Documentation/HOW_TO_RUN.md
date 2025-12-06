# How to Run Trend Master 🚀

Three ways to run this project.

## Way 1: Interactive Streamlit Dashboard (Recommended)

### Command
```bash
streamlit run main.py
```

### What Happens
- Opens interactive web dashboard at `http://localhost:8501`
- 4 tabs for different tasks
- Real-time training progress
- Download forecasts as CSV
- Save trained models

### Best For
- Learning & exploring
- Experimenting with different stocks
- Fine-tuning hyperparameters
- Production use

### Typical Runtime
- First run: 10-15 minutes (download + train)
- Subsequent runs: 5-10 minutes
- Forecast: 1-2 minutes

---

## Way 2: Command-Line Script

### Single Stock Training

**File: `train_single_stock.py`**

```python
#!/usr/bin/env python
"""Train LSTM model for single stock"""

from data_processor import DataProcessor
from model_trainer import LSTMStockPredictor
import config as cfg

# Configuration
TICKER = 'AAPL'
LOOKBACK = 60
FORECAST_DAYS = 30

# Download & prepare data
processor = DataProcessor()
datasets = processor.prepare_datasets(
    TICKER,
    lookback=LOOKBACK,
    start_date='2020-01-01'
)

# Build & train model
predictor = LSTMStockPredictor(
    lookback=LOOKBACK,
    lstm_units=100
)

input_shape = (LOOKBACK, len(datasets['feature_cols']))
predictor.build_model(input_shape)

# Train
predictor.train(
    datasets['X_train'],
    datasets['y_train'],
    datasets['X_val'],
    datasets['y_val'],
    epochs=100,
    batch_size=32
)

# Evaluate
metrics, predictions = predictor.evaluate(
    datasets['X_test'],
    datasets['y_test']
)

print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")
print(f"R² Score: {metrics['R2']:.4f}")

# Save model
predictor.save_model(TICKER)

# Forecast
last_sequence = datasets['raw_df'].iloc[-LOOKBACK:][datasets['feature_cols']].values
forecast = predictor.forecast(
    last_sequence,
    FORECAST_DAYS,
    datasets['scaler'],
    datasets['feature_cols']
)

print(f"\n{FORECAST_DAYS}-Day Forecast:")
for i, price in enumerate(forecast[:5], 1):
    print(f"  Day {i}: ${price:.2f}")
```

### Run It
```bash
python train_single_stock.py
```

---

## Way 3: Batch Processing (Multiple Stocks)

### File: `train_multiple_stocks.py`

```python
#!/usr/bin/env python
"""Train LSTM models for multiple stocks"""

from data_processor import DataProcessor
from model_trainer import LSTMStockPredictor
import pandas as pd
import config as cfg

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TCS.NS', 'INFY.NS']
results = []

for ticker in TICKERS:
    print(f"\nTraining: {ticker}")
    
    try:
        processor = DataProcessor()
        datasets = processor.prepare_datasets(
            ticker,
            lookback=60,
            start_date='2020-01-01'
        )
        
        predictor = LSTMStockPredictor(
            lookback=60,
            lstm_units=100
        )
        
        input_shape = (60, len(datasets['feature_cols']))
        predictor.build_model(input_shape)
        
        predictor.train(
            datasets['X_train'],
            datasets['y_train'],
            datasets['X_val'],
            datasets['y_val'],
            epochs=100,
            verbose=0
        )
        
        metrics, _ = predictor.evaluate(
            datasets['X_test'],
            datasets['y_test']
        )
        
        predictor.save_model(ticker)
        
        results.append({
            'Ticker': ticker,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2'],
            'Status': 'OK'
        })
        
        print(f"✓ {ticker} trained successfully")
        
    except Exception as e:
        print(f"✗ {ticker} failed: {e}")
        results.append({
            'Ticker': ticker,
            'Status': 'FAILED'
        })

# Save summary
results_df = pd.DataFrame(results)
results_df.to_csv('training_summary.csv', index=False)
print("\n✓ Results saved to training_summary.csv")
```

### Run It
```bash
python train_multiple_stocks.py
```

---

## Comparing Methods

| Feature | Streamlit | Script | Batch |
|---------|-----------|--------|-------|
| **UI** | Interactive | Command line | Command line |
| **Visualization** | Yes | No | No |
| **Compare stocks** | Yes (one at a time) | Manual | Automatic |
| **Production** | Good | Good | Best |

---

## Performance Tips

### Speed Up Training

1. **Reduce data range**
   ```python
   datasets = processor.prepare_datasets(
       ticker,
       start_date='2021-01-01'  # Recent 3 years
   )
   ```

2. **Reduce model size**
   ```python
   predictor = LSTMStockPredictor(
       lookback=60,
       lstm_units=50,      # Reduced from 100
       num_lstm_layers=1   # Reduced from 2
   )
   ```

3. **Use GPU** (10x faster)
   - Setup NVIDIA CUDA
   - Install tensorflow-gpu
   - Automatic GPU detection

---

## Typical Runtimes

| Task | Time |
|------|------|
| Download 5-year data | 30 seconds |
| Add indicators | 10 seconds |
| Create sequences | 5 seconds |
| Train model (100 epochs) | 2-3 minutes |
| Evaluate metrics | 10 seconds |
| Forecast (30 days) | 5 seconds |
| **Total** | **3-4 minutes** |

With GPU: **1-2 minutes** total

---

**Choose your method and start predicting!** 🚀