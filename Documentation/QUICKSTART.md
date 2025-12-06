# Quick Start Guide ⚡

Get the stock predictor running in **5-10 minutes**.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Step 1: Setup (2 minutes)

### On Windows:
```bash
# Create project folder
mkdir trendmaster
cd trendmaster

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### On macOS/Linux:
```bash
mkdir stock-predictor
cd stock-predictor

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Step 2: Run Dashboard (1 minute)

```bash
streamlit run main.py
```

Your dashboard opens at `http://localhost:8501` 🚀

## Step 3: First Prediction (3-5 minutes)

### Workflow:

1. **Configure** (left sidebar)
   - Market: Select India or USA
   - Stock: Pick a ticker (e.g., TCS.NS or AAPL)
   - Model: Keep defaults for first run

2. **Download Data** (Tab 1: Data & Visualization)
   - Click "📥 Download Data"
   - Wait for ~1000 records to load

3. **Train Model** (Tab 2: Model Training)
   - Click "🤖 Train Model"
   - Watch training progress (1-3 minutes)
   - Review metrics (RMSE, MAE, MAPE, R²)

4. **Forecast** (Tab 3: Predictions & Forecast)
   - Click "🔮 Generate Forecast"
   - See 30-day price predictions
   - Download results as CSV

5. **Save Model**
   - After training, click "💾 Save Model"
   - Reuse model anytime without retraining

## Recommended First Stocks

### India (Stable for learning):
- **TCS.NS** - IT giant, stable, lots of data
- **SBIN.NS** - Banking sector
- **HDFCBANK.NS** - Consistent data

### USA (Reliable):
- **AAPL** - Apple, lots of data, stable
- **MSFT** - Microsoft, large cap
- **JNJ** - Johnson & Johnson, defensive

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "OutOfMemory" during training
- Reduce LSTM Units: 100 → 50
- Reduce Batch Size: 32 → 16
- Reduce Lookback: 60 → 30

### "Failed to download data"
- Check ticker spelling
- Ensure stock is active on exchange
- Try well-known ticker first (AAPL, TCS.NS)

### Model predictions look bad (R² < 0.50)
- Increase data range: 1 Year → 5 Years
- Increase Lookback: 60 → 90
- Train longer: 100 → 150 epochs
- Try more stable stocks first

## Next Steps

- ✅ **Explore indicators**: Tab 1 → "Add Technical Indicators"
- ✅ **Compare stocks**: Train multiple stocks, compare metrics
- ✅ **Fine-tune model**: Adjust hyperparameters, see impact
- ✅ **Review documentation**: See other .md files

---

**Happy predicting!** 📈✨