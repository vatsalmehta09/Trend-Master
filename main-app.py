# main.py - Complete Streamlit Interactive Stock Predictor Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import config as cfg
from data_processor import DataProcessor
from model_trainer import LSTMStockPredictor
from technical_indicators import validate_indicators

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="📈 TrendMaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .good {color: #00d084;}
    .warning {color: #ffa500;}
    .danger {color: #ff4757;}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

if 'predictor' not in st.session_state:
    st.session_state.predictor = None

if 'trained_data' not in st.session_state:
    st.session_state.trained_data = {}

if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("⚙️ TrendMaster Configuration")
    
    # Market selection
    st.subheader("📍 Market Selection")
    market = st.radio("Select Market:", ["🇮🇳 India (NSE)", "🇺🇸 USA (NYSE/NASDAQ)"])
    
    # Stock selection
    st.subheader("📊 Stock Selection")
    
    if "India" in market:
        categories = list(cfg.INDIAN_STOCKS.keys())
        selected_category = st.selectbox("Category:", categories)
        stocks = cfg.INDIAN_STOCKS[selected_category]
        stock_names = list(stocks.keys())
    else:
        categories = list(cfg.US_STOCKS.keys())
        selected_category = st.selectbox("Category:", categories)
        stocks = cfg.US_STOCKS[selected_category]
        stock_names = list(stocks.keys())
    
    selected_ticker = st.selectbox(
        "Stock:",
        stock_names,
        format_func=lambda x: f"{x} - {stocks[x]}"
    )
    
    # Custom ticker input
    custom_ticker = st.text_input("Or enter custom ticker (e.g., AAPL):", "")
    if custom_ticker:
        selected_ticker = custom_ticker.upper()
    
    # Model parameters
    st.subheader("🧠 Model Parameters")
    
    lookback = st.slider(
        "Lookback Period (days):",
        min_value=30,
        max_value=180,
        value=60,
        step=10,
        help="How many days of history to use for learning"
    )
    
    lstm_units = st.select_slider(
        "LSTM Units:",
        options=[50, 100, 150],
        value=100,
        help="Number of neurons in LSTM layers"
    )
    
    num_lstm_layers = st.select_slider(
        "LSTM Layers:",
        options=[1, 2, 3],
        value=2,
        help="Number of LSTM layers"
    )
    
    dropout = st.slider(
        "Dropout Rate:",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Dropout probability to prevent overfitting"
    )
    
    # Training parameters
    st.subheader("🎓 Training Parameters")
    
    epochs = st.slider(
        "Epochs:",
        min_value=50,
        max_value=300,
        value=100,
        step=10,
        help="Maximum training epochs"
    )
    
    batch_size = st.select_slider(
        "Batch Size:",
        options=[16, 32, 64],
        value=32,
        help="Number of samples per gradient update"
    )
    
    # Forecast parameters
    st.subheader("🔮 Forecast Parameters")
    
    forecast_days = st.select_slider(
        "Forecast Period (days):",
        options=[30, 60, 90],
        value=30,
        help="How many days ahead to predict"
    )
    
    # Date range
    st.subheader("📅 Data Range")
    
    date_range = st.select_slider(
        "Historical Data:",
        options=["1 Year", "3 Years", "5 Years", "10 Years"],
        value="5 Years",
        help="Amount of historical data to download"
    )
    
    date_map = {
        "1 Year": (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'),
        "3 Years": (datetime.today() - timedelta(days=365*3)).strftime('%Y-%m-%d'),
        "5 Years": (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d'),
        "10 Years": (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d'),
    }
    
    start_date = date_map[date_range]
    end_date = datetime.today().strftime('%Y-%m-%d')

# ==================== MAIN CONTENT ====================

st.title("📈 TrendMaster - Advanced Stock Prediction")
st.markdown(f"**Selected Stock:** `{selected_ticker}` | **Market:** {market} | **Data Range:** {date_range}")

# ==================== TABS ====================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & Visualization",
    "🤖 Model Training",
    "🔮 Predictions & Forecast",
    "📉 Analytics"
])

# ==================== TAB 1: Data & Visualization ====================

with tab1:
    st.header("Historical Stock Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("📥 Download Data", key="download_btn", help="Download historical stock data"):
            with st.spinner(f"Downloading {selected_ticker} data..."):
                try:
                    data_dict = st.session_state.data_processor.download_stock_data(
                        selected_ticker,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if selected_ticker in data_dict:
                        st.session_state.raw_data = data_dict[selected_ticker]
                        st.success(f"✓ Downloaded {len(st.session_state.raw_data)} records")
                    else:
                        st.error("Failed to download data")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Display data
    if 'raw_data' in st.session_state:
        df = st.session_state.raw_data
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        with col2:
            price_change = df['Close'].iloc[-1] - df['Close'].iloc[-5]
            st.metric("5-Day Change", f"${price_change:.2f}")
        with col3:
            st.metric("52-Week High", f"${df['High'].max():.2f}")
        with col4:
            st.metric("52-Week Low", f"${df['Low'].min():.2f}")
        
        # Candlestick chart
        st.subheader("Candlestick Chart with Volume")
        
        fig = go.Figure(data=[
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            )
        ])
        
        fig.update_layout(
            title=f"{selected_ticker} - OHLC Chart",
            yaxis_title="Stock Price (USD)",
            template="plotly_dark",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_vol = go.Figure(data=[
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker=dict(color='rgba(0, 100, 255, 0.6)')
            )
        ])
        
        fig_vol.update_layout(
            title=f"{selected_ticker} - Daily Volume",
            yaxis_title="Volume",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Display raw data
        with st.expander("View Raw Data"):
            st.dataframe(df.tail(20), use_container_width=True)
        
        # Add indicators
        with st.expander("Add Technical Indicators"):
            if st.button("Calculate Indicators"):
                with st.spinner("Calculating indicators..."):
                    df_with_indicators = st.session_state.data_processor.add_indicators(df)
                    st.session_state.df_with_indicators = df_with_indicators
                    
                    if validate_indicators(df_with_indicators):
                        st.success("✓ Indicators calculated successfully")
                        
                        # Plot indicators
                        fig_sma = go.Figure()
                        
                        fig_sma.add_trace(go.Scatter(
                            x=df_with_indicators['Date'],
                            y=df_with_indicators['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='white', width=2)
                        ))
                        
                        fig_sma.add_trace(go.Scatter(
                            x=df_with_indicators['Date'],
                            y=df_with_indicators['SMA_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(dash='dash')
                        ))
                        
                        fig_sma.add_trace(go.Scatter(
                            x=df_with_indicators['Date'],
                            y=df_with_indicators['SMA_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(dash='dash')
                        ))
                        
                        fig_sma.update_layout(
                            title="Price with Moving Averages",
                            template="plotly_dark",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_sma, use_container_width=True)

# ==================== TAB 2: Model Training ====================

with tab2:
    st.header("Model Training & Evaluation")
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please download data first (Tab 1)")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🤖 Train Model", key="train_btn", help="Train LSTM model on downloaded data"):
                with st.spinner("Preparing data and training model..."):
                    try:
                        # Prepare datasets
                        datasets = st.session_state.data_processor.prepare_datasets(
                            selected_ticker,
                            lookback=lookback,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        st.session_state.trained_data[selected_ticker] = datasets
                        
                        # Build model
                        predictor = LSTMStockPredictor(
                            lookback=lookback,
                            lstm_units=lstm_units,
                            num_lstm_layers=num_lstm_layers,
                            dropout=dropout
                        )
                        
                        input_shape = (lookback, len(datasets['feature_cols']))
                        predictor.build_model(input_shape)
                        
                        # Display model summary
                        st.subheader("Model Architecture")
                        model_summary = predictor.get_model_summary()
                        st.code(model_summary, language="text")
                        
                        # Train
                        history = predictor.train(
                            datasets['X_train'],
                            datasets['y_train'],
                            datasets['X_val'],
                            datasets['y_val'],
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0
                        )
                        
                        # Evaluate
                        metrics, predictions = predictor.evaluate(
                            datasets['X_test'],
                            datasets['y_test']
                        )
                        
                        st.session_state.predictor = predictor
                        st.session_state.metrics = metrics
                        
                        # Display metrics
                        st.success("✓ Training complete!")
                        
                        st.subheader("Model Performance Metrics")
                        
                        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                        
                        with col_m1:
                            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        with col_m2:
                            st.metric("MAE", f"{metrics['MAE']:.4f}")
                        with col_m3:
                            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                        with col_m4:
                            st.metric("R² Score", f"{metrics['R2']:.4f}")
                        with col_m5:
                            quality = "Excellent" if metrics['R2'] > 0.85 else "Good" if metrics['R2'] > 0.70 else "Fair" if metrics['R2'] > 0.50 else "Poor"
                            st.metric("Quality", quality)
                        
                        # Save model
                        if st.button("💾 Save Model"):
                            predictor.save_model(selected_ticker)
                            st.success("✓ Model saved successfully")
                        
                        # Plot training history
                        st.subheader("Training History")
                        
                        df_history = predictor.get_training_history()
                        
                        fig_loss = go.Figure()
                        
                        fig_loss.add_trace(go.Scatter(
                            y=df_history['loss'],
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='blue')
                        ))
                        
                        fig_loss.add_trace(go.Scatter(
                            y=df_history['val_loss'],
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_loss.update_layout(
                            title="Training & Validation Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss (MSE)",
                            template="plotly_dark",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                        # Actual vs Predicted on Test Set
                        st.subheader("Test Set: Actual vs Predicted")
                        
                        # Inverse transform
                        y_test_original = st.session_state.data_processor.inverse_transform(
                            datasets['y_test'],
                            datasets['scaler'],
                            datasets['feature_cols']
                        )
                        
                        predictions_original = st.session_state.data_processor.inverse_transform(
                            predictions,
                            datasets['scaler'],
                            datasets['feature_cols']
                        )
                        
                        fig_pred = go.Figure()
                        
                        fig_pred.add_trace(go.Scatter(
                            y=y_test_original,
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_pred.add_trace(go.Scatter(
                            y=predictions_original,
                            mode='lines',
                            name='Predicted Price',
                            line=dict(color='red', dash='dash', width=2)
                        ))
                        
                        fig_pred.update_layout(
                            title="Test Set: Actual vs Predicted Prices",
                            xaxis_title="Time Steps",
                            yaxis_title="Price (USD)",
                            template="plotly_dark",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during training: {e}")

# ==================== TAB 3: Predictions & Forecast ====================

with tab3:
    st.header("Future Price Predictions")
    
    if st.session_state.predictor is None:
        st.warning("⚠️ Please train a model first (Tab 2)")
    else:
        if st.button("🔮 Generate Forecast", key="forecast_btn"):
            with st.spinner(f"Generating {forecast_days}-day forecast..."):
                try:
                    # Get last sequence
                    datasets = st.session_state.trained_data[selected_ticker]
                    last_sequence = datasets['raw_df'].iloc[-lookback:][datasets['feature_cols']].values
                    
                    # Normalize last sequence
                    last_sequence_scaled, _ = st.session_state.data_processor.normalize_data(
                        datasets['raw_df'].iloc[-lookback:],
                        datasets['feature_cols'],
                        fit=False
                    )
                    
                    # Forecast
                    forecast = st.session_state.predictor.forecast(
                        last_sequence_scaled,
                        forecast_days,
                        datasets['scaler'],
                        datasets['feature_cols']
                    )
                    
                    st.session_state.forecasts[selected_ticker] = forecast
                    
                    # Create forecast dates
                    last_date = datasets['raw_df']['Date'].iloc[-1]
                    forecast_dates = [
                        last_date + timedelta(days=i+1)
                        for i in range(forecast_days)
                    ]
                    
                    # Create DataFrame
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Predicted_Price': forecast
                    })
                    
                    st.success("✓ Forecast generated!")
                    
                    # Display forecast table
                    st.subheader("Forecast Results")
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Plot forecast
                    st.subheader("Historical vs Forecast")
                    
                    # Get last 100 days of history
                    historical = datasets['raw_df'].tail(100).copy()
                    
                    fig_forecast = go.Figure()
                    
                    # Historical prices
                    fig_forecast.add_trace(go.Scatter(
                        x=historical['Date'],
                        y=historical['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Predicted_Price'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='dash', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig_forecast.update_layout(
                        title=f"{selected_ticker} - {forecast_days} Day Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template="plotly_dark",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast statistics
                    st.subheader("Forecast Statistics")
                    
                    current_price = historical['Close'].iloc[-1]
                    forecast_avg = forecast_df['Predicted_Price'].mean()
                    forecast_min = forecast_df['Predicted_Price'].min()
                    forecast_max = forecast_df['Predicted_Price'].max()
                    forecast_change = forecast_df['Predicted_Price'].iloc[-1] - current_price
                    forecast_change_pct = (forecast_change / current_price) * 100
                    
                    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                    
                    with col_s1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col_s2:
                        st.metric("Forecast Average", f"${forecast_avg:.2f}")
                    with col_s3:
                        st.metric("Forecast Min", f"${forecast_min:.2f}")
                    with col_s4:
                        st.metric("Forecast Max", f"${forecast_max:.2f}")
                    with col_s5:
                        color = "🟢" if forecast_change > 0 else "🔴"
                        st.metric(
                            f"End Forecast {color}",
                            f"${forecast_df['Predicted_Price'].iloc[-1]:.2f}",
                            delta=f"{forecast_change_pct:+.2f}%"
                        )
                    
                    # Download forecast
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast as CSV",
                        data=csv,
                        file_name=f"{selected_ticker}_forecast_{datetime.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {e}")

# ==================== TAB 4: Analytics ====================

with tab4:
    st.header("Advanced Analytics")
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please download data first (Tab 1)")
    else:
        df = st.session_state.raw_data
        
        # Daily returns
        st.subheader("Daily Returns Analysis")
        
        daily_returns = df['Close'].pct_change() * 100
        
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        
        with col_a1:
            st.metric("Avg Daily Return", f"{daily_returns.mean():.3f}%")
        with col_a2:
            st.metric("Volatility", f"{daily_returns.std():.3f}%")
        with col_a3:
            st.metric("Max Return", f"{daily_returns.max():.3f}%")
        with col_a4:
            st.metric("Min Return", f"{daily_returns.min():.3f}%")
        
        # Returns distribution
        fig_returns = px.histogram(
            x=daily_returns,
            nbins=50,
            title="Distribution of Daily Returns",
            labels={'x': 'Daily Return (%)', 'y': 'Frequency'},
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Price statistics
        st.subheader("Price Statistics")
        
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        
        with col_p1:
            st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        with col_p2:
            st.metric("Average Price", f"${df['Close'].mean():.2f}")
        with col_p3:
            st.metric("High", f"${df['High'].max():.2f}")
        with col_p4:
            st.metric("Low", f"${df['Low'].min():.2f}")
        with col_p5:
            price_range = df['High'].max() - df['Low'].min()
            st.metric("Price Range", f"${price_range:.2f}")
        
        # Rolling volatility
        st.subheader("30-Day Rolling Volatility")
        
        rolling_vol = daily_returns.rolling(window=30).std()
        
        fig_vol = go.Figure()
        
        fig_vol.add_trace(go.Scatter(
            y=rolling_vol,
            mode='lines',
            name='30-Day Volatility',
            fill='tozeroy',
            line=dict(color='orange')
        ))
        
        fig_vol.update_layout(
            title="30-Day Rolling Volatility",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>📈 Stock Price Predictor | LSTM Neural Networks | Data: Yahoo Finance</p>
    <p>⚠️ Disclaimer: Predictions are for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
