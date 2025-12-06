# model_trainer.py - LSTM Model Building & Training with Advanced Features

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import config as cfg


class LSTMStockPredictor:
    """
    LSTM-based stock price prediction model.
    """
    
    def __init__(self, lookback=60, lstm_units=100, num_lstm_layers=2, 
                 dropout=0.2, learning_rate=0.001, use_bidirectional=False):
        """
        Initialize LSTM predictor.
        
        Parameters:
        -----------
        lookback : int
            Number of lookback days
        lstm_units : int
            LSTM layer units
        num_lstm_layers : int
            Number of LSTM layers (1-3)
        dropout : float
            Dropout rate (0.1-0.5)
        learning_rate : float
            Adam optimizer learning rate
        use_bidirectional : bool
            Use Bidirectional LSTM
        """
        
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_bidirectional = use_bidirectional
        
        self.model = None
        self.history = None
        self.metrics = None
        
        # Create model directory
        Path(cfg.MODEL_PATH).mkdir(exist_ok=True)
    
    def build_model(self, input_shape):
        """
        Build LSTM model.
        
        Parameters:
        -----------
        input_shape : tuple
            (lookback_days, num_features)
        
        Returns:
        --------
        tf.keras.Model
            Compiled LSTM model
        """
        
        model = Sequential()
        
        # ==================== LSTM LAYERS ====================
        
        for layer_idx in range(self.num_lstm_layers):
            return_sequences = (layer_idx < self.num_lstm_layers - 1)
            
            if self.use_bidirectional:
                model.add(Bidirectional(
                    LSTM(
                        self.lstm_units,
                        return_sequences=return_sequences,
                        activation='relu'
                    ),
                    input_shape=input_shape if layer_idx == 0 else None
                ))
            else:
                model.add(LSTM(
                    self.lstm_units,
                    return_sequences=return_sequences,
                    input_shape=input_shape if layer_idx == 0 else None,
                    activation='relu'
                ))
            
            model.add(Dropout(self.dropout))
        
        # ==================== DENSE LAYERS ====================
        
        model.add(Dense(units=self.lstm_units // 2, activation='relu'))
        model.add(Dropout(self.dropout / 2))
        
        model.add(Dense(units=self.lstm_units // 4, activation='relu'))
        model.add(Dropout(self.dropout / 2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # ==================== COMPILE ====================
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        return model
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return '\n'.join(stringlist)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1):
        """
        Train LSTM model.
        
        Parameters:
        -----------
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data
        epochs : int
            Maximum epochs
        batch_size : int
            Batch size
        verbose : int
            Verbosity level
        
        Returns:
        --------
        tf.keras.callbacks.History
            Training history
        """
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=cfg.LSTM_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=cfg.LSTM_CONFIG['reduce_lr_factor'],
            patience=cfg.LSTM_CONFIG['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        print(f"🤖 Training LSTM model...")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
        print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        print("✓ Training complete")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Parameters:
        -----------
        X_test, y_test : np.ndarray
            Test data
        
        Returns:
        --------
        dict
            Evaluation metrics (RMSE, MAE, MAPE, R²)
        np.ndarray
            Test predictions
        """
        
        predictions = self.predict(X_test).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / (np.abs(y_test) + 1e-8))) * 100
        
        # R² Score
        r2 = r2_score(y_test, predictions)
        
        self.metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2),
        }
        
        return self.metrics, predictions
    
    def forecast(self, last_sequence, forecast_days, scaler, feature_cols):
        """
        Generate future predictions (multi-step ahead).
        
        Parameters:
        -----------
        last_sequence : np.ndarray
            Last lookback_days of scaled data (lookback, n_features)
        forecast_days : int
            Number of days to forecast
        scaler : MinMaxScaler
            Fitted scaler for inverse transform
        feature_cols : list
            Feature column names
        
        Returns:
        --------
        np.ndarray
            Future predictions in original scale (forecast_days,)
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        current_sequence = last_sequence.copy()
        predictions_scaled = []
        
        print(f"🔮 Generating {forecast_days}-day forecast...")
        
        for step in range(forecast_days):
            # Predict next value
            batch = current_sequence.reshape(1, self.lookback, current_sequence.shape[1])
            next_pred = self.model.predict(batch, verbose=0)[0, 0]
            predictions_scaled.append(next_pred)
            
            # Update sequence for next iteration
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred  # Update Close price
            current_sequence = np.vstack((current_sequence[1:], new_row))
        
        # Inverse transform to original scale
        predictions_scaled = np.array(predictions_scaled)
        predictions_original = self._inverse_transform_predictions(
            predictions_scaled, scaler, feature_cols
        )
        
        print(f"✓ Forecast generated")
        
        return predictions_original
    
    def _inverse_transform_predictions(self, predictions, scaler, feature_cols):
        """Inverse transform scaled predictions to original scale."""
        
        dummy = np.zeros((len(predictions), len(feature_cols)))
        dummy[:, 0] = predictions
        
        original_scale = scaler.inverse_transform(dummy)
        
        return original_scale[:, 0]
    
    def save_model(self, ticker, version='v1'):
        """
        Save trained model.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        version : str
            Model version
        
        Returns:
        --------
        str
            Path to saved model
        """
        
        if self.model is None:
            raise ValueError("No model to save.")
        
        filepath = os.path.join(cfg.MODEL_PATH, f'{ticker}_{version}.h5')
        self.model.save(filepath)
        
        print(f"💾 Model saved: {filepath}")
        
        return filepath
    
    def load_model(self, ticker, version='v1'):
        """
        Load pre-trained model.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        version : str
            Model version
        
        Returns:
        --------
        tf.keras.Model
            Loaded model
        """
        
        filepath = os.path.join(cfg.MODEL_PATH, f'{ticker}_{version}.h5')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        
        self.model = tf.keras.models.load_model(filepath)
        
        print(f"📂 Model loaded: {filepath}")
        
        return self.model
    
    def get_training_history(self):
        """Get training history as DataFrame."""
        
        if self.history is None:
            return None
        
        df = pd.DataFrame(self.history.history)
        return df
