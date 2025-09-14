import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress warnings before importing TensorFlow
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Additional TensorFlow warning suppression
tf.get_logger().setLevel('ERROR')

# ============================================================================
# DATA COLLECTION
# ============================================================================

def get_data(ticker, period='5y', interval='1d'):
    """Fetch historical market data for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

TICKER = 'BTC-USD'
PERIOD = '10y'
INTERVAL = '1d'

print(f"Fetching {TICKER} stock data...")
stock_data = get_data(TICKER, period=PERIOD, interval=INTERVAL)

# Select relevant columns and set index to date
df = stock_data.copy()[['Close', 'Volume']]
df.index = pd.to_datetime(df.index).date

# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================

# Simple moving averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# Exponential moving averages
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# MACD (Moving Average Convergence Divergence)
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Clean up - remove temporary columns and NaN values
df.drop(columns=['EMA_12', 'EMA_26'], inplace=True)

# Remove rows with NaN values (due to moving averages)
df = df.dropna()
print("Features:", df.columns.tolist())

# ============================================================================
# 2. DATA PREPROCESSING FOR LSTM
# ============================================================================

# Select features for the model
feature_columns = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'MACD', 'Signal_Line', 'RSI']
data = df[feature_columns].values

print(f"Data for modeling: {data.shape}")

# Scale the data to 0-1 range (LSTM works better with normalized data)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ============================================================================
# 3. CREATE SEQUENCES FOR LSTM INPUT
# ============================================================================

def create_sequences(data, sequence_length=60):
    """
    Convert time series data into sequences for LSTM training.
    
    LSTM needs sequences of past data to predict the next value.
    For each day, we use the previous 'sequence_length' days as input
    to predict the next day's closing price.
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        # X: sequence of past 'sequence_length' days (all features)
        X.append(data[i-sequence_length:i])
        
        # y: next day's closing price (index 0 is Close price)
        y.append(data[i, 0])
    
    return np.array(X), np.array(y)

# Create sequences (use 60 days to predict next day)
sequence_length = 60
X, y = create_sequences(data_scaled, sequence_length)

print(f"Sequences created:")
print(f"X shape: {X.shape}")  # (samples, time_steps, features)
print(f"y shape: {y.shape}")  # (samples,)
print(f"Each sequence uses {sequence_length} days to predict 1 day")

# ============================================================================
# 4. SPLIT DATA FOR TRAINING AND TESTING
# ============================================================================

# Split data (80% train, 20% test) - don't shuffle for time series!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Important: don't shuffle time series data
)

print(f"Training data: {X_train.shape[0]} sequences")
print(f"Test data: {X_test.shape[0]} sequences")

# ============================================================================
# 5. LSTM MODEL PIPELINE
# ============================================================================

class LSTMModelPipeline:
    """Pipeline for building, training, and using LSTM models for stock prediction"""
    
    def __init__(self, lstm_units=[50, 50], dropout=0.2, optimizer='adam', 
                 loss='mse', metrics=['mae']):
        """
        Initialize LSTM pipeline with model parameters.
        
        Args:
            lstm_units: List of units for each LSTM layer
            dropout: Dropout rate for regularization
            optimizer: Optimizer for training
            loss: Loss function
            metrics: Metrics to track during training
        """
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape: Tuple of (time_steps, features)
        """
        print("Building LSTM model...")
        
        self.model = keras.Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            if i == 0:
                # First LSTM layer with input shape
                return_sequences = len(self.lstm_units) > 1
                self.model.add(layers.LSTM(
                    units, 
                    return_sequences=return_sequences, 
                    input_shape=input_shape
                ))
            else:
                # Subsequent LSTM layers
                return_sequences = i < len(self.lstm_units) - 1
                self.model.add(layers.LSTM(units, return_sequences=return_sequences))
            
            # Add dropout after each LSTM layer
            self.model.add(layers.Dropout(self.dropout))
        
        # Final dense layer for prediction
        self.model.add(layers.Dense(1))
        
        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        
        print("Model architecture:")
        self.model.summary()
        return self
    
    def setup_callbacks(self, early_stopping_patience=20, lr_reduction_patience=10, 
                       lr_reduction_factor=0.5):
        """
        Setup training callbacks for better performance.
        
        Args:
            early_stopping_patience: Epochs to wait before early stopping
            lr_reduction_patience: Epochs to wait before reducing learning rate
            lr_reduction_factor: Factor to reduce learning rate by
        """
        callbacks = [
            # Stop training if validation loss doesn't improve
            keras.callbacks.EarlyStopping(
                patience=early_stopping_patience, 
                restore_best_weights=True,
                monitor='val_loss',
                verbose=1
            ),
            
            # Reduce learning rate if loss plateaus
            keras.callbacks.ReduceLROnPlateau(
                patience=lr_reduction_patience, 
                factor=lr_reduction_factor,
                monitor='val_loss',
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, 
              validation_split=0.2, verbose=1, **callback_kwargs):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Verbosity level
            **callback_kwargs: Additional arguments for callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Training the model...")
        
        # Setup callbacks
        callbacks = self.setup_callbacks(**callback_kwargs)
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self
    
    def predict(self, X, verbose=0):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=verbose)
    
    def get_training_history(self):
        """Get training history for plotting."""
        return self.history.history if self.history else None

# Initialize the LSTM pipeline
lstm_pipeline = LSTMModelPipeline(
    lstm_units=[150, 100, 50],  # LSTM layers
    dropout=0.3,          
    optimizer='adam',     # Adaptive learning rate optimizer
    loss='mse',          # Mean Squared Error for regression
    metrics=['mae']      # Track Mean Absolute Error during training
)

# Build the model
input_shape = (sequence_length, len(feature_columns))
lstm_pipeline.build_model(input_shape)

# Train the model
lstm_pipeline.train(
    X_train, y_train,
    epochs=100,                    # Maximum epochs
    batch_size=64,                 # Process 64 sequences at a time
    validation_split=0.2,          # Use 20% of training data for validation
    verbose=1,                     # Show training progress
    early_stopping_patience=20,    # Wait 20 epochs before early stopping
    lr_reduction_patience=10,      # Wait 10 epochs before reducing learning rate
    lr_reduction_factor=0.5        # Reduce learning rate by half
)

# Get the trained model and history for further use
model = lstm_pipeline.model
history = lstm_pipeline.history

# ============================================================================
# 7. MAKE PREDICTIONS AND EVALUATE
# ============================================================================

# Predict on test data
y_pred_scaled = model.predict(X_test)

# Convert scaled predictions back to original price scale
# Create dummy array to inverse transform (scaler needs all features)
dummy_pred = np.zeros((len(y_pred_scaled), len(feature_columns)))
dummy_pred[:, 0] = y_pred_scaled.flatten()  # Put predictions in Close price column
y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

# Convert actual values back to original scale
dummy_actual = np.zeros((len(y_test), len(feature_columns)))
dummy_actual[:, 0] = y_test
y_actual = scaler.inverse_transform(dummy_actual)[:, 0]

# Calculate performance metrics
mse = mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# ============================================================================
# 8. VISUALIZE RESULTS
# ============================================================================

print("Creating visualizations...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training history - Loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss During Training')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot 2: Training history - MAE
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('Model MAE During Training')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()

# Plot 3: Actual vs Predicted prices
ax3.plot(y_actual, label='Actual Price', alpha=0.7, color='blue')
ax3.plot(y_pred, label='Predicted Price', alpha=0.7, color='red')
ax3.set_title('Actual vs Predicted Stock Prices')
ax3.set_xlabel('Time (Test Period)')
ax3.set_ylabel('Price ($)')
ax3.legend()

# Plot 4: Scatter plot of predictions
ax4.scatter(y_actual, y_pred, alpha=0.5)
ax4.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Price ($)')
ax4.set_ylabel('Predicted Price ($)')
ax4.set_title('Prediction Accuracy Scatter Plot')

plt.tight_layout()
plt.savefig('AAPL_lstm_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. PREDICT FUTURE PRICES
# ============================================================================

print("\n" + "="*50)
print("FUTURE PRICE PREDICTIONS")
print("="*50)

# Get the last sequence for prediction
last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, len(feature_columns))

# Predict next 5 days
future_predictions = []
current_sequence = last_sequence.copy()

for day in range(5):
    # Predict next day
    next_pred_scaled = model.predict(current_sequence, verbose=0)
    
    # Convert to actual price
    dummy = np.zeros((1, len(feature_columns)))
    dummy[0, 0] = next_pred_scaled[0, 0]
    next_pred_actual = scaler.inverse_transform(dummy)[0, 0]
    
    future_predictions.append(next_pred_actual)
    
    # Update sequence for next prediction
    # Shift sequence left and add new prediction
    new_row = current_sequence[0, -1, :].copy()
    new_row[0] = next_pred_scaled[0, 0]  # Update close price
    
    current_sequence = np.roll(current_sequence, -1, axis=1)
    current_sequence[0, -1, :] = new_row

# Display predictions
last_known_price = df['Close'].iloc[-1]
print(f"Last known price: ${last_known_price:.2f}")
print("\nNext 5 days predictions:")

for i, pred_price in enumerate(future_predictions, 1):
    change = pred_price - last_known_price
    change_pct = (change / last_known_price) * 100
    print(f"Day {i}: ${pred_price:.2f} (Change: {change:+.2f}, {change_pct:+.2f}%)")
    last_known_price = pred_price

print(f"\nVisualization saved as 'AAPL_lstm_simple.png'")
print("Model training completed!")

# ============================================================================
# KEY LEARNING POINTS:
# ============================================================================
"""
1. SEQUENCE CREATION: LSTM needs sequences of past data to learn patterns
2. DATA SCALING: MinMaxScaler normalizes data for better training
3. LSTM LAYERS: 
   - return_sequences=True: passes sequences to next LSTM layer
   - return_sequences=False: outputs final prediction
4. DROPOUT: Prevents overfitting by randomly ignoring some neurons
5. CALLBACKS: Early stopping prevents overfitting, LR reduction helps convergence
6. TIME SERIES SPLIT: Never shuffle time series data when splitting
7. INVERSE SCALING: Convert predictions back to original price scale
"""