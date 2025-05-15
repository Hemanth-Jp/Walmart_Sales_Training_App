# Walmart Sales Model Training App - Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd streamlit_app_training

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### 2. Access the App

Open your browser to: `http://localhost:8501`

## Step-by-Step Guide

### Step 1: Upload Data Files

The app requires three CSV files:

1. **train.csv** - Historical sales data
   - Required columns: Store, Date, Weekly_Sales, IsHoliday
   
2. **features.csv** - Store features and markdown data
   - Required columns: Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday
   
3. **stores.csv** - Store information
   - Required columns: Store, Type, Size

Simply drag and drop or browse to select each file.

### Step 2: Data Verification

After uploading, the app will:
- Merge the three datasets
- Clean the data (remove negative sales, fill missing values)
- Prepare time series data
- Split into 70% training and 30% test data

You'll see information about the dataset size and splits.

### Step 3: Model Selection

Choose between two models:

#### Auto ARIMA
- Automatically finds optimal parameters
- Best for complex seasonal patterns
- May take longer to train

#### Exponential Smoothing
- Simpler, faster to train
- Good for data with clear trends and seasonality

### Step 4: Hyperparameter Configuration

#### For Auto ARIMA:
- **ARIMA Parameters**: p, q values (how many lags to consider)
- **Seasonal Parameters**: P, Q values (seasonal component)
- **Limits**: Max values to search for optimal parameters

#### For Exponential Smoothing:
- **Seasonal Periods**: Number of weeks in a season (default: 20)
- **Seasonal Type**: Additive or multiplicative
- **Trend Type**: Additive, multiplicative, or none
- **Damped**: Whether to use damped trend

### Step 5: Training Process

Click "Start Training" to:
1. Train the model with your data
2. Generate predictions on test data
3. Calculate WMAE (error metric)
4. Display diagnostic plots

### Step 6: Results and Download

After training completes:
- View the WMAE score (lower is better)
- Examine diagnostic plots showing:
  - Training data (blue)
  - Test data (orange)
  - Predictions (green)
- Download the trained model file

### Step 7: Model Storage

Models are automatically saved to:
```
models/default/
├── auto_arima.pkl
└── exponential_smoothing.pkl
```

These can be used directly in the prediction app.

## Best Practices

### Data Preparation
- Ensure data files are properly formatted
- Check for missing dates or stores
- Verify data quality before training

### Model Selection
- For stable data: Use Exponential Smoothing
- For complex patterns: Use Auto ARIMA
- Start with default parameters, then tune if needed

### Hyperparameter Tuning
- Auto ARIMA: Increase max_p/q for better accuracy (slower training)
- Exponential Smoothing: Try different seasonal/trend combinations
- Monitor WMAE to compare model performance

### Performance Monitoring
- Lower WMAE indicates better accuracy
- Check diagnostic plots for overfitting
- Ensure predictions follow actual patterns

## Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Check file format (must be CSV)
   - Verify required columns exist
   - Ensure data types are correct

2. **Training Failures**
   - Reduce model complexity
   - Check for data quality issues
   - Try different hyperparameters

3. **Memory Issues**
   - Process smaller data chunks
   - Reduce max parameter values
   - Restart the application

4. **Slow Training**
   - Decrease max_p/q values
   - Use stepwise search (Auto ARIMA)
   - Consider Exponential Smoothing

### Error Messages

| Error | Solution |
|-------|----------|
| "File format error" | Ensure CSV format with correct columns |
| "Training timeout" | Reduce model complexity |
| "Memory error" | Process smaller datasets |
| "Invalid parameters" | Check hyperparameter ranges |

## Technical Details

### Data Processing Pipeline
1. Load CSV files
2. Merge on Store and Date
3. Clean data (remove negatives, fill NaN)
4. Add holiday indicators
5. Convert to weekly time series
6. Apply differencing for stationarity
7. Split into train/test sets

### Model Specifics

#### Auto ARIMA
- Uses pmdarima.auto_arima()
- Performs grid search for optimal parameters
- Handles seasonal decomposition automatically

#### Exponential Smoothing
- Uses statsmodels.tsa.holtwinters
- Implements Holt-Winters method
- Configurable trend and seasonality

### Performance Metrics
- WMAE = Weighted Mean Absolute Error
- Calculated on differenced data
- Lower values indicate better performance

## Advanced Usage

### Custom Scripts
For automated training:
```python
from app import train_auto_arima, train_exponential_smoothing

# Load your data
df = load_your_data()

# Train model
model = train_auto_arima(df, hyperparams={...})

# Save model
joblib.dump(model, 'custom_model.pkl')
```

### API Integration
The models can be loaded and used programmatically:
```python
import joblib

# Load trained model
model = joblib.load('models/default/auto_arima.pkl')

# Make predictions
predictions = model.predict(n_periods=4)
```

## Support

For additional help:
1. Check the README.md file
2. Review error messages carefully
3. Verify data format and quality
4. Contact support for persistent issues

© 2025 Walmart Sales Forecasting Project