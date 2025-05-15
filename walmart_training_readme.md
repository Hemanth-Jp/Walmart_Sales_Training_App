# Walmart Sales Model Training App

This Streamlit application allows users to train time series models for Walmart sales forecasting using historical data.

## Features

- Upload three CSV files (train.csv, features.csv, stores.csv)
- Train two types of models:
  - Auto ARIMA (using pmdarima)
  - Exponential Smoothing (using statsmodels)
- Customize hyperparameters for both models
- View diagnostic plots showing training performance
- Calculate and display WMAE (Weighted Mean Absolute Error)
- Download trained models for use in the prediction app

## File Structure

```
streamlit_app_training/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/default/        # Default model storage directory
│   ├── auto_arima.pkl
│   └── exponential_smoothing.pkl
└── README.md             # This file
```

## Installation

1. Clone or download this repository
2. Navigate to the app directory:
   ```bash
   cd streamlit_app_training
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser to `http://localhost:8501`

## Usage Guide

### 1. Upload Data Files

Upload three required CSV files:
- **train.csv**: Historical sales data
- **features.csv**: Store features and markdown data
- **stores.csv**: Store information

### 2. Select Model

Choose between:
- **Auto ARIMA**: Automatically determines optimal ARIMA parameters
- **Exponential Smoothing**: Holt-Winters exponential smoothing method

### 3. Configure Hyperparameters

For Auto ARIMA:
- ARIMA parameters (p, q, max_p, max_q)
- Seasonal parameters (P, Q, max_P, max_Q)

For Exponential Smoothing:
- Seasonal periods
- Seasonal component (additive/multiplicative)
- Trend component (additive/multiplicative/none)
- Damped trend option

### 4. Train Model

Click "Start Training" to:
- Train the selected model
- Evaluate on test data
- Display WMAE metric
- Show diagnostic plots

### 5. Save and Download

- Models are automatically saved to `models/default/`
- Use the download button to save trained models

## Model Formats

- **Auto ARIMA**: Saved using joblib as .pkl file
- **Exponential Smoothing**: Saved using statsmodels .save() method as .pkl file

## Data Processing Pipeline

1. **Merge datasets**: Combines train, features, and stores data
2. **Clean data**: Removes non-positive sales, fills missing values
3. **Add holiday indicators**: Super Bowl, Labor Day, Thanksgiving, Christmas
4. **Create time series**: Convert to weekly aggregated data
5. **Difference data**: Make data stationary for better modeling
6. **Split data**: 70% training, 30% testing

## Error Handling

The app includes error handling for:
- Invalid file formats
- Data processing errors
- Model training failures
- File saving issues

## Dependencies

See `requirements.txt` for a complete list of required packages.

## Notes

- Training time varies based on data size and model complexity
- Default models are overwritten each time new models are trained
- App runs locally - not designed for production deployment

## Troubleshooting

1. **Module Import Errors**: Ensure all requirements are installed
2. **File Upload Issues**: Check file format (CSV) and structure
3. **Training Failures**: Verify data quality and try different hyperparameters
4. **Memory Issues**: Large datasets may require more RAM

## Support

For issues or questions, please contact the development team or refer to the source code documentation.

## License

© 2025 Walmart Sales Forecasting Project