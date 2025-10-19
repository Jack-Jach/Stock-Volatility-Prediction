# Stock Volatility Prediction

## Project Description

This project aims to predict stock volatility using machine learning. It uses historical stock data to train a model that can forecast future volatility. The implementation is done in a Jupyter Notebook (`stock-volatility-prediction.ipynb`).

## Dataset

The primary dataset used for training the model is `stock_volatility_labeled.csv`. The project also includes `stock_volatility_prediction.csv`, likely for a Kaggle competition.

## Methodology

The project follows these steps:

1.  **Data Loading**: The data is loaded from `stock_volatility_labeled.csv`.
2.  **Feature Engineering**: New features are created from the existing data, including:
    *   Date-based features (Year, Month, Day of Week)
    *   Interaction features (e.g., Volume * Return)
    *   Lagged features for volatility and returns
    *   Rolling statistics (mean and standard deviation)
    *   Technical indicators like Simple Moving Averages (SMA)
3.  **Data Preprocessing**: Numerical features are scaled using `RobustScaler`.
4.  **Model Training and Tuning**:
    *   A `RandomForestRegressor` model is trained to predict the next period's volatility.
    *   Hyperparameter tuning is performed using `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation to find the best model parameters.
5.  **Forecasting**: The best model is used to forecast volatility for a future period (the "23rd month").

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib scikit-learn scipy jupyter
    ```
2.  **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook stock-volatility-prediction.ipynb
    ```

## Results

The project outputs a `pred_value.csv` file containing the predicted volatility for each stock.
