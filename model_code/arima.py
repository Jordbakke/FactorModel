import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def apply_arima(group, time_series_column, entity_col, p=5, q=5, d=0):
    # Check stationarity with Augmented Dickey-Fuller test
    try:
        data_to_fit = group[time_series_column]
        # Fit ARIMA model
        model = ARIMA(data_to_fit, order=(p, q, d))
        fitted_model = model.fit()
        group['fitted'] = fitted_model.fittedvalues
        
        # Forecasting future values
        group['forecast'] = fitted_model.forecast(steps=5)  # Forecast 5 steps ahead
        
        # Calculate MSE
        mse = mean_squared_error(group[time_series_column], group['fitted'])
        print(f"Mean Squared Error: {mse}")
        
        return group, mse
    
    except Exception as e:
        print("Error processing group: ", group[entity_col].iloc[0])
        return None, -1

def plot_fitted_vs_actual(group, mse, time_series_column, time_column, entity_col):
    """
    Plots the actual vs fitted values for the ARIMA model using a time column for the x-axis.
    """
    plt.figure(figsize=(10, 6))
    group[time_column] = group[time_column].astype(str)
    # Plot actual values using the time column for the x-axis
    plt.plot(group[time_column], group[time_series_column], label="Actual", color='blue')
    
    # Plot fitted values using the time column for the x-axis
    plt.plot(group[time_column], group['fitted'], label="Fitted", linestyle='--', alpha=0.8, color='orange')
    
    plt.legend()
    plt.title(f"Actual vs Fitted Values: {group[entity_col].iloc[0]} MSE: {mse}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
