import pandas as pd



def fit_source_battery_sink(T_source, T_battery, T_sink):
    # Calculate the rate of change of T_slab (radiator_return_temp)
    delta_t = 60  # Time interval in seconds (since data is at 1-minute intervals)

    # Compute dT_slab/dt in °C per second
    dT_battery_dt = ((T_battery.diff() / delta_t)
                     .dropna())

    # Ensure all series are aligned with dT_battery_dt
    T_source = T_source.loc[dT_battery_dt.index]
    T_battery = T_battery.loc[dT_battery_dt.index]
    T_sink = T_sink.loc[dT_battery_dt.index]

    # Compute X1 and X2
    Heating = T_source - T_battery
    Consuming = T_sink - T_battery

    from sklearn.linear_model import LinearRegression

    # Prepare the data for regression
    # Remove any potential NaN values due to differencing
    regression_data = pd.DataFrame({
        'dT_dt': dT_battery_dt,
        'Heating': Heating,
        'Consuming': Consuming
    }).dropna()

    # Variables for regression
    Y = regression_data['dT_dt'].values  # Dependent variable
    X = regression_data[['Heating', 'Consuming']].values  # Independent variables

    # Perform linear regression without intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)

    # Extract coefficients
    C_source = model.coef_[0]
    C_sink = model.coef_[1]

    print(f"Estimated C_source: {C_source}")
    print(f"Estimated C_sink: {C_sink}")

    # Calculate R^2 score
    r_squared = model.score(X, Y)
    print(f"R-squared: {r_squared}")
    return C_source, C_sink
