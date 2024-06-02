"""
Dictionary containing asset information and investment parameters.

The `assets` dictionary holds configuration and parameters for each asset, identified by its ticker symbol.
Each asset has several properties such as the quantity held, the percentage of total cash to invest,
technical indicators, and stop-loss settings.
"""

assets = {
    "TQQQ": {
        "qty": 0,  # Quantity of the asset currently held
        "percent_to_invest": 0.1,  # Percentage of the total cash to invest in the asset
        "recommendation": "",  # Recommendation status for the asset (e.g., "buy", "hold", "sell")
        "Hull_moving_average": 50,  # Period for the Hull Moving Average indicator
        "average_true_range": 60,  # Period for the Average True Range indicator
        "channel_stop_loss": 0.97,  # Multiplier for the channel stop loss, expressed as a percentage
        "disaster_stop_loss": 0.9,  # Multiplier for the disaster stop loss, expressed as a percentage
        "buy_lag": 6,  # Lag period for buying the asset
        "range_stop_loss_multiple": 2.5,  # Multiplier for the range stop loss
    }
}
