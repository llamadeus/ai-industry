import logging

import numpy as np
import pandas as pd
import pandas.testing as pdt
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from pandas._typing import InterpolateOptions
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
data_folder = '../resources/dataset'
anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize = (9, 3)
best_imputation_method = 'linear'
best_window_length = 10
best_aggregation_length = 50


def load_dataset(filename):
    """
    Load the dataset from a CSV file.

    Parameters:
        filename (str): The name of the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """

    # Load the input data
    data_path = f'{data_folder}/{filename}'
    raw_data = pd.read_csv(data_path)
    raw_data['Time'] = pd.to_datetime(raw_data['Time'])
    raw_data.set_index('Time', inplace=True)

    # The index was stored as an unnamed column
    return raw_data.drop(columns=["Unnamed: 0"])


def find_longest_segment_max_trues(series, max_true_count):
    """
    Find the longest segment in a Series with at most `max_true_count` missing values.

    Parameters:
        series (pd.Series): The input series with potential missing values.
        max_true_count (int): Maximum allowed missing values in a segment.

    Returns:
        tuple: (start_index, end_index) of the best segment.
    """
    # Given an array M if the indices of the missing values
    # Extending any segment up to the the next missing value adds exactly one missing value.
    # Therefore, for any index `i`, any segment `series[M[i]:M[i+max_true_count]] contains `max_true_count` missing values.
    # As such, we look for the longest such segment.
    indices_of_missing_values = np.array(
        [-1, *np.asarray(series).nonzero()[0], len(series)])
    # The number must not be larger than the total number of Nas
    max_true_count = min(max_true_count, len(indices_of_missing_values) - 2)
    segment_lengths_plus_1 = indices_of_missing_values[max_true_count +
                                                       1:] - indices_of_missing_values[:-max_true_count-1]
    location_where_we_find_the_index_where_the_segment_starts = segment_lengths_plus_1.argmax()
    bounds_as_array = [1, -1] + indices_of_missing_values[np.array(
        [0, max_true_count+1]) + location_where_we_find_the_index_where_the_segment_starts]
    return tuple(map(int, bounds_as_array))


def find_best_segment_in_series(series, max_missing):
    return find_longest_segment_max_trues(series.isna(), max_missing)


def calculate_true_series(series):
    # Find the indices where segments of True start and end
    logging.debug(series)
    padded_array = np.pad(series, (1, 1), constant_values=False)
    diff = np.diff(padded_array.astype(int))
    starts = (diff == 1).nonzero()[0]
    ends = (diff == -1).nonzero()[0]

    # Calculate lengths of segments
    return np.column_stack((starts, ends - starts))


def calculate_na_series(series):
    """
    Count continuous segments of True values for all lengths in a boolean array.

    Parameters:
        series (pd.Series): A 1D boolean array.

    Returns:
        dict: A dictionary where keys are segment lengths and values are counts.
    """
    return calculate_true_series(series.isna())


def test_count_non_nan_segments():
    # Define test cases as a list of (input, expected_output) tuples
    test_cases = [
        # Format: (Input Series, Expected Output (zipped starts and lengths))
        # Case 1: Single segment of NAs
        (pd.Series([1, None, None, None, 2]), np.array([(1, 3)])),
        # Case 2: Multiple segments of NAs
        (pd.Series([1, None, None, 2, None, None, None, 3, None]),
         np.array([(1, 2), (4, 3), (8, 1)])),
        # Case 3: No NAs
        (pd.Series([1, 2, 3]), np.array([])),
        # Case 4: Entire Series is NAs
        (pd.Series([None, None, None, None]), np.array([(0, 4)])),
        # Case 5: Alternating values
        (pd.Series([None, 1, None, 2, None]),
         np.array([(0, 1), (2, 1), (4, 1)])),
        # Case 6: Empty array
        (pd.Series([]), np.array([])),
        # Case 7: Mixed edge case
        (pd.Series([None, None, 1, None, 2, 3, None, None, None]),
         np.array([(0, 2), (3, 1), (6, 3)])),
    ]

    # Iterate through test cases
    for i, (array, expected) in enumerate(test_cases):
        result = calculate_na_series(array)  # Treat non-NaN values as True
        # I don't want to fiddle with the types so just convert them as strings
        assert str(result) == str(expected), f"Test case {
            i + 1} failed: Expected {expected}, got {result}"


def test_find_best_segment_in_series():
    data = pd.Series([1, None, 2, 3, 4, None, None, 5,
                     6, 7, 8, None, 9, 10, None, 11, 12])
    ndata = [
        [3,      np.nan, np.nan],
        [np.nan, 4,      np.nan],
        [2,      1,      3],
        [2,      1,      3],
        [5,      6,      np.nan]
    ]
    ndata = pd.DataFrame(ndata)

    start, end = find_best_segment_in_series(data, 1)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (7, 13)

    start, end = find_best_segment_in_series(data, 2)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (7, 16)

    start, end = find_best_segment_in_series(pd.Series([1, 2, 3, 4]), 1)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (0, 3)

    start, end = find_best_segment_in_series(ndata, 3)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (1, 4)


def simulate_missing_values(orig_segment, na_proportion, anomaly_distribution_mean, anomaly_distribution_std, anomaly_distribution_overall_percentage):
    segment = orig_segment.copy() if isinstance(
        orig_segment, pd.DataFrame) else orig_segment.to_frame()
    drop_count = round(len(segment) * na_proportion)

    # Introduce missing values at random positions
    mv_idx = np.random.choice(np.arange(1, len(segment) - 1),
                              size=drop_count, replace=False)
    mv_columns = np.random.randint(len(segment.columns), size=drop_count)
    segment_mv = segment.copy()
    for column_idx in range(len(segment.columns)):
        correct_column_mask = mv_columns == column_idx
        segment_mv.iloc[mv_idx[correct_column_mask], column_idx] = np.nan

    # Delete blocks of values to simulate anomalies
    # Per deletion, there are going to be approx. anomaly_distribution_mean values deleted
    delete_anomaly_count = round(len(
        segment) * anomaly_distribution_overall_percentage / anomaly_distribution_mean * 15)
    delete_lengths = np.random.normal(loc=anomaly_distribution_mean,
                                      scale=anomaly_distribution_std,
                                      size=delete_anomaly_count)
    starts = np.random.randint(1, len(segment), size=delete_anomaly_count)
    ends = np.minimum(
        starts + np.round(delete_lengths).astype(int), len(segment) - 2)
    for start, end in zip(starts, ends):
        segment_mv.iloc[start:end] = np.nan

    return segment_mv if isinstance(orig_segment, pd.DataFrame) else segment_mv[segment_mv.columns[0]]


def plot_multiple_autocorrelations(columns, max_lag=None, figsize=(10, 6), ylim=None):
    """
    Plots multiple autocorrelation plots in a single figure.

    Parameters:
        columns (list of pd.Series): A list of pandas Series to plot autocorrelation for.
        max_lag (int, optional): Maximum lag to plot (customizes x-axis limits). If None, no limit is applied.
        figsize (tuple): Figure size.
        ylim (tuple): Y-axis limits.
    """
    # Calculate time delta between consecutive index entries
    delta = columns[0].index[1] - columns[0].index[0]
    logging.debug(f"Delta: {delta} (type: {type(delta)})")

    # Open a new figure and clear existing ones
    plt.close('all')
    plt.figure(figsize=figsize)

    # Plot autocorrelation for each column
    for column in columns:
        pd.plotting.autocorrelation_plot(column, label=column.name)

    # Define a formatter for the x-axis based on the starting time and delta
    formatter = FuncFormatter(lambda x, pos: (
        columns[0].index[0] + x * delta).strftime('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_formatter(formatter)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=20))

    # If max_lag is provided, set the x-axis limits accordingly
    if max_lag is not None:
        plt.xlim(0, max_lag)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xticks(rotation=45)
    plt.grid(':')
    plt.legend()  # Add a legend to distinguish between columns
    plt.tight_layout()
    plt.show()


def plot_series(data, labels=None,
                windows=None,
                predictions=None,
                highlights=None,
                val_start=None,
                test_start=None,
                threshold=None,
                figsize=figsize,
                xlabel=None,
                ylabel=None):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Plot data
    plt.plot(data.index, data.values, zorder=0)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)

    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3)
    # Plot threshold
    if threshold is not None:
        plt.plot([data.index[0], data.index[-1]], [threshold,
                                                   threshold], linestyle=':', color='tab:red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(':')
    plt.tight_layout()


def impute_missing_values(df: pd.DataFrame, method: InterpolateOptions = best_imputation_method):
    """
    Impute missing values in a pandas DataFrame using the specified method.

    Parameters:
        df (pd.DataFrame): The DataFrame to impute.
        method (InterpolateOptions): The imputation method to use. Default is 'linear'.

    Returns:
        pd.DataFrame: The imputed DataFrame.
    """
    # Check if the method is valid
    if method not in ['linear', 'polynomial', 'spline', 'nearest']:
        raise ValueError(f"Invalid imputation method: {method}")

    return df.interpolate(method='linear', axis=0)


def sliding_window(data, wlen):
    # If data is a Series, convert to DataFrame for uniformity.
    if isinstance(data, pd.Series):
        data = data.to_frame()

    m = data.shape[0]
    # Collect slices corresponding to each offset in the window.
    windows = [data.iloc[i: m - wlen + i + 1].values for i in range(wlen)]

    # Horizontally stack to flatten the window slices.
    wdata = np.hstack(windows)

    # The new index corresponds to the end of each window.
    new_index = data.index[wlen - 1:]
    # New columns: flattened dimensions (wlen * original number of columns).
    num_features = data.shape[1]
    new_columns = range(wlen * num_features)

    return pd.DataFrame(wdata, index=new_index, columns=new_columns)


def test_sliding_window():
    df = pd.DataFrame({
        "Column1": [1, 2, 3, 4],
        "Column2": [-1, -2, -3, -4]
    })
    winds = sliding_window(df, 2)
    expected_df = pd.DataFrame({
        0: [1, 2, 3],
        1: [-1, -2, -3],
        2: [2, 3, 4],
        3: [-2, -3, -4]
    }, index=[1, 2, 3])
    pdt.assert_frame_equal(winds, expected_df)


def apply_sliding_window_and_aggregate(df: pd.DataFrame, window_length: int = best_window_length, aggregation_length: int = best_aggregation_length):
    """
    Apply a sliding window and aggregate the data.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        window_length (int, optional): The length of the sliding window. Default is the best window length.
        aggregation_length (int, optional): The length of the aggregation window. Default is the best aggregation length.

    Returns:
        pd.DataFrame: The aggregated DataFrame.
    """
    # Get the feature columns
    features = get_feature_columns(df)

    # Apply a sliding window
    windows = sliding_window(df[features], window_length)

    # Aggregate the windows
    aggregates = df[features].rolling(
        window=aggregation_length, min_periods=1).agg(['mean', 'var'])

    # Flatten the multi-level columns
    aggregates.columns = [f'{col}_{func}' for col, func in aggregates.columns]

    # Aggregate the 'Event' column: if any event occurs in the window, mark the window as anomalous.
    agg_event = df['Event'].rolling(window=window_length, min_periods=1).max()

    # Align the aggregated event labels with the aggregated features.
    df_agg = pd.concat([windows, aggregates.iloc[window_length-1:]], axis=1)
    df_agg.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else f'window_{
        col}' for col in df_agg.columns]

    # Drop the rows with missing values (this is just the first row, as it doesn't contain a value for the variance)
    df_agg = df_agg.dropna()

    # Add the event column
    df_agg['Event'] = agg_event

    return df_agg


def get_feature_columns(df: pd.DataFrame):
    """
    Get the feature columns from a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to get the feature columns from.

    Returns:
        list[str]: The list of feature columns.
    """
    return [col for col in df.columns if col != 'Event']


def remove_anomalies(df: pd.DataFrame):
    cp = df.copy()
    cp.drop("Event", axis=1, inplace=True)
    cp.iloc[df.Event] = np.nan
    return cp


def get_predictions_from_log_likelihood(log_likelihood, percentile):
    """
    Get the predictions from a log likelihood.

    Parameters:
        log_likelihood (float): The log likelihood of the model.
        percentile (float): The percentile to compute the performance for.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the predictions and the threshold.
    """
    # Compute threshold for this percentile.
    threshold = np.percentile(log_likelihood, percentile)

    # Get anomaly predictions based on this threshold.
    y_pred = log_likelihood < threshold

    return y_pred, threshold


def compute_model_performance(y_pred, y_true):
    """
    Compute the performance of a model.

    Parameters:
        y_pred (np.ndarray): The predictions of the model.
        y_true (np.ndarray): The true labels of the model.

    Returns:
        Tuple[float, float, float]: A tuple containing the F1 score, precision, and recall of the model.
    """

    # Evaluate against true labels.
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return f1, precision, recall


def bold(text):
    return f"\033[1m{text}\033[0m"
