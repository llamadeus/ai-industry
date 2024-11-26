import numpy as np
import pandas as pd
from fontTools.designspaceLib.types import locationInRegion
from matplotlib import pyplot as plt

# Configuration
anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'
figsize = (9, 3)


def find_best_segment_in_series(series, max_missing):
    """
    Find the longest segment in a Series with at most `max_missing` missing values.

    Parameters:
        series (pd.Series): The input series with potential missing values.
        max_missing (int): Maximum allowed missing values in a segment.

    Returns:
        tuple: (start_index, end_index) of the best segment.
    """
    # Given an array M if the indices of the missing values
    # Extending any segment up to the the next missing value adds exactly one missing value.
    # Therefore, for any index `i`, any segment `series[M[i]:M[i+max_missing]] contains `max_missing` missing values.
    # As such, we look for the longest such segment.
    indices_of_missing_values = np.array([-1, *np.where(series.isna())[0], len(series)])
    # The number must not be larger than the total number of Nas
    max_missing = min(max_missing, len(indices_of_missing_values) - 2)
    segment_lengths_plus_1 = indices_of_missing_values[max_missing+1:] - indices_of_missing_values[:-max_missing-1]
    location_where_we_find_the_index_where_the_segment_starts = segment_lengths_plus_1.argmax()
    return [1, -1] + indices_of_missing_values[np.array([0, max_missing+1]) + location_where_we_find_the_index_where_the_segment_starts]


def test_find_best_segment_in_series():
    data = pd.Series([1, None, 2, 3, 4, None, None, 5, 6, 7, 8, None, 9, 10, None, 11, 12])

    start, end = find_best_segment_in_series(data, 1)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (7, 13)

    start, end = find_best_segment_in_series(data, 2)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (7, 16)

    start, end = find_best_segment_in_series(pd.Series([1, 2, 3, 4]), 1)
    print(f"Best segment: {start} to {end}")
    assert (start, end) == (0, 3)


def plot_autocorrelation(data, max_lag=100, figsize=figsize):
    # Open a new figure
    plt.close('all')
    plt.figure(figsize=figsize)
    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(data['value'])
    # Customized x limits
    plt.xlim(0, max_lag)
    # Rotated x ticks
    plt.xticks(rotation=45)
    plt.grid(':')
    plt.tight_layout()


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


def bold(text):
    return f"\033[1m{text}\033[0m"