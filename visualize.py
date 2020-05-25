import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def moving_average(data_src: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the moving average values for the 1D source array based on n values in window.
    Shamelessly stolen from https://gordoncluster.wordpress.com/2014/01/29/python-numpy-how-to-generate-moving-averages-efficiently-part-1/
    :param data_src: input dataset, must be single-dimension
    :param n: window size for moving average
    :return: series of moving window average values of the same dimensions as data_src
    """
    weights = np.repeat(1.0, n)/n
    return np.convolve(data_src, weights, 'same')


def draw_io_plot(io_array: np.ndarray, title: str, y_label: str = None, x_label: str = None):
    plt_colors = ['blue', 'red']
    plt_option = dict(linewidth=0.75)
    window_size_s = 5  # 5 second moving average

    for io, color in zip(io_array, plt_colors):
        io_avg = moving_average(io, window_size_s)
        io_t = np.arange(0, len(io_avg))
        plt.plot(io_t, io_avg, color=color, **plt_option)

    plt.title(title)

    if x_label is not None:
        plt.xlabel('time (seconds)')
    if y_label is not None:
        plt.ylabel(y_label)

    plt.xlim((120, 240))  # zoom into middle segment
    plt.ylim((4000, 12000))  # show consistent scales across all plots

    plt.grid()


def extract_bytes(filename: str):
    raw_data = pd.read_csv(filename)
    bytes_in = raw_data['Bytes IN'].to_numpy(dtype=float, copy=True)
    bytes_out = raw_data['Bytes OUT'].to_numpy(dtype=float, copy=True)

    return np.stack([bytes_in, bytes_out])


if __name__ == '__main__':
    vovr_io = extract_bytes('data/prelim_measurements/gan_prelim.csv')
    real_io = extract_bytes('data/real_01.5hr.csv')
    strw_io = extract_bytes('data/prelim_measurements/non_prelim.csv')

    ROWS = 3
    COLS = 1

    plt_legend = ['inbound', 'outbound']

    plt.figure(figsize=(8, 6))
    plt.subplot(ROWS, COLS, 1)
    draw_io_plot(real_io, title='Real Conversation')
    plt.legend(plt_legend, loc='upper right')
    plt.subplot(ROWS, COLS, 2)
    draw_io_plot(strw_io, title='Strawman Transmission', y_label='bytes per second')
    plt.subplot(ROWS, COLS, 3)
    draw_io_plot(vovr_io, title='Voiceover Transmission', x_label='time (s)')
    plt.show()
