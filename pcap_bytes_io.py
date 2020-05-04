from scapy.utils import RawPcapNgReader
from scapy.layers.inet import IP, Ether
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Qt5Agg')

MY_IP = '10.0.2.15'
MY_IP_BACKWARDS = '2.15.10.0'
T_STEP_MS = 100


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
    plt_legend = ['inbound', 'outbound']
    plt_colors = ['blue', 'orange']
    plt_option = dict(linewidth=0.5)
    window_size = int(round(5 * (1000 / T_STEP_MS)))  # 5 second moving average

    for io, color in zip(io_array, plt_colors):
        io_avg = moving_average(io, window_size)
        io_t = np.arange(0, len(io_avg)*T_STEP_MS, T_STEP_MS) / 1000  # units of seconds not ms
        plt.plot(io_t, io_avg, color=color, **plt_option)

    plt.title(title)

    if x_label is not None:
        plt.xlabel('time (seconds)')
    if y_label is not None:
        plt.ylabel(y_label)

    plt.xlim((0, 3600))  # zoom into middle segment
    plt.ylim((0, 50000))  # show consistent scales across all plots

    plt.grid()
    plt.legend(plt_legend)


def count_io(filename: str):
    next_time_ms = None
    running_out_bytes = 0
    running_out_packets = 0
    running_in_bytes = 0
    running_in_packets = 0

    out_bytes = []
    out_packets = []
    in_bytes = []
    in_packets = []

    addresses = {}

    file = open(filename, mode='rb')
    for (p_data, p_meta) in RawPcapNgReader(file):
        packet = Ether(p_data)
        if 'type' not in packet.fields or packet.type != 0x0800:
            # ignore non-IPv4 packets
            continue

        packet = packet[IP]
        src = packet.src
        dst = packet.dst
        time_raw = ((p_meta.tshigh << 32) | p_meta.tslow)
        time_ms = int(round((time_raw / p_meta.tsresol) * 1000))

        if src in addresses:
            addresses[src] = addresses[src] + 1
        else:
            addresses[src] = 1

        if dst in addresses:
            addresses[dst] = addresses[dst] + 1
        else:
            addresses[dst] = 1

        if next_time_ms is None:
            next_time_ms = time_ms - (time_ms % T_STEP_MS)
            next_time_ms += T_STEP_MS

        if time_ms <= next_time_ms:
            if src == MY_IP or src == MY_IP_BACKWARDS:
                running_out_bytes += len(packet)  # p_meta.wirelen
                running_out_packets += 1
            elif dst == MY_IP or dst == MY_IP_BACKWARDS:
                running_in_bytes += len(packet)  # p_meta.wirelen
                running_in_packets += 1

        else:
            out_bytes.append(running_out_bytes)
            out_packets.append(running_out_packets)
            in_bytes.append(running_in_bytes)
            in_packets.append(running_in_packets)

            running_out_bytes = 0
            running_out_packets = 0
            running_in_bytes = 0
            running_in_packets = 0

            next_time_ms += T_STEP_MS

    file.close()

    # convert to bytes/packets per second and stack
    io_bytes = np.array([in_bytes, out_bytes], dtype=int) * (1000 / T_STEP_MS)
    io_packets = np.array([in_packets, out_packets], dtype=int) * (1000 / T_STEP_MS)

    # # print the count of all addresses
    # d_view = [(v, k) for k, v in addresses.items()]
    # d_view.sort(reverse=True)
    # for v, k in d_view:
    #     print('{}: {}'.format(k, v))

    return io_bytes, io_packets


if __name__ == '__main__':
    real_bytes: np.ndarray
    real_packets: np.ndarray
    straw_bytes: np.ndarray
    straw_packets: np.ndarray
    voiceover_bytes: np.ndarray
    voiceover_packets: np.ndarray

    # real_bytes, real_packets = count_io('data/prelim_measurements/en4245.pcapng')
    # straw_bytes, straw_packets = count_io('data/prelim_measurements/non_prelim.pcapng')
    # voiceover_bytes, voiceover_packets = count_io('data/prelim_measurements/gan_prelim.pcapng')

    real_bytes, real_packets = count_io('data/real_01.5hr_trim.pcapng')
    np.savez_compressed('data/real_01.5hr_trim.pcapng.npz', real_bytes=real_bytes, real_packets=real_packets)

    straw_bytes, straw_packets = count_io('data/prelim_measurements/non_prelim.pcapng')

    voiceover_bytes, voiceover_packets = count_io('data/gan_01.0hr_trim.pcapng')
    np.savez_compressed('data/gan_01.0hr_trim.pcapng.npz', gan_bytes=voiceover_bytes, gan_packets=voiceover_packets)

    ROWS = 3
    COLS = 1

    plt.subplot(ROWS, COLS, 1)
    draw_io_plot(real_bytes, 'Real Conversation: Bitrate vs Time')
    plt.subplot(ROWS, COLS, 2)
    draw_io_plot(straw_bytes, 'Strawman Solution: Bitrate vs Time', y_label='bytes per second')
    plt.subplot(ROWS, COLS, 3)
    draw_io_plot(voiceover_bytes, 'Voiceover Solution: Bitrate vs Time', x_label='time (seconds)')

    plt.show()
