import numpy as np


def gen_waveform(qam_symbols: np.ndarray, sym_baud: int = 16, sample_hz: int = 8e3, carrier_hz: int = 1e3):
    """
    Create a sampled audio waveform based on the provided QAM symbols.

    The Skype encoding algorithm potentially uses a sample rate of 16kHz, so we will default to a sample rate that is
    half of that in order to avoid major problems in the transcoding and compression pipeline. Similarly, our carrier
    frequency will be one-eigth of our own sample rate, allowing for no chance of missing the nyquist rate for sampling
    arbitrary modulations to the waveform.

    :param qam_symbols: 1d array of complex floats representing qam symbols to generate the audio
    :param sym_baud: frequency of symbols per second (NOT bits per second)
    :param sample_hz: sampling frequency to use for the generated waveform
    :param carrier_hz: carrier tone (cosine wave) frequency to modulate

    :return: sampled audio as 1d array of floats representing raw values at each sample
    """
    f_w = 2*np.pi*carrier_hz
    t_series = np.arange(0, 1/sym_baud, 1/sample_hz)

    wave_samples = []

    for curr_qam_s in qam_symbols:
        curr_samples = np.abs(curr_qam_s) * np.cos(f_w * t_series + np.angle(curr_qam_s))
        wave_samples = np.append(wave_samples, curr_samples)

    from datetime import datetime as dt
    timestamp = dt.utcnow().isoformat().replace(':', '_')

    # import matplotlib
    # matplotlib.use('Qt5Agg')
    # from matplotlib import pyplot as plt
    # plt.plot(np.arange(0, len(wave_samples)/sample_hz, 1/sample_hz), wave_samples)
    # plt.savefig('data/{}.png'.format(timestamp))

    import wavio
    normalized = (wave_samples - np.min(wave_samples)) / np.ptp(wave_samples)
    centered = 2*normalized - 1
    scaled = np.int16(centered * 32767)  # max value of int16 = 32767
    wavio.write('data/{}.wav'.format(timestamp), scaled, sample_hz, sampwidth=2)

