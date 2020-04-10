import numpy as np
from matplotlib import pyplot as plt


def _construct_rrc_filter(sym_width_s, sample_hz):
    from commpy.filters import rrcosfilter
    filter_len_sec = 6*sym_width_s  # span six symbols, three ahead and three behind
    filter_len_samples = int(filter_len_sec * sample_hz)
    _, rrc = rrcosfilter(N=filter_len_samples, alpha=0.4, Ts=sym_width_s, Fs=sample_hz)

    delay = filter_len_sec / 2

    return delay, rrc


def gen_waveform(qam_symbols: np.ndarray,
                 filename: str,
                 sym_per_second: int = 512,
                 samples_per_second: int = int(8e3),
                 carrier_hz: int = int(1e3)):
    """
    Create a sampled audio waveform based on the provided QAM symbols.

    The SILK encoding algorithm potentially uses a sample rate of 16kHz, so we will default to a sample rate that is
    half of that in order to avoid major problems in the transcoding and compression pipeline. Similarly, our carrier
    frequency will be one-eigth of our own sample rate, allowing for no chance of missing the nyquist rate for sampling
    arbitrary modulations to the waveform.

    :param qam_symbols: 1d array of complex floats representing qam symbols to generate the audio
    :param filename: save waveform (.wav and .plt) files to this name in the data directory
    :param sym_per_second: frequency of symbols per second (NOT bits per second)
    :param samples_per_second: sampling frequency to use for the generated waveform
    :param carrier_hz: carrier tone (cosine wave) frequency to modulate

    :return: sampled audio as 1d array of floats representing raw values at each sample
    """
    # alternate variable names
    # Fs = int(samples_per_second)  # sample rate of analog signals in Hz
    # fc = int(carrier_hz)  # carrier frequency in Hz
    # Ts = 1/sym_per_second  # symbol spacing (i.e., seconds per symbol)
    # BN = 1/(2*Ts)  # baseband nyquist bandwidth
    #
    # ups = int(Ts*Fs)  # upsample factor (e.g., samples per symbol in analog output waveform)
    # N = len(qam_symbols)  # num baseband symbols to send
    #
    # t0 = 3*Ts  # half filter length
    # end alternate variable names

    # compute angular frequency of carrier and symbol period (in samples)
    w_c = 2*np.pi*carrier_hz
    sym_width_s = 1/sym_per_second

    # filter with root raised cosine (pulse shaping) to avoid intersymbol interference
    _, rrc = _construct_rrc_filter(sym_width_s, samples_per_second)

    # upsample the baseband digital signal and apply the filter
    from scipy.signal import upfirdn
    baseband_signal = upfirdn(rrc, qam_symbols, int(sym_width_s*samples_per_second))
    t_signal = np.arange(len(baseband_signal)) / samples_per_second

    # compute the in-phase and quadrature components of the signal
    i_quad = baseband_signal.real * np.cos(w_c*t_signal)
    q_quad = baseband_signal.imag * np.sin(w_c*t_signal) * (-1)

    signal = i_quad + q_quad

    import wavio
    wavio.write('data/{}.wav'.format(filename), signal, samples_per_second, sampwidth=2)


def parse_waveform(filename: str,
                   sym_per_second: int = 512,
                   samples_per_second: int = int(8e3),
                   carrier_hz: int = int(1e3)) -> np.ndarray:
    """
    Read a sampled waveform and extract the symbols modulated into it.

    :param filename: wave file to read (without the .wav extension)
    :param sym_per_second: frequency of symbols per second (NOT bits per second)
    :param samples_per_second: sampling frequency to use for the generated waveform
    :param carrier_hz: carrier tone (cosine wave) frequency to modulate
    :return: symbols extracted from the waveform
    """
    # compute angular frequency of carrier and symbol period (in samples)
    w_c = 2*np.pi*carrier_hz
    samples_per_symbol = int(samples_per_second / sym_per_second)

    import wavio
    input_wave: np.ndarray = wavio.read('data/{}.wav'.format(filename)).data[:, 0]

    t_signal = np.arange(len(input_wave)) / samples_per_second

    # extract the quadrature components
    i_quad = input_wave * np.cos(w_c*t_signal)
    q_quad = input_wave * np.sin(w_c*t_signal) * (-1)

    # construct lowpass image rejection filter
    from scipy.signal import firwin
    nyquist_bandwidth = sym_per_second / 2
    lp_cutoff = 5*nyquist_bandwidth
    lp_order = 51
    lp_delay = (lp_order // 2) / samples_per_second
    lowpass = firwin(lp_order, lp_cutoff / (samples_per_second / 2))

    # filter the quadrature signals
    from scipy.signal import lfilter
    i_quad = lfilter(lowpass, 1, i_quad)
    q_quad = lfilter(lowpass, 1, q_quad)

    # combine the signals back into complex quadrature representation
    recovered = i_quad + 1.j*q_quad

    # apply the second root raised cosine filter for a full raised cosine filter
    rrc_delay, rrc = _construct_rrc_filter(1 / sym_per_second, samples_per_second)
    recovered_signal = np.convolve(recovered, rrc)

    # calculate total expected delay from filtering
    delay_samples = int((2*rrc_delay + lp_delay)*samples_per_second)

    # sample the signal at appropriate points to recover baseband signal
    recovered_symbols = recovered_signal[delay_samples::int(samples_per_symbol)]

    return recovered_symbols
