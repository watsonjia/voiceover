import numpy as np
from matplotlib import pyplot as plt


def gen_waveform(qam_symbols: np.ndarray, filename: str, sym_baud: int = 32, sample_hz: int = 8e3, carrier_hz: int = 1e3):
    """
    Create a sampled audio waveform based on the provided QAM symbols.

    The SILK encoding algorithm potentially uses a sample rate of 16kHz, so we will default to a sample rate that is
    half of that in order to avoid major problems in the transcoding and compression pipeline. Similarly, our carrier
    frequency will be one-eigth of our own sample rate, allowing for no chance of missing the nyquist rate for sampling
    arbitrary modulations to the waveform.

    :param qam_symbols: 1d array of complex floats representing qam symbols to generate the audio
    :param filename: save waveform (.wav and .plt) files to this name in the data directory
    :param sym_baud: frequency of symbols per second (NOT bits per second)
    :param sample_hz: sampling frequency to use for the generated waveform
    :param carrier_hz: carrier tone (cosine wave) frequency to modulate

    :return: sampled audio as 1d array of floats representing raw values at each sample
    """
    # compute angular frequency and define time series of each symbol
    f_w = 2*np.pi*carrier_hz
    t_series = np.arange(0, 1/sym_baud, 1/sample_hz)

    # precompute in-phase carrier and quadrature carrier
    i_carrier = np.cos(f_w*t_series)
    q_carrier = np.sin(f_w*t_series)

    # start with raw carrier wave
    wave_samples = i_carrier

    # append each modulated wave per symbol
    for curr_qam_s in qam_symbols:
        # compute in-phase waveform
        i_phase_wave = np.real(curr_qam_s) * i_carrier
        # compute quadrature waveform
        q_phase_wave = np.imag(curr_qam_s) * q_carrier
        # generate wave
        curr_samples = i_phase_wave - q_phase_wave
        # append to existing wave
        wave_samples = np.append(wave_samples, curr_samples)

    # plot the signal
    plt.plot(np.arange(0, len(wave_samples)/sample_hz, 1/sample_hz), wave_samples)
    plt.savefig('data/{}.png'.format(filename))

    import wavio
    normalized_wave = (wave_samples - np.min(wave_samples)) / np.ptp(wave_samples)
    centered_wave = 2*normalized_wave - 1
    scaled_wave = np.int16(centered_wave * 32767)  # max value of int16 = 32767
    wavio.write('data/{}.wav'.format(filename), scaled_wave, sample_hz, sampwidth=2)


def parse_waveform(filename: str, sym_baud: int = 32, sample_hz: int = 8e3, carrier_hz: int = 1e3) -> np.ndarray:
    """
    Read a sampled waveform and extract the symbols modulated into it.

    :param filename: wave file to read (without the .wav extension)
    :param sym_baud: frequency of symbols per second (NOT bits per second)
    :param sample_hz: sampling frequency to use for the generated waveform
    :param carrier_hz: carrier tone (cosine wave) frequency to modulate
    :return: symbols extracted from the waveform
    """

    raise NotImplementedError()
