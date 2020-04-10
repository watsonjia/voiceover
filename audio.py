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
    # compute angular frequency of carrier and symbol period (in samples)
    w_c = 2*np.pi*carrier_hz
    samples_per_symbol = int(sample_hz / sym_baud)

    # sample_t represents the current sample index
    sample_t = 0
    signal = []

    for curr_qam_s in qam_symbols:
        for _ in range(samples_per_symbol):
            # compute carrier angle (w*t + phi) at this sample
            w_t = sample_t * w_c / sample_hz

            # compute the quadrature signals
            i_quad = np.real(curr_qam_s) * np.cos(w_t)
            q_quad = np.imag(curr_qam_s) * (-1) * np.sin(w_t)

            # sum the quadratures and append the sample
            signal.append(i_quad + q_quad)

            # increment the sample counter
            sample_t += 1

    import wavio
    signal = np.array(signal)
    wavio.write('data/{}.wav'.format(filename), signal, sample_hz, sampwidth=2)


def parse_waveform(filename: str, sym_baud: int = 32, sample_hz: int = 8e3, carrier_hz: int = 1e3) -> np.ndarray:
    """
    Read a sampled waveform and extract the symbols modulated into it.

    :param filename: wave file to read (without the .wav extension)
    :param sym_baud: frequency of symbols per second (NOT bits per second)
    :param sample_hz: sampling frequency to use for the generated waveform
    :param carrier_hz: carrier tone (cosine wave) frequency to modulate
    :return: symbols extracted from the waveform
    """
    # compute angular frequency of carrier and symbol period (in samples)
    w_c = 2*np.pi*carrier_hz
    samples_per_symbol = int(sample_hz / sym_baud)

    import wavio
    input_wave: np.ndarray = wavio.read('data/{}.wav'.format(filename)).data

    symbols = []

    for sample_t, signal_sample in enumerate(input_wave):
        # compute carrier angle (w*t + phi) at this sample
        w_t = sample_t * w_c / sample_hz

        # extract the quadrature signals
        i_quad = signal_sample * np.cos(w_t)
        q_quad = signal_sample * (-1) * np.sin(w_t)

        # calculate the complex signal's QI symbol
        recovered = np.complex(i_quad, q_quad)
        symbols.append(recovered)

    # average phasor across each symbol period
    symbols = np.array(symbols)
    avg_parsed_symbols = np.mean(symbols.reshape(-1, samples_per_symbol), 1)

    return avg_parsed_symbols
