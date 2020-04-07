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
    w_c = 2*np.pi*carrier_hz
    t_series = np.arange(0, 1/sym_baud, 1/sample_hz)

    # precompute in-phase carrier and quadrature carrier
    i_carrier = np.cos(w_c*t_series)
    q_carrier = np.sin(w_c*t_series)

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

    normalized_wave = (wave_samples - np.min(wave_samples)) / np.ptp(wave_samples)
    centered_wave = 2*normalized_wave - 1

    import wavio
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
    # compute angular frequency of carrier
    w_c = 2*np.pi*carrier_hz

    import wavio
    input_wave: np.ndarray = wavio.read('data/{}.wav'.format(filename)).data[:, 0]
    scaled_wave = input_wave / np.maximum(np.abs(np.min(input_wave)), np.abs(np.max(input_wave)))
    scaled_wave = scaled_wave * np.sqrt(2)  # TODO: get scale factor automatically, not assuming sqrt(1^2+1^2)

    # average phasor across each symbol period
    symbol_period = int(sample_hz / sym_baud)
    curr_period_phasors = []

    # collect one symbol per symbol period
    parsed_symbols = np.empty((0,), dtype=complex)

    for t, sample in enumerate(scaled_wave):
        # compute angle at this sample
        w_t = t * w_c / sample_hz
        # calculate the in-phase component
        i_phase = sample * np.cos(w_t)
        # calculate the quadrature component
        q_phase = -1 * sample * np.sin(w_t)

        # calculate the recovered symbol
        recovered = np.complex(i_phase, q_phase)
        curr_period_phasors.append(recovered)

        if t % symbol_period == symbol_period - 1:
            parsed_symbols = np.append(parsed_symbols, np.mean(curr_period_phasors))
            curr_period_phasors = []

    return parsed_symbols[1:]  # ignore the training signal for now
