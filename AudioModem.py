import numpy as np


class AudioModem:

    def __init__(self, f_symbol: int = 1024, f_sample: int = int(8e3), f_carrier: int = int(1e3)):
        """
        Initialize the audio component of a modulator/demodulator using IQ modulation of complex QAM symbols.

        :param f_symbol: symbol frequency (symbols per second)
        :param f_carrier: carrier frequency for the signal
        :param f_sample: sample rate for DAC/ADC
        """
        # define basic parameters
        self.f_symbol = f_symbol
        self.f_sample = f_sample
        self.f_carrier = f_carrier

        self.w_carrier = 2*np.pi*self.f_carrier  # angular frequency of raw carrier
        self.symbol_period = 1/self.f_symbol  # seconds per symbol
        self.upsmple_factor = int(self.symbol_period*self.f_sample)  # samples per symbol

        # compute/construct filter FIR coefficents and delays
        rrc_delay, self.rrc_fir = self._construct_rrc_filter()
        lp_delay, self.lp_fir = self._construct_lowpass_filter()
        self.filter_delay_samples = int((2*rrc_delay + lp_delay)*self.f_sample)

    def _construct_rrc_filter(self):
        # filter parameters
        rrc_len_factor = 6
        rrc_alpha_coefficient = 0.4

        from commpy.filters import rrcosfilter
        filter_len_sec = rrc_len_factor * self.symbol_period
        filter_len_samples = int(filter_len_sec * self.f_sample)
        _, rrc = rrcosfilter(N=filter_len_samples, alpha=rrc_alpha_coefficient, Ts=self.symbol_period, Fs=self.f_sample)

        delay = filter_len_sec / 2
        return delay, rrc

    def _construct_lowpass_filter(self):
        # filter parameters
        lp_cutoff_factor = 5
        lp_order = 51

        from scipy.signal import firwin
        nyquist_bandwidth = self.f_symbol / 2
        cutoff = lp_cutoff_factor * nyquist_bandwidth
        lp_filter = firwin(lp_order, cutoff / (self.f_sample / 2))

        delay = (lp_order // 2) / self.f_sample
        return delay, lp_filter

    def modulate(self, qam_symbols: np.ndarray) -> np.ndarray:
        """
        Take a series of complex QAM signals (the baseband digital signal) and modulate it into an analog bandpass
        signal at the carrier frequency using IQ modulation.

        :param qam_symbols: digital baseband signal as series of complex QAM signals
        :return: sampled analog signal of the modulated waveform
        """
        # upsample the qam symbols to a comb signal and apply pulse shaping rrc filter
        from scipy.signal import upfirdn
        baseband_signal = upfirdn(self.rrc_fir, qam_symbols, self.upsmple_factor)
        baseband_signal_t = np.arange(len(baseband_signal)) / self.f_sample

        # compute the in-phase and quadrature components of the signal
        i_quad = baseband_signal.real * np.cos(self.w_carrier * baseband_signal_t)
        q_quad = baseband_signal.imag * np.sin(self.w_carrier * baseband_signal_t) * (-1)

        # sum the quadrature signals to get a real-valued signal for transmission
        signal = i_quad + q_quad
        return signal

    def demodulate(self, rx_wave: np.ndarray) -> np.ndarray:
        """
        Extract the complex QAM symbols modulated into the given rx_wave. Demodulates the IQ modulation implemented by
        the modulate method above.

        :param rx_wave: sampled analog IQ modulated waveform
        :return: complex QAM symbols extracted from the modulated signal
        """
        # extract the quadrature components
        rx_wave_t = np.arange(len(rx_wave)) / self.f_sample
        i_quad = rx_wave * np.cos(self.w_carrier * rx_wave_t)
        q_quad = rx_wave * np.sin(self.w_carrier * rx_wave_t) * (-1)

        # filter the quadrature signals
        from scipy.signal import lfilter
        i_quad = lfilter(self.lp_fir, 1, i_quad)
        q_quad = lfilter(self.lp_fir, 1, q_quad)

        # combine the signals back into complex quadrature representation
        recovered = i_quad + 1.j * q_quad

        # apply the second rrc filter for full raised cosine filter
        recovered_signal = np.convolve(recovered, self.rrc_fir)

        # discard prepended delay samples from filtering and sample remaining signal to recover original symbols
        recovered_symbols = recovered_signal[self.filter_delay_samples::int(self.upsmple_factor)]

        return recovered_symbols

