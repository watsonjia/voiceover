import numpy as np


class Wavinator:

    def __init__(self):
        from Wavinator.ConvolutionCodec import ConvolutionCodec
        from Wavinator.IQModem import IQModem

        self._codec = ConvolutionCodec()
        self._modem = IQModem()

    def wavinate(self, message: bytes):
        # convert bytes to ndarray of uint8 (unsigned byte array)
        data_type = np.dtype('uint8')
        data_type = data_type.newbyteorder('>')
        byte_array = np.frombuffer(message, dtype=data_type)

        # encode with convolution coder and modulate into waveform
        coded = self._codec.encode(byte_array)
        return self._modem.modulate(coded)

    def dewavinate(self, rx_wave: np.ndarray):
        coded = self._modem.demodulate(rx_wave)
        return self._codec.decode(coded)

    @property
    def bit_rate(self):
        return self._codec.coding_rate * self._modem.bitrate
