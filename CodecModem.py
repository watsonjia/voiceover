import numpy as np

import commpy.channelcoding.convcode as cc
import commpy.modulation as mod


class CodecModem:

    def __init__(self, trellis_type=1):
        """
        Initialize the codec modem scheme with one of two available trellis structures.

        :param trellis_type: default 1 for convolutional code {G(D) = [1+D^2, 1+D+D^2]} or 2 for
                             {G(D) = [[1, 0, 0], [0, 1, 1+D]], F(D) = [[D, D], [1+D, 1]]}
        """
        # instantiate trellis object
        self.trellis = self._make_trellis1() if trellis_type == 1 else self._make_trellis2()
        # create QPSK modem
        self.modem = mod.QAMModem(4)

    @staticmethod
    def _make_trellis1() -> cc.Trellis:
        """
        Convolutional Code 1:
            G(D) = [1+D^2, 1+D+D^2]

        :return: trellis object implementing this convolutional encoding scheme
        """
        # Number of delay elements in the convolutional encoder
        memory = np.array(2, ndmin=1)
        # Generator matrix
        g_matrix = np.array((0o5, 0o7), ndmin=2)

        # Create trellis data structure
        return cc.Trellis(memory, g_matrix)

    @staticmethod
    def _make_trellis2() -> cc.Trellis:
        """
        Convolutional Code 2:
            G(D) = [[1, 0, 0], [0, 1, 1+D]]
            F(D) = [[D, D], [1+D, 1]]

        :return: trellis object implementing this convolutional encoding scheme
        """
        # Number of delay elements in the convolutional encoder
        memory = np.array((1, 1))

        # Generator matrix & feedback matrix
        g_matrix = np.array(((1, 0, 0), (0, 1, 3)))
        feedback = np.array(((2, 2), (3, 1)))

        # Create trellis data structure
        return cc.Trellis(memory, g_matrix, feedback, 'rsc')

    def encode_modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        Encode and modulate bits into an array of constellation symbols.

        :param bits: 1D array containing one bit (integer) {0, 1} per element
        :return: 1D array of complex symbols (floats) representing the modulated bits
        """
        # perform the convolutional encoding
        encoded = cc.conv_encode(bits, self.trellis, termination='cont')
        # modulate the encoded bits and return the symbols
        return self.modem.modulate(encoded)

    def demodulate_decode(self, symbols):
        """
        Demodulate and decode the (previously modulated) symbols into the bits they represent. This is the inverse of
        the self.encode_modulate(...) function above.

        :return: 1D array containing one bit (integer) {0, 1} which was likely used to create this string of symbols
        """
        # demodulate the symbols into the encoded bits they likely represent
        encoded = self.modem.demodulate(symbols, demod_type='hard')
        return cc.viterbi_decode(encoded, self.trellis, decoding_type='hard')

