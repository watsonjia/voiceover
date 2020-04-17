import numpy as np
import commpy.channelcoding.convcode as cc


class ConvolutionCodec:
    def __init__(self):
        """
        Initialize the codec scheme with trellis structure:
            G(D) = [[1, 0, 0], [0, 1, 1+D]]
            F(D) = [[D, D], [1+D, 1]]
        """
        # instantiate trellis object
        self._trellis = self._make_trellis()

    @staticmethod
    def _make_trellis() -> cc.Trellis:
        """
        Convolutional Code:
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

    def encode(self, message: np.ndarray) -> np.ndarray:
        # convert bytes to bit array
        bits = np.unpackbits(message, bitorder='big')

        # perform the convolution encoding
        return cc.conv_encode(bits, self._trellis, termination='cont')

    def decode(self, encoded):
        # decode the probable bits from the encoded string
        decoded = cc.viterbi_decode(encoded, self._trellis, decoding_type='hard')

        # return the bytes from the decoded bits
        return np.packbits(decoded, bitorder='big')

    @property
    def coding_rate(self):
        return self._trellis.k / self._trellis.n
