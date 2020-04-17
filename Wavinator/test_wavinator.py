from unittest import TestCase
import numpy as np


class TestWavinator(TestCase):
    def test_wavinator_random(self):
        # instantiate the wavinator
        from Wavinator.Wavinator import Wavinator
        waver = Wavinator()

        # create test data
        data_tx = bytes(np.random.bytes(1024))

        # create transmit signal
        signal_tx = waver.wavinate(data_tx)

        # perfect loss-less signal transmission
        signal_rx = signal_tx

        # recover original data from signal
        data_rx = waver.dewavinate(signal_rx)

        for i, byte in enumerate(data_rx):
            assert byte == data_rx[i]

