import logging
import numpy as np
from time import time
from datetime import datetime as dt
timestamp = dt.utcnow().isoformat().replace(':', '_')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # instantiate codec and modem parameters
    from Wavinator.Wavinator import Wavinator
    waver = Wavinator()
    print("Theoretical bitrate: {} bps".format(waver.bit_rate))

    # create test data
    data_tx = bytes(np.random.bytes(2**8))

    # create transmit signal
    start = time()
    signal_tx = waver.wavinate(data_tx)
    end = time()
    tx_time = end - start
    print("tx {} bits in {} seconds ({} bps)".format(len(data_tx), tx_time, int(len(data_tx)/tx_time)))

    # perfect loss-less signal transmission
    import wavio
    filename = 'data/random_signal_{}.wav'.format(timestamp)
    wavio.write(filename, signal_tx, waver.sample_rate, sampwidth=2)
    signal_rx = wavio.read(filename).data[:, 0]

    # recover original data from signal
    start = time()
    data_rx = waver.dewavinate(signal_rx)
    end = time()
    rx_time = end - start
    print("rx {} bits in {} seconds ({} bps)".format(len(data_rx), rx_time, int(len(data_rx)/rx_time)))

    # check that the transmitted data is recovered from the received data (even if it has some trailing bits)
    for i, byte in enumerate(data_tx):
        assert byte == data_rx[i]

    print("Success!")

    while True:
        sentence = input()
        if sentence is not None and len(sentence) >= 16:
            sentence_bytes = np.frombuffer(bytes(sentence, encoding='utf-8'), dtype=np.uint8)
            signal_tx = waver.wavinate(sentence_bytes)
            signal_rx = signal_tx
            recovered_bytes = waver.dewavinate(signal_rx)
            recovered_sentence = str(recovered_bytes, encoding='utf-8')
            print(recovered_sentence)
        else:
            break
