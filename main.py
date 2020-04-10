import numpy as np
from time import time
from datetime import datetime as dt
timestamp = dt.utcnow().isoformat().replace(':', '_')


if __name__ == '__main__':
    # instantiate codec and modem parameters
    from CodecModem import CodecModem
    cod_mod = CodecModem(2)
    from AudioModem import AudioModem
    aud_mod = AudioModem()
    print("Theoretical bitrate ignoring coding overhead: {} bps".format(aud_mod.f_symbol*2))  # 2 bits per symbol

    # create test data
    data_tx = np.random.randint(0, 2, (2**16,))

    # create transmit signal
    start = time()
    symbol_tx = cod_mod.encode_modulate(data_tx)
    signal_tx = aud_mod.modulate(symbol_tx)
    end = time()
    tx_time = end - start
    print("tx {} bits in {} seconds ({} bps)".format(len(data_tx), tx_time, int(len(data_tx)/tx_time)))

    # perfect loss-less signal transmission
    signal_rx = signal_tx

    # recover original data from signal
    start = time()
    symbol_rx = aud_mod.demodulate(signal_rx)
    data_rx = cod_mod.demodulate_decode(symbol_rx)
    end = time()
    rx_time = end - start
    print("rx {} bits in {} seconds ({} bps)".format(len(data_rx), rx_time, int(len(data_rx)/rx_time)))

    # check that the transmitted data is recovered from the received data (even if it has some trailing bits)
    for i, bit in enumerate(data_tx):
        assert bit == data_rx[i]

    print("Success!")
