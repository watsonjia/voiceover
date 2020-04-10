import numpy as np

from datetime import datetime as dt
timestamp = dt.utcnow().isoformat().replace(':', '_')


if __name__ == '__main__':
    from CodecModem import CodecModem
    cm = CodecModem(2)
    data_tx = np.random.randint(0, 2, (2*8192,))

    # test encoding/modulation
    encoded = cm.encode_modulate(data_tx)
    data_rx = cm.demodulate_decode(encoded)
    assert np.array_equal(data_tx, data_rx)

    # test waveform generation and parsing
    from audio import gen_waveform, parse_waveform

    # test full pipeline
    encoded = cm.encode_modulate(data_tx)
    gen_waveform(encoded, '{}_full'.format(timestamp))
    parsed = parse_waveform('{}_full'.format(timestamp))
    data_rx = cm.demodulate_decode(parsed)

    for i, bit in enumerate(data_tx):
        assert bit == data_rx[i]
