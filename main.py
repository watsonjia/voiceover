import numpy as np

from datetime import datetime as dt
timestamp = dt.utcnow().isoformat().replace(':', '_')


if __name__ == '__main__':
    from CodecModem import CodecModem
    cm = CodecModem(2)
    data = np.random.randint(0, 2, (32,))

    # test encoding/modulation
    encoded = cm.encode_modulate(data)
    decoded = cm.demodulate_decode(encoded)
    assert np.array_equal(data, decoded)

    # test waveform generation and parsing
    from audio import gen_waveform, parse_waveform

    # test full pipeline
    encoded = cm.encode_modulate(data)
    gen_waveform(encoded, '{}_full'.format(timestamp))
    parsed = parse_waveform('{}_full'.format(timestamp))
    decoded = cm.demodulate_decode(parsed)
    assert np.array_equal(data, decoded)
