import numpy as np

from datetime import datetime as dt
timestamp = dt.utcnow().isoformat().replace(':', '_')


if __name__ == '__main__':
    from CodecModem import CodecModem
    cm = CodecModem(2)
    data = np.random.randint(0, 2, (32,))

    # test encoding/modulation
    modulated = cm.encode_modulate(data)
    demodulated = cm.demodulate_decode(modulated)
    assert np.array_equal(data, demodulated)

    # test waveform generation and parsing
    from audio import gen_waveform, parse_waveform
    gen_waveform(modulated, timestamp, sym_baud=64)
    parsed = parse_waveform(timestamp, sym_baud=64)
    assert np.array_equal(modulated, parsed)

    # test full pipeline
    data = np.random.randint(0, 2, (32,))
    modulated = cm.demodulate_decode(data)
    gen_waveform(modulated, '{}_full'.format(timestamp), sym_baud=64)
    parsed = parse_waveform('{}_full'.format(timestamp), sym_baud=64)
    demodulated = cm.demodulate_decode(parsed)
    assert np.array_equal(data, demodulated)

