import numpy as np

from CodecModem import CodecModem

cm = CodecModem(2)
data = np.random.randint(0, 2, (32,))

modulated = cm.encode_modulate(data)
demodulated = cm.demodulate_decode(modulated)

assert np.array_equal(data, demodulated)

from audio import gen_waveform

gen_waveform(modulated, sym_baud=4)
