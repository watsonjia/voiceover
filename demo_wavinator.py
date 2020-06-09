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

    import wavio

    while True:
        sentence = input('Input text for coding and modulation...')
        if sentence is not None and len(sentence) >= 16:
            sentence_bytes = bytes(sentence, encoding='utf-8')
            signal_tx = waver.wavinate(sentence_bytes)
            signal_rx = signal_tx
            recovered_bytes = waver.dewavinate(signal_rx)
            recovered_sentence = str(recovered_bytes, encoding='utf-8')

            wavio.write('data/sentence_signal_{}.wav'.format(timestamp), signal_tx, waver.sample_rate, sampwidth=2)
            logging.info('Saved audio to data/sentence_signal_{}.wav'.format(timestamp))
            print('Recovered from audio: ' + recovered_sentence)
        else:
            break
