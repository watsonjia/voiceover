import numpy as np
import pandas as pd


WINDOW_SECONDS = 30
STEP_SECONDS = 10

WINDOW_SIZE = WINDOW_SECONDS * 10  # 60 ~s * (1000 ~ms / 1 ~s) * (1 element / 100 ~ms) = 600 elements per minute
STEP_SIZE = STEP_SECONDS * 10


def extract_bytes(src_file: str, is_fake: bool):
    src_df = pd.read_csv(src_file)

    tmp_df = src_df.rename(columns={'Interval start': 't', 'Bytes OUT': 'b_out', 'Bytes IN': 'b_in'})
    if is_fake:
        tmp_df['fake'] = np.ones_like(tmp_df['t'])
    else:
        tmp_df['fake'] = np.zeros_like(tmp_df['t'])

    return window_samples(tmp_df)


def extract_bytes_multi(src_files: list, is_fake: bool):
    total_samples = []

    for filename in src_files:
        curr_samples, curr_labels = extract_bytes(filename, is_fake)

        total_samples.append(curr_samples)

    total_samples = np.concatenate(total_samples, axis=0)

    if is_fake:
        total_labels = np.ones(total_samples.shape[0])
    else:
        total_labels = np.zeros(total_samples.shape[0])

    return total_samples, total_labels


def window_samples(data_set: pd.DataFrame, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    new_samples = []
    new_labels = []

    start_element = 0
    end_element = window_size

    while end_element < len(data_set):
        bytes_in = data_set['b_in'][start_element:end_element]
        bytes_out = data_set['b_out'][start_element:end_element]

        tmp_sample = np.stack((bytes_in, bytes_out))
        new_samples.append(tmp_sample)
        new_labels.append(data_set['fake'][end_element - 1])

        start_element += step_size
        end_element += step_size

    new_samples = np.stack(new_samples)
    new_labels = np.array(new_labels)
    return new_samples, new_labels


if __name__ == '__main__':
    gan_samples, gan_labels = extract_bytes('../data/gan_01.0hr.csv', is_fake=True)
    real_samples, real_labels = extract_bytes('../data/real_01.5hr.csv', is_fake=False)
    straw_samples, straw_labels = extract_bytes('../data/straw_01.0hr.csv', is_fake=True)
