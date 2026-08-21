import argparse
import numpy as np
import scipy.io as sio
import math as mt
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
import pickle
import pywt
import pandas as pd
import os
from tqdm import tqdm


def hampel_filter(input_matrix, window_size, n_sigmas=3):
    n_rows, n_cols = input_matrix.shape
    new_matrix = np.zeros_like(input_matrix)
    std_dev = np.std(input_matrix)
    mad = np.median(np.abs(input_matrix - np.median(input_matrix)))
    k = std_dev / mad

    for col_idx in range(n_cols):
        for ti in range(n_rows):
            start_idx = max(0, ti - window_size)
            end_idx = min(n_rows, ti + window_size)

            x0 = np.nanmedian(input_matrix[start_idx:end_idx, col_idx])
            s0 = k * np.nanmedian(np.abs(input_matrix[start_idx:end_idx, col_idx] - x0))

            if np.abs(input_matrix[ti, col_idx] - x0) > n_sigmas * s0:
                new_matrix[ti, col_idx] = x0
            else:
                new_matrix[ti, col_idx] = input_matrix[ti, col_idx]

    return pd.DataFrame(new_matrix)


def denoise(df, wavelet):
    dwt = pd.DataFrame()
    for i in range(len(df.columns)):
        sig = df.iloc[:, i]
        coeff = pywt.wavedec(sig, wavelet=wavelet, mode="per")

        d = np.std(coeff[-1])
        sigma = 4 * d
        uthresh = sigma * np.sqrt(2 * np.log(len(sig)))

        denoised_coeff = [coeff[0]]
        for c in coeff[1:]:
            denoised_coeff.append(pywt.threshold(c, value=uthresh, mode='soft'))

        denoised_signal = pywt.waverec(denoised_coeff, wavelet=wavelet, mode='per')
        dwt[i] = np.nan_to_num(denoised_signal)

    return dwt


def smooth(df, window_length=5, poly_order=2):
    from scipy.signal import savgol_filter
    return pd.DataFrame(savgol_filter(df, window_length, poly_order))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute CSI Doppler profiles from labeled CSI captures.")
    parser.add_argument('--input-dir', required=True,
                         help="Root folder containing per-class subfolders of CSI CSV captures.")
    parser.add_argument('--output-dir', required=True,
                         help="Folder where the computed Doppler profiles will be written.")
    parser.add_argument('--num-symbols', type=int, default=31, help="Sliding window length in symbols.")
    parser.add_argument('--sliding', type=int, default=1, help="Step size for the sliding window.")
    parser.add_argument('--noise-floor', type=float, default=-0.5, help="log10 noise floor clamp for the profile.")
    parser.add_argument('--num-subcarriers', type=int, default=15, help="Number of subcarriers in each capture.")
    return parser.parse_args()


def main():
    args = parse_args()

    num_symbols = args.num_symbols
    sliding = args.sliding
    noise_lev = args.noise_floor

    Tc = 6e-3
    fc = 2.4e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)

    classes = sorted(os.listdir(args.input_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join(args.input_dir, class_name)
        files = sorted(os.listdir(class_dir))

        out_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for fname in tqdm(files, desc=class_name):
            src_path = os.path.join(class_dir, fname)
            with open(src_path, 'rb') as f:
                csi_buff = pd.read_csv(f)

            if isinstance(csi_buff, pd.DataFrame):
                csi_buff = csi_buff.to_numpy()

            csi_buff = csi_buff.reshape((len(csi_buff), -1))
            if csi_buff.shape[0] < num_symbols:
                continue

            csi_buff = csi_buff[:, :-1]  # drop trailing label/timestamp column
            csi_buff = hampel_filter(csi_buff, 100)
            csi_buff = csi_buff.dropna()
            csi_buff = denoise(csi_buff, 'db4')
            csi_buff = smooth(csi_buff)

            csi_matrix = np.asarray(csi_buff)
            csi_matrix = csi_matrix.reshape(csi_matrix.shape[0], args.num_subcarriers, 2)
            csi_matrix = csi_matrix[1:-1, :, :]

            csi_matrix[:, :, 0] = csi_matrix[:, :, 0] / np.mean(csi_matrix[:, :, 0], axis=1, keepdims=True)
            csi_complex = csi_matrix[:, :, 0] * np.exp(1j * csi_matrix[:, :, 1])

            hann_window = np.expand_dims(hann(num_symbols), axis=-1)

            doppler_profiles = []
            for i in range(0, csi_complex.shape[0] - num_symbols, sliding):
                window = np.nan_to_num(csi_complex[i:i + num_symbols, :])
                windowed = window * hann_window

                doppler = fft(windowed, n=100, axis=0)
                doppler = fftshift(doppler, axes=0)

                d_map = np.abs(doppler * np.conj(doppler))
                d_map = np.sum(d_map, axis=1)
                doppler_profiles.append(d_map)

            doppler_profiles = np.asarray(doppler_profiles)
            doppler_profiles = doppler_profiles / np.max(doppler_profiles, axis=1, keepdims=True)
            doppler_profiles[doppler_profiles < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)

            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.txt')
            with open(out_path, "wb") as fp:
                pickle.dump(doppler_profiles, fp)


if __name__ == '__main__':
    main()
