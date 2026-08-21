import argparse
import glob
import os
import numpy as np
import pickle
import math as mt
import shutil
from dataset_utility import create_windows_antennas, convert_to_number


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build train/val/test windows from Doppler traces.')
    parser.add_argument('dir', help='Directory of data')  # ./doppler_traces/
    parser.add_argument('subdirs', help='Sub-directories')  # S1a
    parser.add_argument('sample_lengths', help='Number of packets in a sample', type=int)  # 31
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)  # 1
    parser.add_argument('window_length', help='Number of samples per window', type=int)  # 340
    parser.add_argument('stride_length', help='Number of samples to stride', type=int)  # 30
    parser.add_argument('labels_activities', help='Labels of the activities to be considered')
    parser.add_argument('n_tot', help='Number of streams * number of antennas', type=int)  # 1
    parser.add_argument('start_with', help='start_with the file name')
    args = parser.parse_args()

    labels_activities = args.labels_activities  # e.g. A,B,C,D,F
    csi_label_dict = labels_activities.split(',')
    activities = np.asarray(labels_activities)

    n_tot = args.n_tot
    num_packets = args.sample_lengths
    middle = int(np.floor(num_packets / 2))
    list_subdir = args.subdirs

    for subdir in list_subdir.split(','):
        exp_dir = args.dir + subdir + '/'

        path_train = exp_dir + 'train_antennas_' + str(activities)
        path_val = exp_dir + 'val_antennas_' + str(activities)
        path_test = exp_dir + 'test_antennas_' + str(activities)
        paths = [path_train, path_val, path_test]

        for pat in paths:
            if os.path.exists(pat):
                for f in glob.glob(pat + '/*'):
                    os.remove(f)
            else:
                os.mkdir(pat)

        path_complete = exp_dir + 'complete_antennas_' + str(activities)
        if os.path.exists(path_complete):
            shutil.rmtree(path_complete)

        all_files = os.listdir(exp_dir)
        names = []
        for i in all_files:
            if "Forward" in i or "Looking" in i or "Nodding" in i or "Shaking" in i:
                names.append(i[:-4])
        names.sort()  # files like S1a_F_stream_1.txt sort by activity/stream

        csi_matrices = []
        labels = []
        lengths = []
        label = 'null'
        prev_label = label
        csi_matrix = []
        processed = False

        for i_name, name in enumerate(names):
            if i_name % n_tot == 0 and i_name != 0 and processed:
                ll = csi_matrix[0].shape[1]
                for i_ant in range(1, n_tot):
                    if ll != csi_matrix[i_ant].shape[1]:
                        break
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                csi_matrix = []

            label = csi_label_dict[i_name]

            if label not in csi_label_dict:
                processed = False
                continue
            processed = True

            label = convert_to_number(label, csi_label_dict)

            if i_name % n_tot == 0:
                prev_label = label
            elif label != prev_label:
                print('error in ' + str(name))
                break

            name_file = exp_dir + name + '.txt'
            with open(name_file, "rb") as fp:
                stft_sum_1 = pickle.load(fp)

            stft_sum_1_mean = stft_sum_1 - np.mean(stft_sum_1, axis=0, keepdims=True)
            csi_matrix.append(stft_sum_1_mean.T)

        error = False
        if processed:
            # flush the last block
            if len(csi_matrix) < n_tot:
                print('error in ' + str(name))
            ll = csi_matrix[0].shape[1]
            for i_ant in range(1, n_tot):
                if ll != csi_matrix[i_ant].shape[1]:
                    print('error in ' + str(name))
                    error = True
            if not error:
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)

        if error:
            continue

        lengths = np.asarray(lengths)
        csi_train = []
        csi_val = []
        csi_test = []
        length_train = []
        length_val = []
        length_test = []

        for i in range(len(labels)):
            ll = lengths[i]
            train_len = int(np.floor(ll * 0.6))
            length_train.append(train_len)
            csi_train.append(csi_matrices[i][:, :, :train_len])

            start_val = train_len + mt.ceil(num_packets / 1)
            val_len = int(np.floor(ll * 0.2))
            length_val.append(val_len)
            csi_val.append(csi_matrices[i][:, :, start_val:start_val + val_len])

            start_test = start_val + val_len + mt.ceil(num_packets / 1)
            length_test.append(ll - val_len - train_len - 2 * mt.ceil(num_packets / 1))
            csi_test.append(csi_matrices[i][:, :, start_test:])

        print('length_train:', length_train, 'sum:', sum(length_train))
        print('length_val:', length_val, 'sum:', sum(length_val))
        print('length_test:', length_test, 'sum:', sum(length_test))

        list_sets_name = ['train', 'val', 'test']
        list_sets = [csi_train, csi_val, csi_test]
        list_sets_lengths = [length_train, length_val, length_test]

        for set_idx in range(3):
            csi_matrices_set, labels_set = create_windows_antennas(
                list_sets[set_idx], labels, args.window_length, args.stride_length, remove_mean=False)

            num_windows = np.floor(
                (np.asarray(list_sets_lengths[set_idx]) - args.window_length) / args.stride_length + 1)

            names_set = []
            suffix = '.txt'
            for ii in range(len(csi_matrices_set)):
                name_file = exp_dir + list_sets_name[set_idx] + '_antennas_' + str(activities) + '/' + str(ii) + suffix
                names_set.append(name_file)
                with open(name_file, "wb") as fp:
                    pickle.dump(csi_matrices_set[ii], fp)

            name_labels = exp_dir + '/labels_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
            with open(name_labels, "wb") as fp:
                pickle.dump(labels_set, fp)

            name_f = exp_dir + '/files_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
            with open(name_f, "wb") as fp:
                pickle.dump(names_set, fp)

            name_f = exp_dir + '/num_windows_' + list_sets_name[set_idx] + '_antennas_' + str(activities) + suffix
            with open(name_f, "wb") as fp:
                pickle.dump(num_windows, fp)
