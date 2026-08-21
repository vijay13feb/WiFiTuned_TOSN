"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from dataset_utility import create_dataset_single, expand_antennas
from network_utility import *
from tcn import TCN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue training a saved CSI activity-recognition model.')
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Comma-separated subdirs for training')
    parser.add_argument('feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument('sample_length', help='Length along the time dimension (width)', type=int)
    parser.add_argument('channels', help='Number of channels', type=int)
    parser.add_argument('batch_size', help='Number of samples in a batch', type=int)
    parser.add_argument('num_tot', help='Number of antennas * number of spatial streams', type=int)
    parser.add_argument('name_base', help='Name base for the cache/model/output files')
    parser.add_argument('activities', help='Comma-separated activities to be considered')
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                             '(default 80)', default=80, type=int)
    parser.add_argument('--sub_band', help='Sub-band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                            '(default 1)', default=1, type=int)
    parser.add_argument('--load-model', required=True, help='Path to the checkpoint to resume training from')
    parser.add_argument('--model-dir', default='models', help='Directory to save the updated model')
    parser.add_argument('--output-dir', default='outputs', help='Directory to save the evaluation metrics')
    parser.add_argument('--epochs', type=int, default=4, help='Number of additional training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = np.asarray(csi_act.split(','))
    name_base = args.name_base

    for cache_suffix in ['_cache_train', '_cache_val', '_cache_train_test', '_cache_test']:
        cache_prefix = name_base + '_' + str(csi_act) + cache_suffix
        if os.path.exists(cache_prefix + '.data-00000-of-00001'):
            os.remove(cache_prefix + '.data-00000-of-00001')
            os.remove(cache_prefix + '.index')

    subdirs_training = args.subdirs
    labels_train, all_files_train = [], []
    labels_val, all_files_val = [], []
    labels_test, all_files_test = [], []

    sample_length = args.sample_length
    feature_length = args.feature_length
    channels = args.channels
    num_antennas = args.num_tot
    input_network = (sample_length, feature_length, channels)
    batch_size = args.batch_size

    output_shape = activities.shape[0]
    labels_considered = np.arange(output_shape)
    activities = activities[labels_considered]

    suffix = '.txt'

    for sdir in subdirs_training.split(','):
        exp_dir = args.dir + sdir + '/'

        with open(exp_dir + 'labels_train_antennas_' + str(csi_act) + suffix, "rb") as fp:
            labels_train.extend(pickle.load(fp))
        with open(exp_dir + 'files_train_antennas_' + str(csi_act) + suffix, "rb") as fp:
            all_files_train.extend(pickle.load(fp))

        with open(exp_dir + 'labels_val_antennas_' + str(csi_act) + suffix, "rb") as fp:
            labels_val.extend(pickle.load(fp))
        with open(exp_dir + 'files_val_antennas_' + str(csi_act) + suffix, "rb") as fp:
            all_files_val.extend(pickle.load(fp))

        with open(exp_dir + 'labels_test_antennas_' + str(csi_act) + suffix, "rb") as fp:
            labels_test.extend(pickle.load(fp))
        with open(exp_dir + 'files_test_antennas_' + str(csi_act) + suffix, "rb") as fp:
            all_files_test.extend(pickle.load(fp))

    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train))
                            if labels_train[idx] in labels_considered]
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train))
                              if labels_train[idx] in labels_considered]
    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected, labels_train_selected, num_antennas)

    name_cache = name_base + '_' + str(csi_act) + '_cache_train'
    dataset_csi_train = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                               stream_ant_train, input_network, batch_size,
                                               shuffle=True, cache_file=name_cache)

    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val))
                         if labels_val[idx] in labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val))
                           if labels_val[idx] in labels_considered]
    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas)

    name_cache_val = name_base + '_' + str(csi_act) + '_cache_val'
    dataset_csi_val = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                             stream_ant_val, input_network, batch_size,
                                             shuffle=False, cache_file=name_cache_val)

    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test))
                          if labels_test[idx] in labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test))
                            if labels_test[idx] in labels_considered]
    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas)

    name_cache_test = name_base + '_' + str(csi_act) + '_cache_test'
    dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                              stream_ant_test, input_network, batch_size,
                                              shuffle=False, cache_file=name_cache_test)

    num_samples_train = len(file_train_selected_expanded)
    num_samples_val = len(file_val_selected_expanded)
    num_samples_test = len(file_test_selected_expanded)
    train_steps_per_epoch = int(np.ceil(num_samples_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val / batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test / batch_size))

    tf.random.set_seed(42)
    np.random.seed(42)

    # loaded checkpoint already carries its compiled optimizer/loss state, no need to recompile
    with tf.keras.utils.custom_object_scope({'TCN': TCN}):
        csi_model = tf.keras.models.load_model(args.load_model)

    callback_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy', patience=args.patience)
    name_model = os.path.join(args.model_dir, f'{name_base}_{args.epochs}ep.h5')
    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                        monitor='val_sparse_categorical_accuracy')

    csi_model.fit(dataset_csi_train, epochs=args.epochs, steps_per_epoch=train_steps_per_epoch,
                  validation_data=dataset_csi_val, validation_steps=val_steps_per_epoch,
                  callbacks=[callback_save, callback_stop])

    csi_model.save(name_model, save_format='tf')

    train_labels_true = np.array(labels_train_selected_expanded)
    name_cache_train_test = name_base + '_' + str(csi_act) + '_cache_train_test'
    dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                    stream_ant_train, input_network, batch_size,
                                                    shuffle=False, cache_file=name_cache_train_test, prefetch=False)
    train_prediction_list = csi_model.predict(
        dataset_csi_train_test, steps=train_steps_per_epoch)[:train_labels_true.shape[0]]
    train_labels_pred = np.argmax(train_prediction_list, axis=1)
    conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)

    val_labels_true = np.array(labels_val_selected_expanded)
    val_prediction_list = csi_model.predict(dataset_csi_val, steps=val_steps_per_epoch)[:val_labels_true.shape[0]]
    val_labels_pred = np.argmax(val_prediction_list, axis=1)
    conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)

    test_labels_true = np.array(labels_test_selected_expanded)
    test_prediction_list = csi_model.predict(dataset_csi_test, steps=test_steps_per_epoch)[:test_labels_true.shape[0]]
    test_labels_pred = np.argmax(test_prediction_list, axis=1)
    conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_labels_true, test_labels_pred, labels=labels_considered)
    accuracy = accuracy_score(test_labels_true, test_labels_pred)

    # merge antennas belonging to the same sample via majority vote / summed softmax
    labels_true_merge = np.array(labels_test_selected)
    pred_max_merge = np.zeros_like(labels_test_selected)
    for i_lab in range(len(labels_test_selected)):
        pred_antennas = test_prediction_list[i_lab * num_antennas:(i_lab + 1) * num_antennas, :]
        lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

        pred_max_antennas = test_labels_pred[i_lab * num_antennas:(i_lab + 1) * num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
        if lab_unique.shape[0] > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]
            if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:  # ex aequo between two labels
                lab_max_merge = lab_merge_max
            else:
                lab_max_merge = lab_unique_sort[0]
        else:
            lab_max_merge = lab_unique[0]
        pred_max_merge[i_lab] = lab_max_merge

    conf_matrix_max_merge = confusion_matrix(labels_true_merge, pred_max_merge, labels=labels_considered)
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = precision_recall_fscore_support(
        labels_true_merge, pred_max_merge, labels=labels_considered, zero_division=1)
    accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)

    print('Accuracy:', accuracy, 'Max merge accuracy:', accuracy_max_merge)
    print('Training confusion matrix:\n', conf_matrix_train)
    print('Validation confusion matrix:\n', conf_matrix_val)
    print('Test confusion matrix:\n', conf_matrix)
    print('Merged confusion matrix:\n', conf_matrix_max_merge)

    metrics_matrix_dict = {'conf_matrix': conf_matrix,
                            'accuracy_single': accuracy,
                            'precision_single': precision,
                            'recall_single': recall,
                            'fscore_single': fscore,
                            'conf_matrix_max_merge': conf_matrix_max_merge,
                            'accuracy_max_merge': accuracy_max_merge,
                            'precision_max_merge': precision_max_merge,
                            'recall_max_merge': recall_max_merge,
                            'fscore_max_merge': fscore_max_merge}

    name_file = os.path.join(args.output_dir, f'training_{csi_act}_{subdirs_training}_{name_base}{suffix}')
    with open(name_file, "wb") as fp:
        pickle.dump(metrics_matrix_dict, fp)
